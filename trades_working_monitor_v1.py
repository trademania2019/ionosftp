import os
import pandas as pd
import requests
import logging
from datetime import datetime, timedelta
from flask_mail import Mail, Message
from dotenv import load_dotenv
from pytz import timezone
from flaskapp_predict_trades import db, NotificationJob, User, OpenPosition, ClosedPosition, app  # Import the app from your main Flask app

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG,  
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.StreamHandler()
                    ])

# Initialize Flask Mail
app.config['MAIL_SERVER'] = os.getenv('MAIL_SERVER')
app.config['MAIL_PORT'] = int(os.getenv('MAIL_PORT'))
app.config['MAIL_USE_TLS'] = os.getenv('MAIL_USE_TLS') == 'True'
app.config['MAIL_USE_SSL'] = False
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
app.secret_key = os.getenv('SECRET_KEY')

mail = Mail(app)

# Configure the data directory and API token
DATA_DIR = os.getenv('DATA_DIR')
API_TOKEN = os.getenv('API_TOKEN')

# Define interval to time tolerance mapping
INTERVAL_TO_TOLERANCE = {
    '5min': timedelta(minutes=5),
    '15min': timedelta(minutes=15),
    '60min': timedelta(minutes=60),
    '1D': timedelta(days=1)
}

# Helper to get time tolerance based on the interval
def get_time_tolerance(interval):
    return INTERVAL_TO_TOLERANCE.get(interval, timedelta(minutes=5))  # Default to 5min

# Ensure both datetimes are timezone-aware
def is_within_tolerance(existing_position, new_position, time_tolerance):
    # Ensure both datetimes are timezone-aware
    existing_position_time = existing_position.entry_date_time
    new_position_time = new_position['Entry_date_time']

    # Convert tz-naive datetimes to tz-aware in UTC
    if existing_position_time.tzinfo is None:
        existing_position_time = existing_position_time.replace(tzinfo=timezone('UTC'))
    if new_position_time.tzinfo is None:
        new_position_time = new_position_time.replace(tzinfo=timezone('UTC'))

    # Time tolerance check
    time_diff = abs(existing_position_time - new_position_time)
    if time_diff > time_tolerance:
        return False

    # Price tolerance check (1% price difference)
    PRICE_TOLERANCE_PERCENT = 0.01
    price_diff = abs(existing_position.entry_price - new_position['Entry_price'])
    if price_diff / existing_position.entry_price > PRICE_TOLERANCE_PERCENT:
        return False

    return True

# Fetch stock data
def get_stock_data(symbol, interval):
    try:
        logging.debug(f"Fetching stock data for symbol={symbol} and interval={interval}")
        DURATION_MAPPING = {'5min': '5', '15min': '15', '60min': '60', '1D': '1D'}
        days = 30 if interval != '1D' else 360
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        start_date_str = start_date.strftime('%Y-%m-%dT%H:%M:%SZ')
        end_date_str = end_date.strftime('%Y-%m-%dT%H:%M:%SZ')
        api_interval = DURATION_MAPPING.get(interval)

        url = f"https://api.marketdata.app/v1/stocks/candles/{api_interval}/{symbol}?from={start_date_str}&to={end_date_str}&token={API_TOKEN}"
        response = requests.get(url, headers={'Accept': 'application/json'})
        if response.status_code == 200:
            logging.info(f"Successfully fetched stock data for {symbol} at interval {interval}")
            return response.json()
        else:
            logging.error(f"Failed to retrieve data: {response.status_code} {response.text}")
            return None
    except Exception as e:
        logging.error(f"Exception while fetching stock data: {e}")
        return None

# Compute SMI indicator
def compute_smi(df, period=14, smooth_k=3, smooth_d=3):
    logging.debug("Computing SMI indicator")
    df['max_high'] = df['high'].rolling(window=period).max()
    df['min_low'] = df['low'].rolling(window=period).min()
    df['midpoint'] = (df['max_high'] + df['min_low']) / 2
    df['diff'] = df['max_high'] - df['min_low']
    df['smi_raw'] = (df['close'] - df['midpoint']) / (df['diff'] / 2) * 100
    df['SMI'] = df['smi_raw'].rolling(window=smooth_k).mean()
    df['SMI_Signal'] = df['SMI'].rolling(window=smooth_d).mean()
    return df['SMI'], df['SMI_Signal']

# Determine buy and sell signals
def determine_signals(df):
    logging.debug("Determining buy and sell signals")
    df['SMI_Change'] = df['SMI'].diff()
    df['Buy_Signal'] = (df['SMI'] < -20) & (df['SMI_Change'] > 0)
    df['Sell_Signal'] = (df['SMI'] > 20) & (df['SMI_Change'] < 0)
    return df

# Evaluate performance based on signals
def evaluate_performance(df, interval, trade_size, symbol):
    long_trades = []
    short_trades = []
    open_positions = []

    long_entry_price = None
    short_entry_price = None
    long_entry_time = None
    short_entry_time = None

    for i in range(1, len(df)):
        # Long Trades
        if df['Buy_Signal'].iloc[i-1]:
            long_entry_price = round(df['close'].iloc[i-1], 2)
            long_entry_time = df.index[i-1]
        if df['Sell_Signal'].iloc[i-1] and long_entry_price is not None:
            exit_price = round(df['close'].iloc[i], 2)
            exit_time = df.index[i]
            profit = round((exit_price - long_entry_price) * trade_size, 2)
            status = 'Win' if profit > 0 else 'Loss'
            long_trades.append([symbol, interval, long_entry_time, long_entry_price, exit_time, exit_price, profit, status])
            long_entry_price = None

        # Short Trades
        if df['Sell_Signal'].iloc[i-1]:
            short_entry_price = round(df['close'].iloc[i-1], 2)
            short_entry_time = df.index[i-1]
        if df['Buy_Signal'].iloc[i-1] and short_entry_price is not None:
            exit_price = round(df['close'].iloc[i], 2)
            exit_time = df.index[i]
            profit = round((short_entry_price - exit_price) * trade_size, 2)
            status = 'Win' if profit > 0 else 'Loss'
            short_trades.append([symbol, interval, short_entry_time, short_entry_price, exit_time, exit_price, profit, status])
            short_entry_price = None

    # Handle open positions at the end of the script execution
    if long_entry_price is not None:
        open_positions.append([symbol, interval, long_entry_time, long_entry_price, 'open', 'long'])
    if short_entry_price is not None:
        open_positions.append([symbol, interval, short_entry_time, short_entry_price, 'open', 'short'])

    long_df = pd.DataFrame(long_trades, columns=['symbol', 'Interval', 'Entry_date_time', 'Entry_price', 'Exit_date_time', 'Exit_price', 'Profit', 'Status'])
    short_df = pd.DataFrame(short_trades, columns=['symbol', 'Interval', 'Entry_date_time', 'Entry_price', 'Exit_date_time', 'Exit_price', 'Profit', 'Status'])
    open_df = pd.DataFrame(open_positions, columns=['symbol', 'Interval', 'Entry_date_time', 'Entry_price', 'Status', 'Type'])

    return long_df, short_df, open_df

# Send notification emails for open positions
def notify_open_positions(user_id, user_email, open_df, symbol, interval):
    if not open_df.empty:
        time_tolerance = get_time_tolerance(interval)

        for index, row in open_df.iterrows():
            existing_positions = OpenPosition.query.filter_by(
                user_id=user_id,
                symbol=symbol,
                interval=interval,
                status='open',
                type=row['Type']
            ).all()

            is_duplicate = False
            for existing_position in existing_positions:
                if is_within_tolerance(existing_position, row, time_tolerance):
                    logging.info(f"Open position for {symbol} - {interval} within tolerance, skipping email.")
                    is_duplicate = True
                    break

            if not is_duplicate:
                new_position = OpenPosition(
                    user_id=user_id,
                    symbol=symbol,
                    interval=interval,
                    entry_date_time=row['Entry_date_time'],
                    entry_price=row['Entry_price'],
                    status='open',
                    type=row['Type'],
                    email_sent=True
                )
                db.session.add(new_position)
                db.session.commit()

                message_body = (
                    f"New trade opened for {symbol}:\n"
                    f"Type: {row['Type']}\n"
                    f"Entry Time: {row['Entry_date_time']}\n"
                    f"Entry Price: {row['Entry_price']}\n"
                )
                send_notification_email(user_email, f"New Trade Alert: {symbol} - {interval}", message_body)

# Send notification emails for closed positions
def notify_trade_closures(user_id, user_email, long_df, short_df, symbol, interval):
    for df, trade_type in zip([long_df, short_df], ['long', 'short']):
        if not df.empty:
            for index, row in df.iterrows():
                existing_position = OpenPosition.query.filter_by(
                    user_id=user_id,
                    symbol=symbol,
                    interval=interval,
                    entry_date_time=row['Entry_date_time'],
                    status='open',
                    type=trade_type
                ).first()

                if existing_position is not None:
                    try:
                        existing_position.status = 'closed'
                        existing_position.exit_date_time = row['Exit_date_time']
                        existing_position.exit_price = row['Exit_price']
                        existing_position.profit = row['Profit']
                        existing_position.trade_status = row['Status']
                        db.session.commit()

                        closed_position = ClosedPosition(
                            user_id=user_id,
                            symbol=symbol,
                            interval=interval,
                            entry_date_time=row['Entry_date_time'],
                            entry_price=row['Entry_price'],
                            exit_date_time=row['Exit_date_time'],
                            exit_price=row['Exit_price'],
                            profit=row['Profit'],
                            trade_status=row['Status'],
                            type=trade_type
                        )
                        db.session.add(closed_position)
                        db.session.commit()

                        message_body = (
                            f"Trade closed for {symbol}:\n"
                            f"Type: {trade_type.capitalize()}\n"
                            f"Entry Time: {row['Entry_date_time']}\n"
                            f"Entry Price: {row['Entry_price']}\n"
                            f"Exit Time: {row['Exit_date_time']}\n"
                            f"Exit Price: {row['Exit_price']}\n"
                            f"Profit: {row['Profit']}\n"
                            f"Status: {row['Status']}\n"
                        )
                        send_notification_email(user_email, f"Trade Closed: {symbol} - {interval}", message_body)
                        logging.info(f"Trade closure email sent for {symbol} at {interval}")
                    except Exception as e:
                        logging.error(f"Failed to process trade closure: {e}")
                        db.session.rollback()
                else:
                    logging.debug(f"No matching open position found for closure for {symbol} at {interval}")

# Monitor trades
def monitor_trades():
    with app.app_context():
        try:
            logging.info("Starting trade monitoring")
            active_jobs = NotificationJob.query.filter_by(active=True).all()

            for job in active_jobs:
                user_id = job.user_id
                user = User.query.get(user_id)
                if user is None:
                    logging.error(f"No user found with ID: {user_id}")
                    continue

                user_email = user.email
                symbol = job.symbol
                interval = job.interval

                logging.info(f"Processing job for user_id={user_id}, symbol={symbol}, interval={interval}")

                df = process_interval(symbol, interval)
                if df is None:
                    logging.error(f"No data processed for {symbol} at {interval}")
                    continue

                long_df, short_df, open_df = evaluate_performance(df, interval, trade_size=1, symbol=symbol)

                notify_open_positions(user_id, user_email, open_df, symbol, interval)
                notify_trade_closures(user_id, user_email, long_df, short_df, symbol, interval)

                logging.info(f"Long positions detected: {len(long_df)}")
                logging.info(f"Short positions detected: {len(short_df)}")
                logging.info(f"Open positions detected: {len(open_df)}")

        except Exception as e:
            logging.error(f"Error executing script: {str(e)}")

if __name__ == "__main__":
    with app.app_context():
        monitor_trades()
