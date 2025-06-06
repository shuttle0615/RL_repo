"""
Download BTCUSDT futures OHLCV data from Binance using ccxt.
Automatically finds the earliest available data.

Requirements:
pip install ccxt pandas
"""

import ccxt
import pandas as pd
from datetime import datetime, timedelta
import time
from typing import List, Optional
import os

def find_earliest_data(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str
) -> Optional[datetime]:
    """
    Find the earliest available data for a symbol by binary search.
    
    Args:
        exchange: CCXT exchange instance
        symbol: Trading pair symbol
        timeframe: Candlestick timeframe
    
    Returns:
        datetime of earliest available data or None if not found
    """
    print(f"Finding earliest available data for {symbol}...")
    
    # Start with a wide range: 2005 to now
    start = datetime(2005, 1, 1)
    end = datetime.now()
    earliest_valid = None
    
    while (end - start).days > 1:
        mid_date = start + (end - start) // 2
        since = int(mid_date.timestamp() * 1000)
        
        try:
            ohlcv = exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=since,
                limit=1
            )
            time.sleep(exchange.rateLimit / 1000)
            
            if len(ohlcv) > 0:
                # Data exists at this date, try earlier
                earliest_valid = mid_date
                end = mid_date
            else:
                # No data, try later
                start = mid_date
                
        except Exception as e:
            print(f"Error during search: {e}")
            time.sleep(2)
            start = mid_date  # Move forward on error
    
    if earliest_valid:
        # Do a final check slightly before the found date
        buffer_date = earliest_valid - timedelta(days=5)
        since = int(buffer_date.timestamp() * 1000)
        
        try:
            ohlcv = exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=since,
                limit=1
            )
            if len(ohlcv) > 0:
                earliest_valid = datetime.fromtimestamp(ohlcv[0][0]/1000)
        except:
            pass
            
    return earliest_valid

def fetch_ohlcv_data(
    symbol: str = 'BTC/USDT',
    timeframe: str = '1h',
    save_path: str = 'data'
) -> None:
    """
    Fetch complete OHLCV history from Binance Futures and save to CSV.
    
    Args:
        symbol: Trading pair symbol
        timeframe: Candlestick timeframe
        save_path: Directory to save the CSV file
    """
    
    # Initialize Binance Futures client
    exchange = ccxt.binance({
        'options': {
            'defaultType': 'future',
        }
    })
    
    # Find earliest available data
    earliest_date = find_earliest_data(exchange, symbol, timeframe)
    
    if not earliest_date:
        print("Could not find any historical data!")
        return
        
    print(f"Earliest available data: {earliest_date}")
    
    # Create save directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Start from the earliest available date
    since = int(earliest_date.timestamp() * 1000)
    
    # Initialize variables
    all_ohlcv: List = []
    limit = 1000  # Number of candles per request
    
    print(f"\nDownloading {symbol} {timeframe} data from {earliest_date}...")
    
    while True:
        try:
            # Fetch OHLCV data
            ohlcv = exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=since,
                limit=limit
            )
            
            if len(ohlcv) == 0:
                break
                
            # Extend our list of candles
            all_ohlcv.extend(ohlcv)
            
            # Update since timestamp for next iteration
            since = ohlcv[-1][0] + 1
            
            # Print progress
            last_date = datetime.fromtimestamp(ohlcv[-1][0]/1000).strftime('%Y-%m-%d %H:%M:%S')
            print(f"Downloaded data up to {last_date}")
            
            # Check if we've reached current time
            if ohlcv[-1][0] >= int(datetime.now().timestamp() * 1000):
                break
                
            # Rate limiting
            time.sleep(exchange.rateLimit / 1000)
            
        except Exception as e:
            print(f"Error occurred: {e}")
            time.sleep(60)  # Wait longer on error
            continue
    
    if not all_ohlcv:
        print("No data was downloaded!")
        return
        
    # Convert to DataFrame
    df = pd.DataFrame(
        all_ohlcv,
        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
    )
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Save to CSV
    filename = f"{symbol.replace('/', '')}_{timeframe}_full_history.csv"
    filepath = os.path.join(save_path, filename)
    df.to_csv(filepath, index=False)
    
    print(f"\nData saved to {filepath}")
    print(f"Total candles downloaded: {len(df)}")
    print(f"Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")

if __name__ == "__main__":
    # Example usage
    fetch_ohlcv_data(
        symbol='BTC/USDT',
        timeframe='15md',
        save_path='data'
    ) 