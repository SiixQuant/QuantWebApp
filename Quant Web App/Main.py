from __future__ import print_function
import time
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import base64
import yfinance as yf

class MarketData:
    def __init__(self):
        print("Initializing Market Data API...")
        
    def get_price_data(self, ticker, months=6):
        print(f"\nStarting get_price_data for {ticker}")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30*months)
        
        try:
            # Convert SPX to ^GSPC for S&P 500 index
            if ticker == 'SPX':
                ticker = '^GSPC'
                
            print(f"Requesting data for {ticker} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            
            # Get data from yfinance
            data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                progress=False
            )
            
            if not data.empty:
                # Rename columns to match expected format
                data = data.rename(columns={
                    'Close': 'close',
                    'Volume': 'volume',
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Adj Close': 'adj_close'
                })
                print(f"Successfully got data for {ticker}")
                print(f"Data shape: {data.shape}")
                print(f"Date range: {data.index[0]} to {data.index[-1]}")
                print(f"Columns: {data.columns.tolist()}")
                return data
            else:
                print(f"No data returned for {ticker}")
                return None
                
        except Exception as e:
            print(f"Error fetching data for {ticker}: {str(e)}")
            return None

def get_zscore(data):
    """Calculate Z-score for price data"""
    try:
        # Make sure we're using the correct column name
        if 'close' not in data.columns:
            print("Warning: 'close' column not found. Available columns:", data.columns)
            return None
            
        # Calculate returns and z-score
        returns = data['close'].pct_change()
        mean = returns.mean()
        std = returns.std()
        latest_return = returns.iloc[-1]
        zscore = (latest_return - mean) / std
        
        return zscore
        
    except Exception as e:
        print(f"Error calculating Z-score: {str(e)}")
        return None

def create_chart(api, ticker, months=6):
    """Create an interactive line chart with Z-score levels"""
    data = api.get_price_data(ticker, months)
    if data is None:
        return None
    
    # Calculate Z-score
    zscore = get_zscore(data)
    
    # Create the line chart
    fig = go.Figure()
    
    # Add Z-score levels
    z_levels = [-2, -1, 0, 1, 2]
    colors = ['red', 'orange', 'white', 'orange', 'red']
    
    for z, color in zip(z_levels, colors):
        fig.add_hline(
            y=data['close'].mean() + (z * data['close'].std()),
            line_dash="dash",
            line_color=color,
            opacity=0.5,
            annotation_text=f'Z={z}',
            annotation_position="right",
            line_width=1
        )
    
    # Add price line
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['close'],
        name='Price',
        line=dict(color='blue', width=1.5)
    ))
    
    # Add current Z-score annotation
    current_zscore = zscore
    fig.add_annotation(
        text=f'Current Z-score: {current_zscore:.2f}',
        xref="paper", yref="paper",
        x=1, y=1,
        showarrow=False,
        font=dict(size=12, color="white"),
        bgcolor="rgba(0,0,0,0.8)",
        bordercolor="white",
        borderwidth=1
    )
    
    fig.update_layout(
        title=f'{ticker} Price Chart with Z-score Levels',
        yaxis_title='Price',
        template='plotly_dark',
        height=800,
        showlegend=True,
        hovermode='x unified'
    )
    
    return fig

def get_signals(api, months=6):
    """Get signals with modified timeframe"""
    tickers = ['SPX', 'SPY']
    signals = []
    charts = {}
    
    for ticker in tickers:
        print(f"\n{'='*50}")
        print(f"Processing ticker: {ticker}")
        data = api.get_price_data(ticker, months)
        
        if data is None or data.empty:
            print(f"No data returned for {ticker}")
            continue
            
        # Calculate Z-score
        zscore = get_zscore(data)
        if zscore is not None and abs(zscore) >= 2.0:
            signals.append({
                'ticker': ticker,
                'zscore': round(zscore, 2),
                'date': data.index[-1].strftime('%Y-%m-%d')
            })
            
        # Create chart
        charts[ticker] = create_chart(api, ticker, months)
    
    return signals, charts

def display_analysis(months=6):
    """Display analysis with detailed logging"""
    print("\n=== Starting Analysis ===")
    print("Initializing Market Data API...")
    
    api = MarketData()
    print("API initialized")
    
    print("\nFetching signals and creating charts...")
    signals, charts = get_signals(api, months)
    
    if not signals:
        print("\nNo significant signals found (Z-score threshold: ±2.0)")
    else:
        df = pd.DataFrame(signals)
        print("\n=== Z-Score Signal Leaderboard ===")
        print("Threshold: |Z-Score| >= 2.0")
        print("\n", df.to_string(index=False))
    
    for ticker, fig in charts.items():
        if fig is not None:
            fig.show()

if __name__ == "__main__":
    print("\n=== Starting Program ===")
    print(f"Current time: {datetime.now()}")
    display_analysis(months=6)
