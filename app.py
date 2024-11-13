import streamlit as st
import pandas as pd
import numpy as np
import requests
import logging
import uuid
from datetime import datetime, timedelta
from requests_oauthlib import OAuth2Session
from urllib.parse import quote
from oauthlib.oauth2 import BackendApplicationClient

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Schwab API Configuration
SCHWAB_CONFIG = {
    'client_id': 'f6MOF1oqGHpQC6sZPCvTPRe7nyMWDgof',
    'client_secret': 'NCiMvAdTarXnDcIq',
    'base_url': 'https://api.schwab.com/v1',
    'token_url': 'https://api.schwab.com/v1/oauth2/token',
    'auth_url': 'https://api.schwab.com/v1/oauth2/authorize',
    'redirect_uri': 'https://quantwebapp-bjbqck9aebxpadmg6h7pmk.streamlit.app/',
    'scope': ['trading', 'quotes', 'accounts', 'margin']
}

class SchwabAPI:
    def __init__(self):
        self.client = BackendApplicationClient(client_id=SCHWAB_CONFIG['client_id'])
        self.oauth = OAuth2Session(client=self.client)
        self.token = None
        self.authenticate()

    def authenticate(self):
        """Authenticate with Schwab API using client credentials flow"""
        try:
            # Encode credentials properly
            auth = (SCHWAB_CONFIG['client_id'], SCHWAB_CONFIG['client_secret'])
            
            # Get token using client credentials grant
            self.token = self.oauth.fetch_token(
                token_url=SCHWAB_CONFIG['token_url'],
                auth=auth,
                client_id=SCHWAB_CONFIG['client_id'],
                client_secret=SCHWAB_CONFIG['client_secret'],
                include_client_id=True
            )
            
            st.success("Successfully authenticated with Schwab API")
            return True
            
        except Exception as e:
            st.error(f"Authentication failed: {str(e)}")
            logger.error(f"Authentication error: {str(e)}")
            return False

    def get_headers(self):
        """Get headers for API requests"""
        if not self.token:
            return None
            
        return {
            'Authorization': f"Bearer {self.token['access_token']}",
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'X-Security-Token': self.token.get('access_token'),
            'Client-Id': SCHWAB_CONFIG['client_id']
        }

    def refresh_token_if_needed(self):
        """Refresh token if expired"""
        if self.token and self.oauth.token.is_expired():
            try:
                self.token = self.oauth.refresh_token(
                    SCHWAB_CONFIG['token_url'],
                    client_id=SCHWAB_CONFIG['client_id'],
                    client_secret=SCHWAB_CONFIG['client_secret']
                )
                return True
            except Exception as e:
                logger.error(f"Token refresh error: {str(e)}")
                return False
        return True

    def get_market_data(self, symbols, fields=None):
        """Get real-time market data"""
        if not fields:
            fields = ['bid', 'ask', 'last', 'volume', 'high', 'low', 'open']
            
        endpoint = f"{SCHWAB_CONFIG['base_url']}/markets/quotes"
        params = {
            'symbols': ','.join(symbols) if isinstance(symbols, list) else symbols,
            'fields': ','.join(fields)
        }
        
        try:
            response = self.session.get(
                endpoint,
                headers=self.get_headers(),
                params=params
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Market data error: {str(e)}")
            return None

    def get_historical_data(self, symbol, start_date, end_date=None):
        """Get historical price data"""
        if not end_date:
            end_date = datetime.now()
            
        endpoint = f"{SCHWAB_CONFIG['base_url']}/markets/history/{symbol}"
        params = {
            'start': start_date.strftime('%Y-%m-%d'),
            'end': end_date.strftime('%Y-%m-%d'),
            'frequency': 'daily'
        }
        
        try:
            response = self.session.get(
                endpoint,
                headers=self.get_headers(),
                params=params
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Historical data error: {str(e)}")
            return None

    def get_option_chain(self, symbol):
        """Get option chain data"""
        endpoint = f"{SCHWAB_CONFIG['base_url']}/markets/options/{symbol}"
        
        try:
            response = self.session.get(
                endpoint,
                headers=self.get_headers()
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Option chain error: {str(e)}")
            return None

    def get_account_positions(self):
        """Get account positions"""
        endpoint = f"{SCHWAB_CONFIG['base_url']}/accounts/positions"
        
        try:
            response = self.session.get(
                endpoint,
                headers=self.get_headers()
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Account positions error: {str(e)}")
            return None

def process_market_data(api, symbols, months):
    """Process market data for analysis"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=months*30)
    
    results = []
    for symbol in symbols:
        try:
            # Get historical data
            hist_data = api.get_historical_data(symbol, start_date, end_date)
            if hist_data:
                df = pd.DataFrame(hist_data['candles'])
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                
                # Get current quote
                quote = api.get_market_data(symbol)
                if quote:
                    current_price = quote['quotes'][0]['last']
                    
                    results.append({
                        'Symbol': symbol,
                        'Current Price': current_price,
                        'Change %': ((current_price / df['close'].iloc[-2] - 1) * 100).round(2),
                        'Volume': quote['quotes'][0]['volume'],
                        'High': df['high'].max(),
                        'Low': df['low'].min(),
                        'Z-Score': calculate_zscore(df)
                    })
                    
        except Exception as e:
            logger.error(f"Error processing {symbol}: {str(e)}")
            
    return pd.DataFrame(results)

def main():
    st.title("Market Analysis Dashboard 📊")
    
    # Initialize API with proper error handling
    try:
        api = SchwabAPI()
        if not api.token:
            st.error("Failed to authenticate with Schwab API")
            st.stop()
            
        # Display authentication status
        st.sidebar.markdown("### API Status")
        st.sidebar.success("Connected to Schwab API")
        
        # Add logout button
        if st.sidebar.button("Logout"):
            api.token = None
            st.experimental_rerun()
            
        # Sidebar settings
        st.sidebar.header("Settings")
        months = st.sidebar.slider("Months of Historical Data", 1, 24, 12)
        
        # Main content
        tab1, tab2, tab3 = st.tabs(["Market Analysis", "Portfolio", "Options"])
        
        with tab1:
            symbols = st.multiselect(
                "Select Symbols",
                ["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
                default=["AAPL", "MSFT"]
            )
            
            if symbols:
                data = process_market_data(api, symbols, months)
                st.dataframe(data)
                
                # Plot historical data
                for symbol in symbols:
                    hist_data = api.get_historical_data(
                        symbol,
                        datetime.now() - timedelta(days=months*30)
                    )
                    if hist_data:
                        df = pd.DataFrame(hist_data['candles'])
                        st.line_chart(df.set_index('timestamp')['close'])
        
        with tab2:
            positions = api.get_account_positions()
            if positions:
                st.dataframe(pd.DataFrame(positions['positions']))
        
        with tab3:
            symbol = st.selectbox("Select Symbol for Options", symbols)
            if symbol:
                options = api.get_option_chain(symbol)
                if options:
                    st.dataframe(pd.DataFrame(options['options']))

    except Exception as e:
        st.error(f"Error initializing API: {str(e)}")
        logger.error(f"Initialization error: {str(e)}")

if __name__ == "__main__":
    main() 