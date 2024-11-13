import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import time
import requests
from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session
import ssl
import uuid
import logging
import traceback
from urllib.parse import quote

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Configure Streamlit page settings
st.set_page_config(
    page_title="Market Analysis Dashboard 📊",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Hide Streamlit menu and footer
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.stDeployButton {display:none;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Update the SCHWAB_CONFIG with your exact details
SCHWAB_CONFIG = {
    'client_id': 'f6MOF1oqGHpQC6sZPCvTPRe7nyMWDgof',
    'client_secret': 'NCiMvAdTarXnDcIq',
    'token_url': 'https://api.schwabapi.com/v1/oauth/token',
    'auth_url': 'https://api.schwabapi.com/v1/oauth/authorize',
    'redirect_uri': quote('https://quantwebapp-bjbqck9aebxpadmg6h7pmk.streamlit.app/', safe=''),
    'api_base_url': 'https://api.schwabapi.com/v1/',
    'scope': ['readonly']
}

class SchwabAPI:
    def __init__(self):
        self.session = OAuth2Session(
            client_id=SCHWAB_CONFIG['client_id'],
            redirect_uri=SCHWAB_CONFIG['redirect_uri'],
            scope=SCHWAB_CONFIG['scope']
        )
        self.token = None
        self.authenticate()
    
    def authenticate(self):
        """Authenticate with Schwab API using callback URL"""
        try:
            # Check for existing token
            if 'oauth_token' not in st.session_state:
                # Single authentication prompt in sidebar
                with st.sidebar:
                    st.markdown("### Authentication Status")
                    if st.button("Authenticate with Schwab", key="auth_button_unique"):
                        token_response = self.get_token()
                        if token_response:
                            st.success("Authentication successful!")
                            return True
            else:
                self.token = st.session_state['oauth_token']
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            st.error(f"Authentication failed: {str(e)}")
            return False

    def get_token(self):
        """Get token using client credentials"""
        try:
            token_data = {
                'grant_type': 'client_credentials',
                'client_id': SCHWAB_CONFIG['client_id'],
                'client_secret': SCHWAB_CONFIG['client_secret']
            }
            
            response = requests.post(
                SCHWAB_CONFIG['token_url'],
                data=token_data,
                headers={'Accept': 'application/json'}
            )
            
            if response.status_code == 200:
                token = response.json()
                st.session_state['oauth_token'] = token
                self.token = token
                return token
            else:
                st.error(f"Authentication failed: {response.text}")
                return None
                
        except Exception as e:
            st.error(f"Authentication error: {str(e)}")
            return None

    def get_market_data(self, symbol, interval='1d'):
        """Fetch market data from Schwab API"""
        if not self.token:
            return None
            
        headers = {
            'Authorization': f"Bearer {self.token.get('access_token')}",
            'Accept': 'application/json',
            'SchwabClientCustomerId': 'Someone',  # Add based on your requirements
            'SchwabClientCorrelId': str(uuid.uuid4())  # Generate unique ID for each request
        }
        
        # Format request based on Schwab API requirements
        request_data = {
            "requestid": "0",
            "service": "LEVELONE_EQUITIES",
            "command": "SUBS",
            "parameters": {
                "keys": symbol,
                "fields": "0,1,2,3,4,5,6,7,8,9"  # Adjust fields based on your needs
            }
        }
        
        try:
            endpoint = f"{SCHWAB_CONFIG['api_base_url']}markets/quotes/{symbol}"
            response = self.session.post(endpoint, headers=headers, json=request_data)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Failed to fetch market data: {str(e)}")
            if getattr(response, 'status_code', None) == 401:
                if 'oauth_token' in st.session_state:
                    del st.session_state['oauth_token']
            return None

# Add a function to handle the OAuth callback
def handle_oauth_callback():
    if 'code' in st.experimental_get_query_params():
        auth_code = st.experimental_get_query_params()['code'][0]
        api = SchwabAPI()
        if api.authenticate():
            st.success("Successfully authenticated!")
            st.experimental_rerun()

# Modify the data fetching functions to use Schwab API instead of yfinance
def get_historical_data(ticker, months):
    """Get historical data using Schwab API"""
    api = SchwabAPI()
    data = api.get_market_data(ticker)
    
    if data:
        # Transform Schwab API response to match the expected DataFrame format
        df = pd.DataFrame(data['quotes'])
        df.set_index('timestamp', inplace=True)
        df.index = pd.to_datetime(df.index)
        return df
    return None

def calculate_zscore(data, window=20):
    """Calculate Z-score for a given dataset"""
    mean = data['close'].rolling(window=window).mean()
    std = data['close'].rolling(window=window).std()
    zscore = (data['close'] - mean) / std
    return zscore.iloc[-1] if not zscore.empty else None

def calculate_technical_indicators(data):
    """Calculate technical indicators using basic pandas operations"""
    df = data.copy()
    df.columns = df.columns.str.lower()
    
    # Moving Averages
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (std * 2)
    df['bb_lower'] = df['bb_middle'] - (std * 2)
    
    # Volume
    df['volume_sma'] = df['volume'].rolling(window=20).mean()
    
    # Z-Score
    df['zscore'] = (df['close'] - df['close'].rolling(window=20).mean()) / df['close'].rolling(window=20).std()
    
    df = df.replace([np.inf, -np.inf], np.nan)
    return df

def perform_regression_analysis(data, target_col, feature_cols, model_type='linear', test_size=0.2):
    """Perform regression analysis on the selected data"""
    try:
        # Ensure all column names are lowercase
        data.columns = data.columns.str.lower()
        target_col = target_col.lower()
        
        # Check if target column exists
        if target_col not in data.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")
            
        X = data[feature_cols]
        y = data[target_col]
        
        # Check for NaN values
        if X.isna().any().any() or y.isna().any():
            X = X.fillna(method='ffill')
            y = y.fillna(method='ffill')
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42
        )
        
        if model_type == 'linear':
            model = LinearRegression()
        elif model_type == 'ridge':
            model = Ridge(alpha=1.0)
        elif model_type == 'lasso':
            model = Lasso(alpha=1.0)
        
        model.fit(X_train, y_train)
        
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        results = {
            'model': model,
            'r2_train': r2_score(y_train, y_pred_train),
            'r2_test': r2_score(y_test, y_pred_test),
            'rmse_train': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'rmse_test': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'feature_importance': pd.DataFrame({
                'Feature': feature_cols,
                'Coefficient': model.coef_
            }).sort_values('Coefficient', key=abs, ascending=False),
            'y_test': y_test,
            'y_pred_test': y_pred_test
        }
        return results
    except Exception as e:
        st.error(f"Error in regression analysis: {str(e)}")
        return None

def create_regression_plots(results, title):
    """Create plots for regression analysis results"""
    fig1 = px.scatter(
        x=results['y_test'],
        y=results['y_pred_test'],
        labels={'x': 'Actual Values', 'y': 'Predicted Values'},
        title=f'{title}: Actual vs Predicted Values'
    )
    fig1.add_trace(
        go.Scatter(
            x=[results['y_test'].min(), results['y_test'].max()],
            y=[results['y_test'].min(), results['y_test'].max()],
            mode='lines',
            name='Perfect Prediction',
            line=dict(dash='dash', color='red')
        )
    )
    
    fig2 = px.bar(
        results['feature_importance'],
        x='Coefficient',
        y='Feature',
        orientation='h',
        title=f'{title}: Feature Importance'
    )
    
    return fig1, fig2

def get_futures_symbols():
    """Return a dictionary of futures symbols and their descriptions"""
    return {
        "ES=F": "E-mini S&P 500 Futures",
        "NQ=F": "E-mini Nasdaq 100 Futures",
        "YM=F": "E-mini Dow Futures",
        "RTY=F": "E-mini Russell 2000 Futures",
        "ZB=F": "U.S. Treasury Bond Futures",
        "GC=F": "Gold Futures",
        "SI=F": "Silver Futures",
        "CL=F": "Crude Oil Futures",
        "NG=F": "Natural Gas Futures"
    }

def get_index_symbols():
    """Return a dictionary of major index symbols and their descriptions"""
    return {
        "^GSPC": "S&P 500",
        "^NDX": "Nasdaq 100",
        "^DJI": "Dow Jones Industrial Average",
        "^RUT": "Russell 2000",
        "^VIX": "VIX Volatility Index"
    }

def format_futures_price(price, ticker):
    """Format futures prices with appropriate decimal places"""
    if ticker in ['ES=F', 'NQ=F', 'YM=F']:
        return round(price, 2)
    elif ticker in ['GC=F']:
        return round(price, 1)
    elif ticker in ['SI=F', 'CL=F']:
        return round(price, 3)
    else:
        return round(price, 4)

def main():
    st.title("Market Analysis Dashboard 📊")
    
    # Single logout button in sidebar
    if st.sidebar.button("Logout", key="logout_button_unique"):
        if 'oauth_token' in st.session_state:
            del st.session_state['oauth_token']
        st.experimental_rerun()
    
    # Initialize API once
    api = SchwabAPI()
    if not api.token:
        st.warning("Please authenticate to access market data")
        return
        
    # Rest of your main function code... 