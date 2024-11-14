import streamlit as st
import requests 
import socket
from requests_oauthlib import OAuth2Session 
import logging
import urllib3

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Update configuration with IP address instead of domain
SCHWAB_CONFIG = {
    'client_id': 'f6MOF1oqGHpQC6sZPCvTPRe7nyMWDgof',
    'client_secret': 'NCiMvAdTarXnDcJq',
    'base_url': 'https://api.schwabapi.com/v1',
    'token_url': 'https://api.schwabapi.com/v1/oauth/token',
    'auth_url': 'https://api.schwabapi.com/v1/oauth/authorize',
    'redirect_uri': 'https://quantwebapp-bjbqck9aebxpadmg6h7pmk.streamlit.app/',
    'scope': ['readonly'],
    'host_header': 'api.schwabapi.com'
}

class SchwabAPI:
    def __init__(self):
        # Configure session with SSL warning suppression
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        self.session = requests.Session()
        self.session.verify = False  # Disable SSL verification
        
        # Configure custom DNS resolution
        self.session.mount('https://', CustomHTTPAdapter())
        
        # Add proxy support if needed
        self.proxies = {
            'http': '',  # Add your proxy if needed
            'https': ''  # Add your proxy if needed
        }
        
        self.token = None
        self.authenticate()
        self.request_id = 0  # To track unique request IDs

    def authenticate(self):
        """Authenticate with Schwab API"""
        try:
            headers = {
                'Accept': 'application/json',
                'Content-Type': 'application/x-www-form-urlencoded',
                'Host': SCHWAB_CONFIG['host_header'],  # Add host header
                'Connection': 'keep-alive'
            }
            
            data = {
                'grant_type': 'client_credentials',
                'client_id': SCHWAB_CONFIG['client_id'],
                'client_secret': SCHWAB_CONFIG['client_secret']
            }
            
            logger.debug(f"Attempting authentication to: {SCHWAB_CONFIG['token_url']}")
            
            # Make request with custom headers and proxy support
            response = self.session.post(
                SCHWAB_CONFIG['token_url'],
                headers=headers,
                data=data,
                proxies=self.proxies,
                timeout=30,
                verify=False  # Temporarily disable SSL verification
            )
            
            logger.debug(f"Authentication response status: {response.status_code}")
            logger.debug(f"Authentication response: {response.text}")
            
            if response.status_code == 200:
                self.token = response.json()
                st.success("Successfully authenticated with Schwab API")
                return True
            else:
                st.error(f"Authentication failed: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            st.error(f"Authentication failed: {str(e)}")
            return False

    def get_quotes(self, symbols):
        """Get quotes for a list of symbols"""
        try:
            self.request_id += 1
            
            # Format request according to documentation
            request_data = {
                "service": "LEVELONE_EQUITY",
                "command": "SUBS",
                "requestid": str(self.request_id),
                "parameters": {
                    "symbols": symbols if isinstance(symbols, str) else ','.join(symbols),
                    "fields": "0,1,2,3,4,5"  # Basic quote fields
                }
            }
            
            headers = {
                'Accept': 'application/json',
                'Authorization': f'Bearer {self.token["access_token"]}',
                'Host': SCHWAB_CONFIG['host_header']
            }
            
            url = f"{SCHWAB_CONFIG['base_url']}/stream"  # Adjust endpoint as needed
            
            response = self.session.post(
                url,
                json=request_data,
                headers=headers,
                proxies=self.proxies,
                timeout=30,
                verify=False
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Failed to get quotes: {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Quote retrieval error: {str(e)}")
            st.error(f"Failed to get quotes: {str(e)}")
            return None

# Custom HTTP Adapter for DNS resolution
from requests.adapters import HTTPAdapter
class CustomHTTPAdapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        # Force IP resolution
        kwargs['assert_hostname'] = False
        kwargs['server_hostname'] = SCHWAB_CONFIG['host_header']
        return super().init_poolmanager(*args, **kwargs)

def main():
    st.title("Market Analysis Dashboard 📊")
    
    # Add network status check
    try:
        # Test DNS resolution
        ip = socket.gethostbyname('api.schwabapi.com')
        st.sidebar.success(f"DNS Resolution successful: {ip}")
    except socket.gaierror as e:
        st.sidebar.error(f"DNS Resolution failed: {str(e)}")
    
    try:
        api = SchwabAPI()
        if not api.token:
            st.error("Failed to authenticate with Schwab API")
            st.stop()
            
        st.sidebar.markdown("### API Status")
        st.sidebar.success("Connected to Schwab API")
        
        # Add quote retrieval interface
        st.sidebar.markdown("### Get Quotes")
        symbols = st.sidebar.text_input("Enter symbols (comma-separated)", "AAPL,MSFT,GOOGL")
        
        if st.sidebar.button("Get Quotes"):
            quotes = api.get_quotes(symbols)
            if quotes:
                st.write("### Market Quotes")
                st.json(quotes)
        
    except Exception as e:
        st.error(f"Error initializing API: {str(e)}")
        logger.error(f"Initialization error: {str(e)}")

if __name__ == "__main__":
    main() 