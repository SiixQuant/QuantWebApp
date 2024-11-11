import streamlit as st
import yfinance as yf
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
    
    st.sidebar.header("Settings")
    months = st.sidebar.slider("Months of Historical Data", 1, 24, 6)
    zscore_window = st.sidebar.slider("Z-Score Window", 5, 50, 20)
    zscore_threshold = st.sidebar.slider("Z-Score Threshold", 1.0, 3.0, 2.0)
    
    # Default tickers with indices and futures
    default_tickers = ['SPY', 'QQQ', 'IWM', 'DIA']
    indices = get_index_symbols()
    futures = get_futures_symbols()
    
    # Add selection for instrument type
    instrument_type = st.sidebar.selectbox(
        "Select Instrument Type",
        ["ETFs", "Indices", "Futures", "Custom"]
    )
    
    if instrument_type == "ETFs":
        tickers = default_tickers
    elif instrument_type == "Indices":
        tickers = list(indices.keys())
    elif instrument_type == "Futures":
        tickers = list(futures.keys())
    else:  # Custom
        custom_ticker = st.sidebar.text_input("Enter Custom Ticker").upper()
        tickers = [custom_ticker] if custom_ticker else default_tickers
    
    # Display instrument description
    if instrument_type == "Indices":
        st.sidebar.markdown("### Selected Indices")
        for symbol in tickers:
            st.sidebar.text(f"{symbol}: {indices[symbol]}")
    elif instrument_type == "Futures":
        st.sidebar.markdown("### Selected Futures")
        for symbol in tickers:
            st.sidebar.text(f"{symbol}: {futures[symbol]}")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📊 Z-Score Analysis", "📈 Technical Analysis", "🔍 Regression Analysis"])
    
    with tab1:
        st.header("Z-Score Analysis")
        
        # Add instrument type indicator
        st.subheader(f"Currently Analyzing: {instrument_type}")
        
        # Create progress bar
        progress_bar = st.progress(0)
        results = []
        
        for i, ticker in enumerate(tickers):
            try:
                data = yf.download(ticker, 
                                 start=datetime.now() - timedelta(days=30*months),
                                 end=datetime.now(),
                                 progress=False)
                if not data.empty:
                    data.columns = data.columns.str.lower()
                    zscore = calculate_zscore(data, zscore_window)
                    if zscore is not None:
                        instrument_name = indices.get(ticker, futures.get(ticker, ticker))
                        results.append({
                            'Ticker': ticker,
                            'Name': instrument_name,
                            'Z-Score': round(zscore, 2),
                            'Last Price': format_futures_price(data['close'].iloc[-1], ticker),
                            'Change %': round((data['close'].iloc[-1] / data['close'].iloc[-2] - 1) * 100, 2),
                            'Last Date': data.index[-1].strftime('%Y-%m-%d'),
                            'Signal': '🔴 Oversold' if zscore <= -zscore_threshold else '🟢 Overbought' if zscore >= zscore_threshold else '⚪ Neutral'
                        })
            except Exception as e:
                st.warning(f"Error processing {ticker}: {str(e)}")
            progress_bar.progress((i + 1) / len(tickers))
        
        if results:
            df = pd.DataFrame(results)
            # Add color coding for Change %
            def color_change(val):
                color = 'red' if val < 0 else 'green'
                return f'color: {color}'
            
            st.dataframe(
                df.style
                .background_gradient(subset=['Z-Score'])
                .map(color_change, subset=['Change %']),
                use_container_width=True
            )
    
    with tab2:
        st.header("Technical Analysis")
        selected_ticker = st.selectbox("Select Ticker", tickers)
        
        data = yf.download(selected_ticker, 
                          start=datetime.now() - timedelta(days=30*months),
                          end=datetime.now())
        
        if not data.empty:
            data.columns = data.columns.str.lower()
            data = calculate_technical_indicators(data)
            
            fig = make_subplots(
                rows=2, 
                cols=1,
                vertical_spacing=0.03,
                row_heights=[0.7, 0.3],
                shared_xaxes=True
            )
            
            fig.add_trace(
                go.Candlestick(
                    x=data.index,
                    open=data['open'],
                    high=data['high'],
                    low=data['low'],
                    close=data['close'],
                    name='Price'
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=data['volume'],
                    name='Volume'
                ),
                row=2, col=1
            )
            
            fig.update_layout(
                height=800,
                title=f'{selected_ticker} Price Chart',
                xaxis_rangeslider_visible=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Technical Indicators Display
            col1, col2, col3, col4 = st.columns(4)
            latest = data.iloc[-1]
            col1.metric("RSI", f"{latest['rsi']:.2f}")
            col2.metric("MACD", f"{latest['macd']:.2f}")
            col3.metric("Z-Score", f"{latest['zscore']:.2f}")
            col4.metric("SMA 20", f"{latest['sma_20']:.2f}")
    
    with tab3:
        st.header("Regression Analysis")
        selected_ticker = st.selectbox("Select Ticker for Regression", tickers, key='regression_ticker')
        
        data = yf.download(selected_ticker, 
                          start=datetime.now() - timedelta(days=30*months),
                          end=datetime.now())
        
        if not data.empty:
            # Ensure column names are lowercase before processing
            data.columns = data.columns.str.lower()
            data = calculate_technical_indicators(data)
            data = data.dropna()
            
            available_features = [
                'sma_20', 'sma_50', 'rsi', 'macd',
                'macd_signal', 'macd_hist', 'volume_sma',
                'bb_lower', 'bb_middle', 'bb_upper', 'zscore'
            ]
            
            selected_features = st.multiselect(
                "Select Features for Regression",
                available_features,
                default=['sma_20', 'rsi', 'macd']
            )
            
            if selected_features:
                model_type = st.selectbox(
                    "Select Regression Model",
                    ['linear', 'ridge', 'lasso'],
                    format_func=lambda x: x.capitalize() + " Regression"
                )
                
                if st.button("Run Regression Analysis"):
                    with st.spinner("Running regression analysis..."):
                        results = perform_regression_analysis(
                            data,
                            'close',  # Using lowercase target column name
                            selected_features,
                            model_type
                        )
                        
                        if results:
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("Train R²", f"{results['r2_train']:.3f}")
                            col2.metric("Test R²", f"{results['r2_test']:.3f}")
                            col3.metric("Train RMSE", f"{results['rmse_train']:.2f}")
                            col4.metric("Test RMSE", f"{results['rmse_test']:.2f}")
                            
                            fig1, fig2 = create_regression_plots(results, selected_ticker)
                            st.plotly_chart(fig1, use_container_width=True)
                            st.plotly_chart(fig2, use_container_width=True)

if __name__ == "__main__":
    main() 