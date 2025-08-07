import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# --- Page Configuration ---
st.set_page_config(
    page_title="Financial Sentiment Dashboard",
    page_icon="💹",
    layout="wide"
)

# --- Data Loading and Preprocessing ---
# This section now loads data directly from your specified JSON file.
# Make sure 'data_sentiment_kimi.json' is in the './data/' directory relative to your script.
try:
    df = pd.read_json('./data/data_sentiment_kimi.json')
    df = df[df['confidence'] != 0]  # Remove rows with zero confidence
    df['dt'] = pd.to_datetime(df['dt'])
    df.set_index('dt', inplace=True)
    df = df.sort_index(ascending=True)

    # --- Feature Engineering (must be done after loading) ---
    # Convert to category type for efficiency
    df['sentiment'] = df['sentiment'].astype('category')
    df['impact_horizon'] = df['impact_horizon'].astype('category')
    
    # Map sentiment to a numerical score for calculations
    sentiment_map = {'positive': 1, 'neutral': 0, 'negative': -1}
    df['sentiment_score'] = df['sentiment'].map(sentiment_map)
    df['sentiment_score'] = df['sentiment_score'].astype(str).astype(float)
    print(df.head())
except FileNotFoundError:
    st.error("Error: The data file was not found. Please make sure `data_sentiment_kimi.json` is placed in the `./data/` directory.")
    st.stop() # Stop the app if data cannot be loaded


# --- Sidebar for Filters ---
st.sidebar.header("Dashboard Filters")

# Date range selector
min_date = df.index.min().date()
max_date = df.index.max().date()
start_date, end_date = st.sidebar.date_input(
    "Select Date Range",
    [min_date, max_date],
    min_value=min_date,
    max_value=max_date
)

# Filter data based on date range
mask = (df.index.date >= start_date) & (df.index.date <= end_date)
filtered_df = df.loc[mask]

# --- Main Dashboard ---
st.title("📈 Financial News Sentiment Dashboard")
st.markdown("An interactive dashboard to analyze financial news sentiment over time.")

# --- Key Metrics ---
st.header("Key Metrics")
col1, col2 = st.columns(2)

total_articles = len(filtered_df)
avg_confidence = filtered_df['confidence'].mean()

col1.metric("Total News Articles", f"{total_articles}")
col2.metric("Average Confidence", f"{avg_confidence:.2f}")


# --- Time Series Analysis ---
st.header("Sentiment Time Series Analysis")

# Calculate daily weighted net sentiment
def weighted_sentiment(x):
    # print("sentiment_score dtype:", x['sentiment_score'].dtype)
    # print("confidence dtype:", x['confidence'].dtype)
    if x['confidence'].sum() == 0: return 0
    return (x['sentiment_score'] * x['confidence']).sum() / x['confidence'].sum()

daily_sentiment = filtered_df.resample('D').apply(weighted_sentiment).rename('Sentiment')
daily_sentiment_ma = daily_sentiment.rolling(window=7).mean().rename('Sentiment_MA')

# Create the plot
fig_sentiment_ts = px.line(
    daily_sentiment, 
    y='Sentiment', 
    title='Daily Weighted Net Sentiment Index',
    labels={'Sentiment': 'Net Sentiment Score', 'dt': 'Date'}
)
fig_sentiment_ts.add_scatter(
    x=daily_sentiment_ma.index, 
    y=daily_sentiment_ma, 
    mode='lines', 
    name='7-Day Moving Average',
    line=dict(color='red', width=3)
)
fig_sentiment_ts.add_hline(y=0, line_dash="dash", line_color="grey")
st.plotly_chart(fig_sentiment_ts, use_container_width=True)


# --- Correlation with Market Data ---
st.header("Sentiment vs. Market Performance ")

# Generate mock market data for the filtered range
market_index = pd.Series(
    100 + np.random.randn(len(daily_sentiment)).cumsum(),
    index=daily_sentiment.index,
    name='Market'
)
market_index += (daily_sentiment.shift(1) * 20).fillna(0) # Sentiment affects next day's market

# Create the dual-axis plot
fig_corr = make_subplots(specs=[[{"secondary_y": True}]])

# Add Market Index trace
fig_corr.add_trace(
    go.Scatter(x=market_index.index, y=market_index, name="Mock Market Index", line=dict(color='blue')),
    secondary_y=False,
)

# Add Sentiment MA trace
fig_corr.add_trace(
    go.Scatter(x=daily_sentiment_ma.index, y=daily_sentiment_ma, name="7-Day MA Sentiment", line=dict(color='red')),
    secondary_y=True,
)

# Add figure title and axis labels
fig_corr.update_layout(title_text="Sentiment Index vs. Market Performance")
fig_corr.update_xaxes(title_text="Date")
fig_corr.update_yaxes(title_text="<b>Market Index</b>", secondary_y=False, color='blue')
fig_corr.update_yaxes(title_text="<b>Net Sentiment Score</b>", secondary_y=True, color='red')

st.plotly_chart(fig_corr, use_container_width=True)


# --- Data Distributions (EDA) ---
st.header("Exploratory Data Analysis")
col1_eda, col2_eda = st.columns(2)

with col1_eda:
    # Sentiment Distribution Pie Chart
    sentiment_counts = filtered_df['sentiment'].value_counts()
    fig_pie = px.pie(
        values=sentiment_counts.values, 
        names=sentiment_counts.index, 
        title='News Sentiment Distribution',
        color_discrete_map={'positive':'green', 'negative':'red', 'neutral':'grey'}
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col2_eda:
    # Confidence Score Histogram
    fig_hist = px.histogram(    
        filtered_df, 
        x='confidence', 
        nbins=30,
        title='Sentiment Confidence Score Distribution'
    )
    st.plotly_chart(fig_hist, use_container_width=True)

# --- Raw Data Viewer ---
st.header("Raw Data Explorer")
if st.checkbox("Show Raw Data Table", value=False):
    st.dataframe(filtered_df)
