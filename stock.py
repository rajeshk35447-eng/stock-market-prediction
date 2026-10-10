"""
Stock Market Prediction Dashboard
----------------------------------
LSTM based stock price forecasting dashboard.
Input   : Historical stock data (via Yahoo Finance)
Output  : Future price trend prediction

Author  : Rajesh Kumar Mahto
Course  : B.Tech CSE, Jharkhand Rai University
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import feedparser
from keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler

# ==========================================================================
# PAGE CONFIG
# ==========================================================================
st.set_page_config(
    page_title="StockSense | AI Market Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================================================
# THEME  —  "trading terminal" dark theme
# ==========================================================================
ACCENT_UP = "#2DD4BF"       # teal-green : bullish / gains
ACCENT_DOWN = "#FB7185"     # coral-red  : bearish / losses
ACCENT_WARN = "#FBBF24"     # amber      : neutral / hold
ACCENT_PRIMARY = "#38BDF8"  # sky blue   : brand / links
BG_DARK = "#0B0F14"
BG_PANEL = "#131922"
BG_PANEL_2 = "#1B232E"
TEXT_MAIN = "#E5EAF0"
TEXT_MUTED = "#8B98A8"
BORDER = "#232C38"

CUSTOM_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"]  {{
    font-family: 'Inter', sans-serif;
}}

.stApp {{
    background: {BG_DARK};
    color: {TEXT_MAIN};
}}

/* Sidebar */
section[data-testid="stSidebar"] {{
    background: {BG_PANEL};
    border-right: 1px solid {BORDER};
}}

/* Headings use the display face */
h1, h2, h3 {{
    font-family: 'Space Grotesk', sans-serif !important;
    letter-spacing: -0.01em;
}}

/* Hero ticker banner */
.ticker-banner {{
    background: linear-gradient(90deg, {BG_PANEL} 0%, {BG_PANEL_2} 100%);
    border: 1px solid {BORDER};
    border-radius: 14px;
    padding: 22px 28px;
    margin-bottom: 18px;
}}
.ticker-symbol {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.6rem;
    font-weight: 700;
    color: {TEXT_MAIN};
    line-height: 1;
}}
.ticker-price {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.6rem;
    font-weight: 700;
}}
.ticker-sub {{
    color: {TEXT_MUTED};
    font-size: 0.85rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}}

/* Metric cards */
div[data-testid="stMetric"] {{
    background: {BG_PANEL};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 14px 16px;
}}
div[data-testid="stMetricLabel"] {{
    color: {TEXT_MUTED} !important;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}}
div[data-testid="stMetricValue"] {{
    font-family: 'JetBrains Mono', monospace !important;
    color: {TEXT_MAIN} !important;
}}

/* Signal pill badges */
.signal-pill {{
    display: inline-block;
    padding: 10px 22px;
    border-radius: 999px;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    font-size: 1.05rem;
    letter-spacing: 0.03em;
    margin: 4px 0 12px 0;
}}
.pill-buy   {{ background: rgba(45, 212, 191, 0.12); color: {ACCENT_UP};   border: 1px solid {ACCENT_UP}; }}
.pill-sell  {{ background: rgba(251, 113, 133, 0.12); color: {ACCENT_DOWN}; border: 1px solid {ACCENT_DOWN}; }}
.pill-hold  {{ background: rgba(251, 191, 36, 0.12);  color: {ACCENT_WARN}; border: 1px solid {ACCENT_WARN}; }}

/* Section divider replacement */
.section-label {{
    color: {TEXT_MUTED};
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    border-bottom: 1px solid {BORDER};
    padding-bottom: 6px;
    margin: 26px 0 14px 0;
}}

/* Tabs */
button[data-baseweb="tab"] {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.95rem;
}}

/* Dataframes */
div[data-testid="stDataFrame"] {{
    border: 1px solid {BORDER};
    border-radius: 8px;
}}

footer, #MainMenu {{ visibility: hidden; }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def section_label(text: str):
    st.markdown(f'<div class="section-label">{text}</div>', unsafe_allow_html=True)


def signal_pill(text: str, kind: str):
    """kind: buy | sell | hold"""
    st.markdown(f'<span class="signal-pill pill-{kind}">{text}</span>', unsafe_allow_html=True)


# ==========================================================================
# CACHED DATA / MODEL HELPERS
# ==========================================================================
MODEL_PATH_DEFAULT = os.path.join(os.path.dirname(__file__), "model", "stock_prediction_model.keras")


@st.cache_resource(show_spinner=False)
def get_model(model_path: str):
    return load_model(model_path)


@st.cache_data(show_spinner=False, ttl=60 * 30)
def get_history(symbol: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(symbol, start=start, end=end, progress=False)
    if df.empty:
        return df
    df.reset_index(inplace=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


@st.cache_data(show_spinner=False, ttl=60 * 30)
def get_company_info(symbol: str) -> dict:
    try:
        return yf.Ticker(symbol).info or {}
    except Exception:
        return {}


@st.cache_data(show_spinner=False, ttl=60 * 15)
def get_news(symbol: str):
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"
    feed = feedparser.parse(url)
    return feed.entries[:5]


def run_future_forecast(scaler, seed_window: np.ndarray, model, n_days: int) -> np.ndarray:
    """Iteratively rolls the model forward n_days steps and returns
    the inverse-transformed predicted price series."""
    window = seed_window.copy()
    preds = []
    for _ in range(n_days):
        pred = model.predict(window.reshape(1, 100, 1), verbose=0)
        preds.append(pred[0, 0])
        window = np.append(window[1:], pred)
    preds = np.array(preds).reshape(-1, 1)
    return scaler.inverse_transform(preds)


# ==========================================================================
# SIDEBAR — CONTROL PANEL
# ==========================================================================
with st.sidebar:
    st.markdown("### 📈 StockSense")
    st.caption("AI-powered market prediction terminal")
    st.markdown("---")

    stock = st.text_input("Stock Symbol", "GOOG").strip().upper()

    future_days = st.selectbox("Forecast Horizon (days)", [30, 90, 180, 365], index=0)

    st.markdown("---")
    compare_stocks = st.multiselect(
        "Compare Against",
        ["AAPL", "GOOG", "MSFT", "AMZN", "META", "TSLA", "NVDA"],
        default=["GOOG", "AAPL"],
    )

    st.markdown("---")
    model_upload = st.file_uploader("Model file (.keras) — optional", type=["keras"])
    st.caption("If not provided, the app looks for a bundled model at `model/stock_prediction_model.keras`.")

    run_btn = st.button("🚀 Run Analysis", use_container_width=True, type="primary")

if "run" not in st.session_state:
    st.session_state.run = False
if run_btn:
    st.session_state.run = True

# ==========================================================================
# HEADER
# ==========================================================================
st.markdown(
    """
    <h1 style="margin-bottom:0;">StockSense</h1>
    <p style="color:#8B98A8; margin-top:4px;">Deep learning (LSTM) based stock price prediction &mdash; historical data in, future trend out.</p>
    """,
    unsafe_allow_html=True,
)

if not st.session_state.run:
    st.info("👈 Set your symbol and forecast horizon in the sidebar, then click **Run Analysis** to begin.")
    st.stop()

# ==========================================================================
# LOAD DATA
# ==========================================================================
START, END = "2016-01-01", pd.Timestamp.today().strftime("%Y-%m-%d")

with st.spinner(f"Fetching historical data for {stock}..."):
    data = get_history(stock, START, END)

if data.empty:
    st.error(f"❌ No data found for symbol **{stock}**. Please check the ticker and try again.")
    st.stop()

required_cols = {"Date", "Open", "High", "Low", "Close", "Volume"}
if not required_cols.issubset(data.columns):
    st.error("❌ Unexpected data format returned by Yahoo Finance. Please try again later.")
    st.stop()

current_price = float(data["Close"].iloc[-1])
day_high = float(data["High"].iloc[-1])
day_low = float(data["Low"].iloc[-1])
volume = int(data["Volume"].iloc[-1])
prev_close = float(data["Close"].iloc[-2]) if len(data) > 1 else current_price
day_change_pct = ((current_price - prev_close) / prev_close) * 100 if prev_close else 0.0

# ---- Hero ticker banner -------------------------------------------------
change_color = ACCENT_UP if day_change_pct >= 0 else ACCENT_DOWN
change_arrow = "▲" if day_change_pct >= 0 else "▼"
st.markdown(
    f"""
    <div class="ticker-banner">
        <div class="ticker-sub">Live Snapshot</div>
        <div style="display:flex; align-items:baseline; gap:18px; flex-wrap:wrap;">
            <span class="ticker-symbol">{stock}</span>
            <span class="ticker-price" style="color:{change_color};">${current_price:,.2f}</span>
            <span style="font-family:'JetBrains Mono',monospace; color:{change_color}; font-size:1.1rem;">
                {change_arrow} {abs(day_change_pct):.2f}%
            </span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Day High", f"${day_high:.2f}")
col2.metric("Day Low", f"${day_low:.2f}")
col3.metric("Volume", f"{volume:,}")
col4.metric("Data Points", f"{len(data):,}")

# ==========================================================================
# TABS
# ==========================================================================
tab_overview, tab_technical, tab_model, tab_forecast, tab_company, tab_compare, tab_portfolio, tab_about = st.tabs(
    [
        "📊 Overview",
        "📉 Technical Analysis",
        "🤖 Model Performance",
        "🔮 Future Forecast",
        "🏢 Company & News",
        "⚖️ Compare",
        "💼 Portfolio",
        "ℹ️ About",
    ]
)

# --------------------------------------------------------------------------
# TAB: OVERVIEW
# --------------------------------------------------------------------------
with tab_overview:
    section_label("Historical Closing Price")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["Date"], y=data["Close"], mode="lines",
                              name="Close", line=dict(color=ACCENT_PRIMARY, width=2)))
    fig.update_layout(
        template="plotly_dark", height=460, hovermode="x unified",
        paper_bgcolor=BG_DARK, plot_bgcolor=BG_PANEL,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    section_label("Candlestick Chart")
    fig = go.Figure(data=[go.Candlestick(
        x=data["Date"], open=data["Open"], high=data["High"],
        low=data["Low"], close=data["Close"],
        increasing_line_color=ACCENT_UP, decreasing_line_color=ACCENT_DOWN,
    )])
    fig.update_layout(template="plotly_dark", height=460,
                       paper_bgcolor=BG_DARK, plot_bgcolor=BG_PANEL,
                       margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    section_label("Trading Volume")
    fig = go.Figure(go.Bar(x=data["Date"], y=data["Volume"], marker_color=ACCENT_PRIMARY))
    fig.update_layout(template="plotly_dark", height=320,
                       paper_bgcolor=BG_DARK, plot_bgcolor=BG_PANEL,
                       margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    section_label("Historical Data Table")
    st.dataframe(data[["Date", "Open", "High", "Low", "Close", "Volume"]], use_container_width=True, height=280)
    st.download_button(
        "📥 Download Historical Data (CSV)",
        data=data.to_csv(index=False),
        file_name=f"{stock}_historical_data.csv",
        mime="text/csv",
    )

    section_label("Summary Statistics")
    daily_return = data["Close"].pct_change().mean() * 100
    volatility_pct = data["Close"].pct_change().std() * 100
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Highest", f"${data['High'].max():.2f}")
    c2.metric("Lowest", f"${data['Low'].min():.2f}")
    c3.metric("Average", f"${data['Close'].mean():.2f}")
    c4.metric("Avg Daily Return", f"{daily_return:.2f}%")
    c5.metric("Volatility", f"{volatility_pct:.2f}%")

# --------------------------------------------------------------------------
# TAB: TECHNICAL ANALYSIS
# --------------------------------------------------------------------------
with tab_technical:
    ma20 = data["Close"].rolling(20).mean()
    ma50 = data["Close"].rolling(50).mean()
    ma100 = data["Close"].rolling(100).mean()
    ma200 = data["Close"].rolling(200).mean()

    section_label("Moving Averages")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["Date"], y=data["Close"], name="Close", line=dict(color=TEXT_MAIN, width=1.4)))
    fig.add_trace(go.Scatter(x=data["Date"], y=ma20, name="MA20", line=dict(width=1.2)))
    fig.add_trace(go.Scatter(x=data["Date"], y=ma50, name="MA50", line=dict(width=1.2)))
    fig.add_trace(go.Scatter(x=data["Date"], y=ma100, name="MA100", line=dict(width=1.2)))
    fig.add_trace(go.Scatter(x=data["Date"], y=ma200, name="MA200", line=dict(width=1.2)))
    fig.update_layout(template="plotly_dark", height=460, hovermode="x unified",
                       paper_bgcolor=BG_DARK, plot_bgcolor=BG_PANEL,
                       margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MA20", f"{ma20.iloc[-1]:.2f}")
    c2.metric("MA50", f"{ma50.iloc[-1]:.2f}")
    c3.metric("MA100", f"{ma100.iloc[-1]:.2f}")
    c4.metric("MA200", f"{ma200.iloc[-1]:.2f}")

    if ma50.iloc[-1] > ma200.iloc[-1]:
        signal_pill("🟢 GOLDEN CROSS — BULLISH", "buy")
    elif ma50.iloc[-1] < ma200.iloc[-1]:
        signal_pill("🔴 DEATH CROSS — BEARISH", "sell")
    else:
        signal_pill("🟡 NEUTRAL — WAIT", "hold")

    section_label("Relative Strength Index (RSI)")
    delta = data["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["Date"], y=rsi, name="RSI", line=dict(color=ACCENT_PRIMARY)))
    fig.add_hline(y=70, line_dash="dash", line_color=ACCENT_DOWN)
    fig.add_hline(y=30, line_dash="dash", line_color=ACCENT_UP)
    fig.update_layout(template="plotly_dark", height=320,
                       paper_bgcolor=BG_DARK, plot_bgcolor=BG_PANEL,
                       margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    current_rsi = float(rsi.iloc[-1])
    st.metric("Current RSI", f"{current_rsi:.2f}")
    if current_rsi > 70:
        signal_pill("OVERBOUGHT — POSSIBLE SELL", "sell")
    elif current_rsi < 30:
        signal_pill("OVERSOLD — POSSIBLE BUY", "buy")
    else:
        signal_pill("NEUTRAL", "hold")

    section_label("MACD")
    ema12 = data["Close"].ewm(span=12).mean()
    ema26 = data["Close"].ewm(span=26).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["Date"], y=macd, name="MACD", line=dict(color=ACCENT_PRIMARY)))
    fig.add_trace(go.Scatter(x=data["Date"], y=macd_signal, name="Signal", line=dict(color=ACCENT_WARN)))
    fig.update_layout(template="plotly_dark", height=320,
                       paper_bgcolor=BG_DARK, plot_bgcolor=BG_PANEL,
                       margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    if macd.iloc[-1] > macd_signal.iloc[-1]:
        signal_pill("MACD BUY SIGNAL", "buy")
    else:
        signal_pill("MACD SELL SIGNAL", "sell")

    section_label("Bollinger Bands")
    bb_ma = data["Close"].rolling(20).mean()
    bb_std = data["Close"].rolling(20).std()
    upper, lower = bb_ma + 2 * bb_std, bb_ma - 2 * bb_std

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["Date"], y=upper, name="Upper Band", line=dict(color=ACCENT_DOWN, width=1)))
    fig.add_trace(go.Scatter(x=data["Date"], y=lower, name="Lower Band", line=dict(color=ACCENT_UP, width=1),
                              fill="tonexty", fillcolor="rgba(148,163,184,0.06)"))
    fig.add_trace(go.Scatter(x=data["Date"], y=data["Close"], name="Close", line=dict(color=TEXT_MAIN, width=1.4)))
    fig.update_layout(template="plotly_dark", height=420,
                       paper_bgcolor=BG_DARK, plot_bgcolor=BG_PANEL,
                       margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    section_label("30-Day Rolling Volatility & Risk")
    returns = data["Close"].pct_change()
    rolling_vol = returns.rolling(30).std() * np.sqrt(252)
    fig = go.Figure(go.Scatter(x=data["Date"], y=rolling_vol, line=dict(color=ACCENT_WARN)))
    fig.update_layout(template="plotly_dark", height=320,
                       paper_bgcolor=BG_DARK, plot_bgcolor=BG_PANEL,
                       margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    latest_vol = rolling_vol.iloc[-1]
    if pd.isna(latest_vol):
        st.info("Not enough data yet to compute a 30-day volatility figure.")
    elif latest_vol < 0.20:
        signal_pill("🟢 LOW RISK", "buy")
    elif latest_vol < 0.40:
        signal_pill("🟡 MEDIUM RISK", "hold")
    else:
        signal_pill("🔴 HIGH RISK", "sell")

    section_label("Composite AI Recommendation")
    buy_score = 0
    buy_score += int(ma50.iloc[-1] > ma200.iloc[-1])
    buy_score += int(macd.iloc[-1] > macd_signal.iloc[-1])
    buy_score += int(current_rsi < 30)
    labels = {3: ("🟢 STRONG BUY", "buy"), 2: ("🟢 BUY", "buy"), 1: ("🟡 HOLD", "hold"), 0: ("🔴 SELL", "sell")}
    text, kind = labels[buy_score]
    signal_pill(f"{text}  ({buy_score}/3 indicators bullish)", kind)

# --------------------------------------------------------------------------
# TAB: MODEL PERFORMANCE  (LSTM backtest)
# --------------------------------------------------------------------------
model = None
scaler = None
data_test_scale = None

with tab_model:
    section_label("Load Model")
    model_path = None
    if model_upload is not None:
        tmp_path = os.path.join("/tmp", model_upload.name)
        with open(tmp_path, "wb") as f:
            f.write(model_upload.getbuffer())
        model_path = tmp_path
    elif os.path.exists(MODEL_PATH_DEFAULT):
        model_path = MODEL_PATH_DEFAULT

    if model_path is None:
        st.warning(
            "No trained model found. Upload a `.keras` LSTM model in the sidebar, "
            "or bundle one at `model/stock_prediction_model.keras`, to see backtest "
            "metrics and future forecasts."
        )
    else:
        try:
            with st.spinner("Loading model..."):
                model = get_model(model_path)
        except Exception as e:
            st.error(f"❌ Could not load model: {e}")

    if model is not None:
        if len(data) < 200:
            st.warning("Need at least 200 trading days of history to backtest reliably; results may be unstable.")

        data_train = pd.DataFrame(data["Close"][: int(len(data) * 0.80)])
        data_test_raw = pd.DataFrame(data["Close"][int(len(data) * 0.80):])

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(data_train)

        past_100 = data_train.tail(100)
        data_test = pd.concat([past_100, data_test_raw], ignore_index=True)
        data_test_scale = scaler.transform(data_test)

        x, y = [], []
        for i in range(100, data_test_scale.shape[0]):
            x.append(data_test_scale[i - 100:i])
            y.append(data_test_scale[i, 0])
        x, y = np.array(x), np.array(y)

        if len(x) == 0:
            st.warning("Not enough test-set data to run a backtest for this symbol.")
        else:
            with st.spinner("Running backtest..."):
                predict = model.predict(x, verbose=0)

            scale = 1 / scaler.scale_[0]
            predict = predict * scale
            y_actual = y * scale

            rmse = np.sqrt(mean_squared_error(y_actual, predict))
            mae = mean_absolute_error(y_actual, predict)
            mse = mean_squared_error(y_actual, predict)
            r2 = r2_score(y_actual, predict)
            accuracy = 100 - (rmse / np.mean(y_actual) * 100)

            section_label("Backtest Metrics")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("RMSE", f"{rmse:.2f}")
            c2.metric("MAE", f"{mae:.2f}")
            c3.metric("MSE", f"{mse:.2f}")
            c4.metric("R² Score", f"{r2:.3f}")
            c5.metric("Accuracy", f"{accuracy:.2f}%")

            section_label("Actual vs Predicted Price (Test Set)")
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=y_actual, name="Actual", line=dict(color=ACCENT_UP)))
            fig.add_trace(go.Scatter(y=predict.flatten(), name="Predicted", line=dict(color=ACCENT_DOWN)))
            fig.update_layout(template="plotly_dark", height=440, hovermode="x unified",
                               paper_bgcolor=BG_DARK, plot_bgcolor=BG_PANEL,
                               margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

            if accuracy >= 95:
                st.success("Excellent model performance.")
            elif accuracy >= 90:
                st.success("Very good model performance.")
            elif accuracy >= 80:
                st.warning("Good model performance — consider more training data or tuning.")
            else:
                st.error("Model needs improvement — accuracy is below an acceptable threshold.")

# --------------------------------------------------------------------------
# TAB: FUTURE FORECAST
# --------------------------------------------------------------------------
with tab_forecast:
    if model is None or scaler is None or data_test_scale is None:
        st.info("Load a model in the **Model Performance** tab first to generate a future forecast.")
    else:
        with st.spinner(f"Forecasting {future_days} days ahead..."):
            future_prices = run_future_forecast(scaler, data_test_scale[-100:], model, future_days)

        future_df = pd.DataFrame({
            "Day": range(1, future_days + 1),
            "Predicted Price": future_prices.flatten(),
        })

        section_label(f"{future_days}-Day Forecast")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=future_df["Day"], y=future_df["Predicted Price"],
                                  name="Forecast", line=dict(color=ACCENT_PRIMARY, width=2.5)))
        fig.update_layout(template="plotly_dark", height=440,
                           paper_bgcolor=BG_DARK, plot_bgcolor=BG_PANEL,
                           xaxis_title="Days Ahead", yaxis_title="Predicted Price",
                           margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

        start_p, end_p = future_df["Predicted Price"].iloc[0], future_df["Predicted Price"].iloc[-1]
        trend_change = ((end_p - start_p) / start_p) * 100 if start_p else 0

        if end_p > start_p:
            signal_pill(f"🟢 BULLISH — projected {trend_change:+.2f}% over {future_days} days", "buy")
        elif end_p < start_p:
            signal_pill(f"🔴 BEARISH — projected {trend_change:+.2f}% over {future_days} days", "sell")
        else:
            signal_pill("🟡 SIDEWAYS / STABLE", "hold")

        c1, c2, c3 = st.columns(3)
        c1.metric("Highest Forecast", f"${future_df['Predicted Price'].max():.2f}")
        c2.metric("Lowest Forecast", f"${future_df['Predicted Price'].min():.2f}")
        c3.metric("Average Forecast", f"${future_df['Predicted Price'].mean():.2f}")

        section_label("Forecast Table")
        st.dataframe(future_df, use_container_width=True, height=280)
        st.download_button(
            "📥 Download Forecast (CSV)",
            data=future_df.to_csv(index=False),
            file_name=f"{stock}_future_prediction.csv",
            mime="text/csv",
        )

        st.caption(
            "⚠️ This forecast is generated by rolling the LSTM model forward one step at a "
            "time. Errors compound over longer horizons — treat long-range forecasts as "
            "indicative of trend direction, not precise price targets."
        )

# --------------------------------------------------------------------------
# TAB: COMPANY & NEWS
# --------------------------------------------------------------------------
with tab_company:
    info = get_company_info(stock)

    section_label("Company Profile")
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Name:**", info.get("longName", "N/A"))
        st.write("**Sector:**", info.get("sector", "N/A"))
        st.write("**Industry:**", info.get("industry", "N/A"))
        st.write("**Country:**", info.get("country", "N/A"))
        st.write("**Website:**", info.get("website", "N/A"))
    with c2:
        st.write("**Market Cap:**", f"{info.get('marketCap', 'N/A'):,}" if isinstance(info.get("marketCap"), (int, float)) else "N/A")
        st.write("**P/E Ratio:**", info.get("trailingPE", "N/A"))
        st.write("**Dividend Yield:**", info.get("dividendYield", "N/A"))
        st.write("**52W High:**", info.get("fiftyTwoWeekHigh", "N/A"))
        st.write("**52W Low:**", info.get("fiftyTwoWeekLow", "N/A"))

    week_high, week_low = info.get("fiftyTwoWeekHigh"), info.get("fiftyTwoWeekLow")
    if week_high and week_low and week_high != week_low:
        progress = min(max((current_price - week_low) / (week_high - week_low), 0.0), 1.0)
        section_label("Price Position in 52-Week Range")
        st.progress(progress)
        st.caption(f"${week_low} ── ${current_price:.2f} ── ${week_high}")

    summary = info.get("longBusinessSummary")
    if summary:
        section_label("Business Summary")
        st.write(summary)

    section_label("Latest News")
    entries = get_news(stock)
    if entries:
        for news in entries:
            st.markdown(f"**{news.title}**")
            st.caption(f"{news.get('published', '')}")
            st.markdown(f"[Read more]({news.link})")
            st.markdown("---")
    else:
        st.info("No recent news found for this symbol.")

# --------------------------------------------------------------------------
# TAB: COMPARE
# --------------------------------------------------------------------------
with tab_compare:
    if not compare_stocks:
        st.info("Select one or more symbols in the sidebar to compare.")
    else:
        comparison = pd.DataFrame()
        for symbol in compare_stocks:
            df = get_history(symbol, START, END)
            if not df.empty:
                comparison[symbol] = df.set_index("Date")["Close"]

        section_label("Closing Price Comparison")
        fig = go.Figure()
        for symbol in comparison.columns:
            fig.add_trace(go.Scatter(x=comparison.index, y=comparison[symbol], name=symbol))
        fig.update_layout(template="plotly_dark", height=460, hovermode="x unified",
                           paper_bgcolor=BG_DARK, plot_bgcolor=BG_PANEL,
                           margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

        section_label("Recent Values")
        st.dataframe(comparison.tail(), use_container_width=True)

        section_label("Total Return (%)")
        performance = {}
        for symbol in comparison.columns:
            series = comparison[symbol].dropna()
            if len(series) > 1:
                performance[symbol] = (series.iloc[-1] - series.iloc[0]) / series.iloc[0] * 100

        if performance:
            perf_df = pd.DataFrame({"Stock": performance.keys(), "Return %": performance.values()})
            st.dataframe(perf_df, use_container_width=True)
            best = perf_df.loc[perf_df["Return %"].idxmax()]
            st.success(f"🏆 Best performer: **{best['Stock']}** ({best['Return %']:.2f}%)")

# --------------------------------------------------------------------------
# TAB: PORTFOLIO (simple simulation)
# --------------------------------------------------------------------------
with tab_portfolio:
    section_label("Buy & Hold Simulation — 100 Shares")
    shares = 100
    investment = shares * float(data["Close"].iloc[0])
    portfolio_value = shares * current_price
    profit = portfolio_value - investment
    return_pct = (profit / investment) * 100 if investment else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Investment", f"${investment:,.2f}")
    c2.metric("Current Value", f"${portfolio_value:,.2f}")
    c3.metric("Profit / Loss", f"${profit:,.2f}")
    c4.metric("Return", f"{return_pct:.2f}%")

    if profit > 0:
        signal_pill("🟢 IN PROFIT", "buy")
    elif profit < 0:
        signal_pill("🔴 IN LOSS", "sell")
    else:
        signal_pill("🟡 BREAK-EVEN", "hold")

    st.caption("Illustrative only: assumes 100 shares bought at the start of the loaded history and held to today.")

# --------------------------------------------------------------------------
# TAB: ABOUT
# --------------------------------------------------------------------------
with tab_about:
    st.markdown(
        """
        ### 📈 Stock Market Prediction using Deep Learning (LSTM)

        **Developed by:** Rajesh Kumar Mahto
        **Course:** B.Tech CSE, Jharkhand Rai University
        **Model:** LSTM (Long Short-Term Memory)
        **Framework:** Streamlit
        **Data Source:** Yahoo Finance

        #### Technologies Used
        Python · TensorFlow / Keras · Streamlit · Pandas · NumPy · Plotly · Matplotlib · Scikit-learn · yfinance
        """
    )
    st.warning(
        "⚠️ **Disclaimer:** This project is developed for educational purposes only. "
        "Stock market investments involve risk. Predictions generated by the LSTM model "
        "should not be considered financial advice."
    )