# ==========================================================
# Rossmann Executive Forecasting Dashboard (Refactored)
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

warnings.filterwarnings("ignore")

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Rossmann Executive Dashboard",
    layout="wide",
    page_icon="📊"
)

st.title("📊 Rossmann Executive Sales Forecasting Dashboard")
st.markdown("---")

# ==========================================================
# LOAD DATA (KEEPING YOUR ORIGINAL STRUCTURE)
# ==========================================================

@st.cache_data
def load_data():
    df = pd.read_csv("train.csv", parse_dates=["Date"])
    return df

data = load_data()

# ==========================================================
# SIDEBAR SETTINGS
# ==========================================================

st.sidebar.header("⚙️ Forecast Settings")

store_list = sorted(data["Store"].unique())
selected_store = st.sidebar.selectbox("Select Store", ["All Stores"] + store_list)

forecast_horizon = st.sidebar.slider("Forecast Horizon (Days)", 7, 60, 30)

# ==========================================================
# TABS
# ==========================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Executive Summary",
    "📈 Forecasting",
    "📉 Model Comparison",
    "🗂 Data Explorer"
])

# ==========================================================
# TAB 1 – EXECUTIVE SUMMARY
# ==========================================================

with tab1:

    total_sales = data.groupby("Store")["Sales"].sum()

    best_store = total_sales.idxmax()
    worst_store = total_sales.idxmin()

    col1, col2, col3 = st.columns(3)
    col1.metric("🏆 Best Store", f"Store {best_store}")
    col2.metric("📉 Worst Store", f"Store {worst_store}")
    col3.metric("💰 Total Sales", f"{total_sales.sum():,.0f}")

    st.subheader("Top 10 Stores by Sales")
    st.bar_chart(total_sales.sort_values(ascending=False).head(10))


# ==========================================================
# TAB 2 – FORECASTING
# ==========================================================

with tab2:

    if selected_store == "All Stores":
        series = data.groupby("Date")["Sales"].sum()
    else:
        df_store = data[data["Store"] == selected_store]
        series = df_store.set_index("Date")["Sales"]

    train_series = series[:-forecast_horizon]
    test_series = series[-forecast_horizon:]

    results = {}

    # ------------------ ARIMA ------------------
    model_arima = ARIMA(train_series, order=(1,1,1)).fit()
    forecast_arima = model_arima.forecast(forecast_horizon)
    results["ARIMA"] = np.sqrt(mean_squared_error(test_series, forecast_arima))

    # ------------------ SARIMA ------------------
    model_sarima = SARIMAX(train_series,
                           order=(1,1,1),
                           seasonal_order=(1,1,1,7)).fit()
    forecast_sarima = model_sarima.forecast(forecast_horizon)
    results["SARIMA"] = np.sqrt(mean_squared_error(test_series, forecast_sarima))

    # ------------------ Holt-Winters ------------------
    model_hw = ExponentialSmoothing(
        train_series,
        trend="add",
        seasonal="add",
        seasonal_periods=7
    ).fit()
    forecast_hw = model_hw.forecast(forecast_horizon)
    results["Holt-Winters"] = np.sqrt(mean_squared_error(test_series, forecast_hw))

    # ------------------ BEST MODEL ------------------

    best_model = min(results, key=results.get)
    st.success(f"🏆 Best Model (Lowest RMSE): {best_model}")

    forecast = {
        "ARIMA": forecast_arima,
        "SARIMA": forecast_sarima,
        "Holt-Winters": forecast_hw
    }[best_model]

    # ==================================================
    # KPI METRICS
    # ==================================================

    mae = mean_absolute_error(test_series, forecast)
    rmse = np.sqrt(mean_squared_error(test_series, forecast))
    mape = np.mean(np.abs((test_series - forecast) / test_series)) * 100

    st.markdown("## 📊 Model Performance")

    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", f"{mae:,.2f}")
    c2.metric("RMSE", f"{rmse:,.2f}")
    c3.metric("MAPE (%)", f"{mape:,.2f}")

    # ==================================================
    # ERROR BAR CHART
    # ==================================================

    fig_metrics = go.Figure(
        data=[
            go.Bar(
                x=["MAE", "RMSE", "MAPE"],
                y=[mae, rmse, mape]
            )
        ]
    )

    fig_metrics.update_layout(
        title="Error Metrics Comparison",
        template="plotly_white",
        height=400
    )

    st.plotly_chart(fig_metrics, use_container_width=True)

    # ==================================================
    # FORECAST PLOT
    # ==================================================

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=train_series.index,
        y=train_series.values,
        mode="lines",
        name="Historical"
    ))

    fig.add_trace(go.Scatter(
        x=test_series.index,
        y=test_series.values,
        mode="lines",
        name="Actual"
    ))

    fig.add_trace(go.Scatter(
        x=test_series.index,
        y=forecast.values,
        mode="lines",
        name="Forecast"
    ))

    fig.update_layout(template="plotly_white", height=500)
    st.plotly_chart(fig, use_container_width=True)

    # ==================================================
    # RESIDUAL ANALYSIS
    # ==================================================

    residuals = test_series - forecast

    fig_res = go.Figure()
    fig_res.add_trace(go.Scatter(
        x=test_series.index,
        y=residuals,
        mode="lines",
        name="Residuals"
    ))

    fig_res.update_layout(
        title="Residual Analysis",
        template="plotly_white",
        height=400
    )

    st.plotly_chart(fig_res, use_container_width=True)


# ==========================================================
# TAB 3 – MODEL COMPARISON
# ==========================================================

with tab3:

    comparison_df = pd.DataFrame(results.items(), columns=["Model", "RMSE"])

    fig_compare = go.Figure(
        data=[go.Bar(
            x=comparison_df["Model"],
            y=comparison_df["RMSE"]
        )]
    )

    fig_compare.update_layout(template="plotly_white")
    st.plotly_chart(fig_compare, use_container_width=True)


# ==========================================================
# TAB 4 – DATA EXPLORER
# ==========================================================

with tab4:

    st.dataframe(data.head(100), use_container_width=True)
    st.write("Dataset Shape:", data.shape)
    st.subheader("Missing Values")
    st.dataframe(data.isna().sum())
