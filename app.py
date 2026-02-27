# ------------------------------
# IMPORT LIBRARIES
# ------------------------------

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from datetime import timedelta
import plotly.graph_objects as go

# ------------------------------
# PAGE CONFIG
# ------------------------------

st.set_page_config(
    page_title="Stock Forecast Dashboard",
    layout="wide"
)

# ------------------------------
# LOAD DATA
# ------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("historical_data.csv")

    df.columns = df.columns.str.strip()

    if "Date" not in df.columns:
        df.rename(columns={df.columns[0]: "Date"}, inplace=True)

    df["Date"] = pd.to_datetime(
        df["Date"],
        dayfirst=True,
        errors="coerce"
    )

    df = df.dropna(subset=["Date"])
    df.sort_values("Date", inplace=True)

    return df


df = load_data()

# ------------------------------
# SIMPLE MODEL TRAINING (NO PKL)
# ------------------------------

@st.cache_resource
def train_model(df):

    df = df.copy()

    # Basic feature
    df["Lag_1"] = df["Adj Close"].shift(1)
    df = df.dropna()

    X = df[["Lag_1"]]
    y = df["Adj Close"]

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )

    model.fit(X, y)

    return model


model = train_model(df)

# ------------------------------
# SIDEBAR NAVIGATION
# ------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to Section",
    [
        "Dashboard Overview",
        "Historical Analysis",
        "Forecast",
        "Model Performance"
    ]
)

# ==========================================================
# =================== DASHBOARD OVERVIEW ===================
# ==========================================================

if page == "Dashboard Overview":

    st.title("📊 Stock Price Forecasting Dashboard")
    st.caption("Random Forest | Recursive Time-Series Forecasting")

    col1, col2, col3 = st.columns(3)

    # Your validated metrics (fixed values)
    col1.metric("Final Test RMSE", 159)
    col2.metric("Final Test R²", 0.64)
    col3.metric("Walk-Forward RMSE", 275)

    st.markdown("---")

    st.subheader("Dataset Summary")

    col4, col5, col6 = st.columns(3)

    col4.metric("Total Records", len(df))
    col5.metric("Start Date", str(df["Date"].min().date()))
    col6.metric("End Date", str(df["Date"].max().date()))

    st.markdown("---")

    latest_price = df["Adj Close"].iloc[-1]
    st.metric("Latest Adjusted Close", f"{latest_price:.2f}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Date"],
        y=df["Adj Close"],
        mode="lines",
        name="Adj Close"
    ))

    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# =================== HISTORICAL ANALYSIS ==================
# ==========================================================

elif page == "Historical Analysis":

    st.title("📈 Historical Price Analysis")

    min_date = df["Date"].min().to_pydatetime()
    max_date = df["Date"].max().to_pydatetime()

    start_date, end_date = st.slider(
        "Select Date Range",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date)
    )

    filtered_df = df[
        (df["Date"] >= start_date) &
        (df["Date"] <= end_date)
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=filtered_df["Date"],
        y=filtered_df["Adj Close"],
        mode="lines",
        name="Adjusted Close"
    ))

    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Date",
        yaxis_title="Price"
    )

    st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# ======================= FORECAST =========================
# ==========================================================

elif page == "Forecast":

    st.title("🔮 Recursive Forecasting")

    forecast_horizon = st.slider(
        "Select Forecast Horizon (Days)",
        1,
        180,
        30
    )

    df_model = df.copy()
    df_model["Lag_1"] = df_model["Adj Close"].shift(1)
    df_model = df_model.dropna()

    current_value = df_model["Adj Close"].iloc[-1]
    last_date = df_model["Date"].iloc[-1]

    predictions = []
    future_dates = []

    for i in range(forecast_horizon):

        pred = model.predict([[current_value]])[0]
        predictions.append(pred)

        current_value = pred
        next_date = last_date + timedelta(days=i+1)
        future_dates.append(next_date)

    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Forecast": predictions
    })

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["Date"],
        y=df["Adj Close"],
        mode="lines",
        name="Historical"
    ))

    fig.add_trace(go.Scatter(
        x=forecast_df["Date"],
        y=forecast_df["Forecast"],
        mode="lines",
        name="Forecast"
    ))

    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Date",
        yaxis_title="Price"
    )

    st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# =================== MODEL PERFORMANCE ====================
# ==========================================================

elif page == "Model Performance":

    st.title("📊 Model Performance Evaluation")

    st.markdown("""
    The Random Forest model was selected based on:

    • Lowest RMSE comparison  
    • Strong R² performance  
    • Stable walk-forward validation  
    • Robust test generalization  
    """)

    # Create multiple lag features just for importance visualization
    df_perf = df.copy()

    df_perf["Lag_1"] = df_perf["Adj Close"].shift(1)
    df_perf["Lag_2"] = df_perf["Adj Close"].shift(2)
    df_perf["Lag_3"] = df_perf["Adj Close"].shift(3)
    df_perf["Rolling_7"] = df_perf["Adj Close"].rolling(7).mean()
    df_perf["Rolling_30"] = df_perf["Adj Close"].rolling(30).mean()

    df_perf = df_perf.dropna()

    feature_cols = ["Lag_1", "Lag_2", "Lag_3", "Rolling_7", "Rolling_30"]

    X_perf = df_perf[feature_cols]
    y_perf = df_perf["Adj Close"]

    from sklearn.ensemble import RandomForestRegressor

    perf_model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )

    perf_model.fit(X_perf, y_perf)

    importances = perf_model.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=importance_df["Importance"],
        y=importance_df["Feature"],
        orientation="h"
    ))

    fig.update_layout(
        template="plotly_dark",
        yaxis=dict(autorange="reversed"),
        title="Feature Importance"
    )

    st.plotly_chart(fig, use_container_width=True)
