#!/usr/bin/env python
import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from datetime import datetime, timedelta

# Streamlit settings
st.set_page_config(page_title="Earthquake Predictor Pro", layout="centered")
st.title("🌍 Advanced Earthquake Prediction System")
st.markdown("""
Predicts earthquake probabilities and magnitudes using XGBoost with real-time USGS data.
Uses rolling averages and location-based time-series forecasting.
""")

# Constants
DAYS_OUT = 7
MODEL_PARAMS = {
    'objective': 'binary:logistic',
    'max_depth': 5,
    'eta': 0.1,
    'eval_metric': 'auc'
}


@st.cache_data
def load_and_preprocess_data():
    """Load and preprocess earthquake data from USGS"""
    url = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_month.csv"
    df = pd.read_csv(url)

    # Clean and transform data
    df['time'] = pd.to_datetime(df['time'])
    df['date'] = df['time'].dt.date.astype(str)
    df = df.sort_values('time')

    # Process locations
    temp_place = df['place'].str.split(', ', expand=True)
    df['region'] = temp_place[1].fillna('Unknown')

    # Calculate region averages
    region_coords = df.groupby('region')[['latitude', 'longitude']].mean().reset_index()
    df = pd.merge(df, region_coords, on='region', suffixes=('', '_mean'))

    # Create rolling features
    features = []
    for region in df['region'].unique():
        region_df = df[df['region'] == region].copy()
        for window in [7, 15, 22]:
            region_df[f'mag_avg_{window}'] = region_df['mag'].rolling(window=window).mean()
            region_df[f'depth_avg_{window}'] = region_df['depth'].rolling(window=window).mean()
        features.append(region_df)

    df = pd.concat(features).dropna()

    # Create shifted target
    df['mag_outcome'] = df.groupby('region')['mag_avg_7'].shift(-DAYS_OUT)
    df['quake_prob'] = (df['mag_outcome'] > 2.5).astype(int)

    return df


def train_model(df):
    """Train XGBoost model and return metrics"""
    features = [f for f in df.columns if any(x in f for x in ['mag_avg', 'depth_avg'])]
    X = df[features]
    y = df['quake_prob']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    model = xgb.train(
        MODEL_PARAMS,
        dtrain,
        num_boost_round=1000,
        early_stopping_rounds=50,
        evals=[(dtest, 'test')],
        verbose_eval=False
    )

    # Calculate metrics
    preds = model.predict(dtest)
    y_pred = (preds > 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, preds)

    # Print detailed metrics to console
    print("\n" + "=" * 50)
    print("MODEL PERFORMANCE METRICS")
    print("=" * 50)
    print(f"Accuracy: {acc:.4f}")
    print(f"AUC-ROC: {auc:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("=" * 50 + "\n")

    st.sidebar.markdown("### Model Metrics")
    st.sidebar.write(f"Accuracy: {acc:.2%}")
    st.sidebar.write(f"AUC-ROC: {auc:.2%}")

    return model, features


def generate_predictions(model, df, features):
    """Generate future predictions for all regions"""
    predictions = []
    start_date = datetime.today() + timedelta(days=1)

    for region in df['region'].unique():
        region_df = df[df['region'] == region].iloc[-DAYS_OUT:].copy()
        if region_df.empty:
            continue

        # Generate predictions
        dlive = xgb.DMatrix(region_df[features])
        probs = model.predict(dlive)

        # Create date range for predictions
        prediction_dates = [start_date + timedelta(days=i) for i in range(len(probs))]

        for i, date in enumerate(prediction_dates):
            try:
                predictions.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'region': region,
                    'latitude': region_df['latitude_mean'].iloc[0],
                    'longitude': region_df['longitude_mean'].iloc[0],
                    'probability': probs[i],
                    'expected_mag': region_df['mag_avg_7'].iloc[i] if i < len(region_df) else region_df[
                        'mag_avg_7'].mean()
                })
            except Exception as e:
                st.warning(f"Skipping {region} due to: {str(e)}")
                continue

    return pd.DataFrame(predictions)


def create_prediction_map(predictions):
    """Create interactive map visualization"""
    st.markdown("### 🗺️ High-Risk Regions")

    # Get valid center coordinates
    try:
        map_center = [
            predictions['latitude'].median(),
            predictions['longitude'].median()
        ]
    except:
        map_center = [20, 0]  # Default center near equator

    m = folium.Map(location=map_center, zoom_start=2)

    for _, row in predictions.iterrows():
        if row['probability'] > 0.3 and not np.isnan(row['latitude']):
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=row['expected_mag'] * 3,
                color='#ff0000' if row['probability'] > 0.7 else '#ffa500',
                fill=True,
                fill_opacity=0.7,
                popup=f"""
                <b>{row['region']}</b><br>
                📅 {row['date']}<br>
                ⚠️ Probability: {row['probability']:.1%}<br>
                ⚡ Magnitude: {row['expected_mag']:.1f}
                """
            ).add_to(m)

    return m


# Main app flow
df = load_and_preprocess_data()

# Sidebar controls
st.sidebar.header("Controls")
probability_threshold = st.sidebar.slider(
    "Alert Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.3,
    step=0.05,
    help="Minimum probability to show on map"
)

if st.button("🚀 Generate Advanced Predictions"):
    with st.spinner("Training model and generating forecasts..."):
        try:
            model, features = train_model(df)
            predictions = generate_predictions(model, df, features)

            # Filter and store predictions
            predictions = predictions.dropna()
            predictions = predictions[predictions['probability'] >= 0]
            st.session_state.predictions = predictions

            st.success(f"✅ Generated {len(predictions)} predictions!")

        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

# Display results
if 'predictions' in st.session_state:
    # Filter by threshold
    filtered_preds = st.session_state.predictions[
        st.session_state.predictions['probability'] >= probability_threshold
        ]

    if not filtered_preds.empty:
        # Show map
        st_folium(
            create_prediction_map(filtered_preds),
            width=800,
            height=500
        )

        # Show data table
        st.markdown("### 📊 Prediction Details")
        st.dataframe(
            filtered_preds.sort_values('probability', ascending=False),
            column_config={
                'probability': st.column_config.ProgressColumn(
                    format="%.1f%%",
                    min_value=0,
                    max_value=1
                ),
                'expected_mag': st.column_config.NumberColumn(
                    format="%.1f"
                )
            },
            hide_index=True
        )
    else:
        st.warning("No predictions meet the current threshold")

# Raw data explorer
with st.expander("🔍 View Raw Data"):
    st.dataframe(df[['date', 'region', 'mag', 'depth']].tail(100))
if 'predictions' in st.session_state and not filtered_preds.empty:
    st.markdown("---")
    st.markdown("## 🚨 Prescriptive Analytics Dashboard")

    # Calculate risk metrics
    high_risk = filtered_preds[filtered_preds['probability'] > 0.7]
    medium_risk = filtered_preds[(filtered_preds['probability'] > 0.3) &
                                 (filtered_preds['probability'] <= 0.7)]

    # Create columns layout
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🔴 High Risk Zones")
        if not high_risk.empty:
            for _, row in high_risk.iterrows():
                st.markdown(f"""
                **{row['region']}**  
                📅 {row['date']}  
                ⚠️ Probability: {row['probability']:.1%}  
                🌀 Expected Magnitude: {row['expected_mag']:.1f}  
                ✅ **Recommended Actions**:  
                - Immediate evacuation planning  
                - Emergency response team deployment  
                - Critical infrastructure inspection  
                - Public alert system activation
                """)
        else:
            st.info("No high risk zones detected")

    with col2:
        st.markdown("### 🟠 Medium Risk Zones")
        if not medium_risk.empty:
            for _, row in medium_risk.iterrows():
                st.markdown(f"""
                **{row['region']}**  
                📅 {row['date']}  
                ⚠️ Probability: {row['probability']:.1%}  
                🌀 Expected Magnitude: {row['expected_mag']:.1f}  
                ✅ **Recommended Actions**:  
                - Community preparedness drills  
                - Emergency supply distribution  
                - Building safety audits  
                - Risk awareness campaigns
                """)
        else:
            st.info("No medium risk zones detected")

    with col3:
        st.markdown("### 📊 Risk Mitigation Resources")
        st.markdown("""
        **Emergency Protocols**:  
        🆘 [Earthquake Safety Guidelines](https://www.ready.gov/earthquakes)  
        🏥 [First Aid Procedures](https://www.redcross.org/)  

        **Preparedness Checklist**:  
        ✅ Emergency water/food (3-day supply)  
        ✅ First aid kit and medications  
        ✅ Flashlight + extra batteries  
        ✅ Important documents backup  

        **Real-time Monitoring**:  
        🌐 [USGS Live Earthquake Map](https://earthquake.usgs.gov/earthquakes/map/)
        """)

    # Add timeline for preparedness actions
    st.markdown("---")
    st.markdown("### 🗓️ Preparedness Timeline")
    timeline_cols = st.columns(5)
    with timeline_cols[0]:
        st.markdown("**Immediate** (0-24hrs):")
        st.markdown("- Activate emergency plan\n- Secure hazardous items")
    with timeline_cols[1]:
        st.markdown("**Short-term** (1-3 days):")
        st.markdown("- Distribute supplies\n- Damage assessment")
    with timeline_cols[2]:
        st.markdown("**Medium-term** (1 week):")
        st.markdown("- Temporary shelters\n- Infrastructure repairs")
    with timeline_cols[3]:
        st.markdown("**Long-term** (1 month):")
        st.markdown("- Reconstruction\n- Mental health support")
    with timeline_cols[4]:
        st.markdown("**Ongoing**:")
        st.markdown("- Seismic monitoring\n- Community training")

# Raw data explorer (keep this at the end)
with st.expander("🔍 View Raw Data"):
    st.dataframe(df[['date', 'region', 'mag', 'depth']].tail(100))