import streamlit as st
import pandas as pd
import plotly.express as px
from src.weather import get_weather
from src.satellite import get_sentinel_ndvi
from src.risk import assess_risks
from src.yield_model import train_model, predict_yield, create_demo_dataset, SAMPLE_PATH, FEATURES, TARGET

st.set_page_config(page_title="Climate Yield Engine", page_icon="🌱", layout="wide")
st.title("🌱 Predictive Climate & Micro-Location Yield Engine")
st.caption("A buildathon MVP combining weather forecasts, optional Sentinel-2 NDVI, transparent risk rules and a demonstrator yield model.")

with st.sidebar:
    st.header("Farm location")
    lat = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=12.9716, format="%.5f")
    lon = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=77.5946, format="%.5f")
    forecast_days = st.slider("Forecast window (days)", 3, 16, 7)
    crop_name = st.selectbox("Crop", ["Rice", "Wheat", "Maize", "Millet"])
    crop_codes = {"Rice": 0, "Wheat": 1, "Maize": 2, "Millet": 3}
    use_satellite = st.checkbox("Attempt Sentinel-2 NDVI lookup", value=False)
    run = st.button("Run climate analysis", type="primary", use_container_width=True)

st.info("Enter a farm's coordinates. Example coordinates are prefilled for Bengaluru; replace them with your field location.")

if run:
    try:
        with st.spinner("Fetching forecast and analysing the location..."):
            weather = get_weather(lat, lon, forecast_days)
        st.session_state["weather"] = weather
        st.session_state["location"] = (lat, lon)
        st.session_state["crop"] = crop_name
        st.session_state["satellite"] = None
        if use_satellite:
            with st.spinner("Searching Sentinel-2 imagery (this can take a while)..."):
                st.session_state["satellite"] = get_sentinel_ndvi(lat, lon)
    except Exception as e:
        st.error(f"Could not retrieve weather data: {e}")

weather = st.session_state.get("weather")
if weather is not None:
    lat0, lon0 = st.session_state["location"]
    sat = st.session_state.get("satellite")
    soil_col = "soil_moisture_0_to_7cm"
    soil_value = None
    if soil_col in weather and weather[soil_col].notna().any():
        soil_value = float(weather[soil_col].dropna().mean())

    st.subheader("Location overview")
    a, b, c = st.columns(3)
    a.metric("Latitude", f"{lat0:.4f}")
    b.metric("Longitude", f"{lon0:.4f}")
    c.metric("Crop selected", st.session_state.get("crop", "—"))

    st.subheader("Weather forecast")
    chart_df = weather[["time", "temperature_2m_max", "temperature_2m_min"]].melt(
        id_vars="time", var_name="temperature", value_name="°C"
    )
    st.plotly_chart(px.line(chart_df, x="time", y="°C", color="temperature",
                            title="Daily temperature forecast"), use_container_width=True)
    st.plotly_chart(px.bar(weather, x="time", y="precipitation_sum",
                           title="Daily forecast precipitation (mm)"), use_container_width=True)
    st.dataframe(weather, use_container_width=True, hide_index=True)

    st.subheader("Satellite vegetation indicator")
    if sat:
        if sat.get("ndvi") is not None:
            st.metric("Sentinel-2 NDVI", f"{sat['ndvi']:.3f}")
            st.caption(f"Scene date: {sat['scene_date']} · Cloud cover: {sat['cloud_cover']}%")
        else:
            st.warning(sat.get("message", "No NDVI available."))
    else:
        st.caption("Satellite lookup was not requested. Enable it in the sidebar to attempt a Sentinel-2 NDVI lookup.")

    st.subheader("Climate risk assessment")
    risks, recommendations, metrics = assess_risks(weather, soil_value)
    m1, m2, m3 = st.columns(3)
    m1.metric("Forecast rainfall", f"{metrics['forecast_rain_mm']} mm")
    m2.metric("Mean daily max temperature", f"{metrics['mean_daily_max_temp_c']} °C")
    m3.metric("Mean rain probability", f"{metrics['mean_rain_probability_pct']}%")
    risk_df = pd.DataFrame(risks, columns=["Risk", "Level", "Explanation"])
    st.dataframe(risk_df, use_container_width=True, hide_index=True)
    st.markdown("**Suggested mitigation actions**")
    for item in recommendations:
        st.write("• " + item)

    st.subheader("Yield prediction demonstrator")
    st.warning("The default training data is synthetic. Its prediction is only a software demonstration and must NOT be used for farm decisions. Replace it with real historical crop-yield data to produce meaningful estimates.")
    ndvi_value = sat.get("ndvi") if sat and sat.get("ndvi") is not None else 0.55
    seasonal_rain = st.number_input("Expected seasonal rainfall (mm)", 0.0, 3000.0, 650.0)
    mean_temp = st.number_input("Expected seasonal mean temperature (°C)", 0.0, 50.0, 28.0)
    moisture = st.number_input("Soil moisture (fraction, e.g. 0.25)", 0.0, 1.0, soil_value if soil_value is not None else 0.25)
    ndvi_input = st.number_input("NDVI (0 to 1)", 0.0, 1.0, float(ndvi_value))
    if st.button("Estimate demo yield"):
        try:
            features = {
                "seasonal_rainfall_mm": seasonal_rain,
                "mean_temp_c": mean_temp,
                "soil_moisture": moisture,
                "ndvi": ndvi_input,
                "crop_encoded": crop_codes[st.session_state.get("crop", "Rice")]
            }
            prediction = predict_yield(features)
            st.metric("Demonstration yield estimate", f"{prediction:.2f} tonnes/hectare")
        except Exception as e:
            st.error(f"Yield model error: {e}")

    with st.expander("Train with your own historical yield CSV"):
        st.write("CSV must contain these columns:")
        st.code(", ".join(FEATURES + [TARGET]))
        uploaded = st.file_uploader("Upload labelled training CSV", type=["csv"])
        if uploaded is not None:
            import tempfile, os
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                tmp.write(uploaded.getvalue())
                temp_path = tmp.name
            if st.button("Train model using uploaded data"):
                try:
                    result = train_model(temp_path)
                    st.success(f"Model trained on {result['rows']} rows. MAE: {result['mae']:.3f}; R²: {result['r2']:.3f}")
                    st.caption("Metrics are measured on a held-out random test split. For a serious agricultural model, validate across seasons/locations to avoid leakage.")
                except Exception as e:
                    st.error(str(e))
                finally:
                    os.unlink(temp_path)

st.divider()
st.caption("Prototype only. Forecasts and satellite indicators can be incomplete. Confirm critical decisions with local agronomists, official advisories and on-field measurements.")
