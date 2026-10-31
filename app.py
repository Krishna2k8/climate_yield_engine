import streamlit as st
import pandas as pd
import plotly.express as px
from src.weather import get_weather
from src.satellite import get_sentinel_ndvi
from src.satellite_mosaic import get_farm_zone_ndvi
from src.risk import assess_risks
from src.yield_model import train_model, predict_yield, create_demo_dataset, SAMPLE_PATH, FEATURES, TARGET
import plotly.express as px
import pydeck as pdk
from src.farm_boundary import load_farm_geojson
from src.sensors import load_sensor_data, summarize_sensor_data
from src.zone_intelligence import assess_zone_risks
from src.recommendations import generate_recommendations

st.set_page_config(page_title="Climate Yield Engine", page_icon="🌱", layout="wide")

# ==============================
# CUSTOM DASHBOARD THEME
# ==============================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {
    background-color: #081810;
    color: #E8F3E9;
    font-family: 'Inter', sans-serif;
}

/* Main content */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1600px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #0D2419;
    border-right: 1px solid #1E3B29;
}

[data-testid="stSidebar"] * {
    color: #E8F3E9;
}

/* Headings */
h1, h2, h3 {
    color: #F1F8F2 !important;
    font-weight: 700 !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background-color: #122D1F;
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #284934;
}

[data-testid="stMetricLabel"] {
    color: #A9C6AD !important;
}

[data-testid="stMetricValue"] {
    color: #A6E36B !important;
}

/* Buttons */
.stButton > button {
    background-color: #8CCF55;
    color: #081810;
    border: none;
    border-radius: 10px;
    font-weight: 700;
    padding: 10px 18px;
}

.stButton > button:hover {
    background-color: #A7E879;
    color: #081810;
    border: none;
}

/* Input fields */
.stTextInput input,
.stNumberInput input {
    background-color: #142D20;
    color: white;
    border: 1px solid #31563A;
    border-radius: 8px;
}

/* Select boxes */
[data-baseweb="select"] > div {
    background-color: #142D20;
    border-color: #31563A;
}

/* Dataframes */
[data-testid="stDataFrame"] {
    border: 1px solid #284934;
    border-radius: 12px;
}

/* Dividers */
hr {
    border-color: #284934;
}

/* Info and warning boxes */
[data-testid="stAlert"] {
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)

# ==============================
# END CUSTOM THEME
# ==============================


# ==============================
# DASHBOARD HEADER
# ==============================

st.markdown("""
<style>
.dashboard-header {
    background: linear-gradient(110deg, #153B26, #10291C);
    border: 1px solid #315B39;
    border-radius: 18px;
    padding: 28px 32px;
    margin-bottom: 25px;
}

.dashboard-header h1 {
    color: #EAF7E8 !important;
    font-size: 30px !important;
    margin-bottom: 8px;
}

.dashboard-header p {
    color: #A9C6AD;
    font-size: 14px;
    margin-bottom: 0;
}

.dashboard-tag {
    color: #A6E36B;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 2px;
}
</style>

<div class="dashboard-header">
    <div class="dashboard-tag">PRECISION AGRICULTURE INTELLIGENCE</div>
    <h1>🌱 AgroVision AI</h1>
    <p>Smart climate monitoring, satellite insights and predictive crop analytics.</p>
</div>
""", unsafe_allow_html=True)


with st.sidebar:
    # AgroVision branding
    st.markdown("""
    <div style="
        padding: 15px 5px 22px 5px;
        border-bottom: 1px solid #294534;
        margin-bottom: 22px;
    ">
        <div style="font-size: 27px; font-weight: 800; color: #A6E36B;">
            🌱 AgroVision AI
        </div>
        <div style="
            font-size: 10px;
            letter-spacing: 2px;
            color: #9BB9A0;
            margin-top: 5px;
        ">
            AGRICULTURE INTELLIGENCE
        </div>
    </div>

    <div style="
        color: #A6E36B;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1.5px;
        margin-bottom: 10px;
    ">
        FARM CONFIGURATION
    </div>
    """, unsafe_allow_html=True)
    
    
    # SIDEBAR NAVIGATION

    st.markdown("### WORKSPACE")

    st.markdown(
        "[🏠 Overview](#farm-overview)\n\n"
        "[🌦️ Weather Analytics](#weather-analytics)\n\n"
        "[🛰️ Satellite Monitoring](#satellite-monitoring)\n\n"
        "[⚠️ Climate Risk](#climate-risk)\n\n"
        "[🌾 Yield Prediction](#yield-prediction)"
    )

    st.divider()

    st.header("Farm location")
    lat = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=12.9716, format="%.5f")
    lon = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=77.5946, format="%.5f")
    forecast_days = st.slider("Forecast window (days)", 3, 16, 7)
    crop_name = st.selectbox("Crop", ["Rice", "Wheat", "Maize", "Millet"])
    crop_codes = {"Rice": 0, "Wheat": 1, "Maize": 2, "Millet": 3}
    st.divider()
    st.subheader("🗺️ Farm Boundary")

    boundary_file = st.file_uploader(
        "Upload farm boundary (GeoJSON)",
        type=["geojson", "json"]
    )

    zone_count = st.slider(
        "Number of micro-zones",
        min_value=2,
        max_value=8,
        value=4
    )
    use_satellite = st.checkbox("Attempt Sentinel-2 NDVI lookup", value=False)
    run = st.button("Run climate analysis", type="primary", use_container_width=True)

st.info("Enter a farm's coordinates. Example coordinates are prefilled for Bengaluru; replace them with your field location.")
# ==============================
# FARM BOUNDARY & MICRO-ZONES
# ==============================

farm_boundary = None
sensor_summary = None

if boundary_file is not None:
    try:
        farm_boundary = load_farm_geojson(
            boundary_file.getvalue().decode("utf-8"),
            number_of_zones=zone_count
        )

        st.subheader("🗺️ Farm Boundary & Micro-Location Zones")

        st.metric(
            "Total Farm Area",
            f"{farm_boundary['area_ha']:.3f} hectares"
        )

        zone_features = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {
                        "zone_id": zone["zone_id"],
                        "area_ha": zone["area_ha"]
                    },
                    "geometry": zone["geometry"]
                }
                for zone in farm_boundary["zones"]
            ]
        }

        farm_feature = {
            "type": "Feature",
            "properties": {"name": "Farm Boundary"},
            "geometry": farm_boundary["geometry"]
        }

        zone_layer = pdk.Layer(
            "GeoJsonLayer",
            data=zone_features,
            pickable=True,
            filled=True,
            stroked=True,
            get_fill_color=[80, 180, 100, 100],
            get_line_color=[166, 227, 107, 255],
            line_width_min_pixels=2
        )

        boundary_layer = pdk.Layer(
            "GeoJsonLayer",
            data=farm_feature,
            pickable=False,
            filled=False,
            stroked=True,
            get_line_color=[255, 220, 100, 255],
            line_width_min_pixels=4
        )

        view_state = pdk.ViewState(
            latitude=farm_boundary["centroid"]["latitude"],
            longitude=farm_boundary["centroid"]["longitude"],
            zoom=15,
            pitch=0
        )

        deck = pdk.Deck(
            layers=[zone_layer, boundary_layer],
            initial_view_state=view_state,
            tooltip={
                "html": "<b>{zone_id}</b><br/>Area: {area_ha} hectares"
            }
        )

        st.pydeck_chart(deck, use_container_width=True)

        zone_table = pd.DataFrame([
            {
                "Zone": zone["zone_id"],
                "Area (hectares)": zone["area_ha"],
                "Latitude": zone["centroid"]["latitude"],
                "Longitude": zone["centroid"]["longitude"]
            }
            for zone in farm_boundary["zones"]
        ])

        st.markdown("### Zone Details")
        st.dataframe(zone_table, use_container_width=True, hide_index=True)
                # ==============================
        # SOIL SENSOR DATA
        # ==============================

        st.divider()
        st.subheader("🌡️ Soil Sensor Intelligence")

        st.caption(
            "Upload readings from sensors installed in your farm. "
            "Each reading must be associated with a valid farm zone."
        )

        sensor_file = st.file_uploader(
            "Upload soil sensor CSV",
            type=["csv"],
            key="farm_sensor_upload"
        )

        if sensor_file is not None:
            try:
                sensor_data = load_sensor_data(
                    sensor_file.getvalue(),
                    [zone["zone_id"] for zone in farm_boundary["zones"]]
                )

                sensor_summary = summarize_sensor_data(sensor_data)

                st.success("Sensor data successfully validated!")

                st.markdown("### Zone-wise Sensor Readings")

                st.dataframe(
                    sensor_summary,
                    use_container_width=True,
                    hide_index=True
                )

                with st.expander("View all uploaded sensor readings"):
                    st.dataframe(
                        sensor_data,
                        use_container_width=True,
                        hide_index=True
                    )

            except Exception as e:
                st.error(f"Sensor data error: {e}")

    except Exception as e:
        st.error(f"Farm boundary processing failed: {e}")

if run:
    try:
        with st.spinner("Fetching forecast and analysing the location..."):
            weather = get_weather(lat, lon, forecast_days)
        st.session_state["weather"] = weather
        st.session_state["location"] = (lat, lon)
        st.session_state["crop"] = crop_name
        
        st.session_state["satellite"] = None
        st.session_state["zone_satellite"] = None

        if use_satellite:
            with st.spinner("Analysing Sentinel-2 imagery across your farm..."):
                st.session_state["satellite"] = get_sentinel_ndvi(lat, lon)

                if farm_boundary is not None:
                    st.session_state["zone_satellite"] = get_farm_zone_ndvi(
                        farm_boundary,
                        days_back=90,
                        max_datatakes=12,
                        target_coverage_pct=95
                    )


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

    
    
    # FARM OVERVIEW CARDS

    st.subheader("Farm Overview", anchor="farm-overview")

    avg_temp = float(weather["temperature_2m_mean"].mean())

    total_rainfall = float(
        weather["precipitation_sum"].fillna(0).sum()
    )

    soil_display = (
        f"{soil_value * 100:.1f}%"
        if soil_value is not None
        else "Unavailable"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "🌾 Selected Crop",
            st.session_state.get("crop", "—")
        )

    with col2:
        st.metric(
            "🌡️ Average Temperature",
            f"{avg_temp:.1f} °C"
        )

    with col3:
        st.metric(
            "🌧️ Forecast Rainfall",
            f"{total_rainfall:.1f} mm"
        )

    with col4:
        st.metric(
            "💧 Soil Moisture",
            soil_display
        )

    st.caption(
        f"📍 Farm coordinates: {lat0:.4f}, {lon0:.4f}"
        f"  |  Forecast period: {len(weather)} days"
    )

    # WEATHER ANALYTICS

    st.subheader("Weather Analytics", anchor="weather-analytics")

    chart_df = weather[
        ["time", "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean"]
    ].melt(
        id_vars="time",
        var_name="temperature",
        value_name="°C"
    )
    chart_df["temperature"] = chart_df["temperature"].map({
        "temperature_2m_max": "Maximum",
        "temperature_2m_min": "Minimum",
        "temperature_2m_mean": "Mean"
    })

    temp_fig = px.line(
        chart_df,
        x="time",
        y="°C",
        color="temperature",
        title="Temperature Forecast",
        labels={
            "time": "Date",
            "°C": "Temperature (°C)",
            "temperature": "Temperature Series"
        },
        color_discrete_map={
            "Maximum": "#A6E36B",
            "Minimum": "#55B8A6",
            "Mean": "#F4C95D"
        }
    )

    temp_fig.update_traces(
        line=dict(width=3),
        mode="lines+markers"
    )

    temp_fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#122D1F",
        plot_bgcolor="#122D1F",
        font=dict(color="#E8F3E9"),
        height=350,
        margin=dict(l=20, r=20, t=60, b=20),
        hovermode="x unified"
    )
    temp_fig.update_xaxes(
        title="Date",
        showgrid=False
    )

    temp_fig.update_yaxes(
        title="Temperature (°C)",
        gridcolor="#294534",
        zeroline=False
    )

    rain_fig = px.bar(
        weather,
        x="time",
        y="precipitation_sum",
        title="Rainfall Forecast",
        color_discrete_sequence=["#8CCF55"],
        labels={
            "time": "Date",
            "precipitation_sum": "Rainfall (mm)"
        },
    )

    rain_fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#122D1F",
        plot_bgcolor="#122D1F",
        font=dict(color="#E8F3E9"),
        height=350,
        margin=dict(l=20, r=20, t=60, b=20),
        showlegend=False
    )
    rain_fig.update_xaxes(
        title="Date",
        showgrid=False
    )

    rain_fig.update_yaxes(
        title="Rainfall (mm)",
        gridcolor="#294534",
        zeroline=False
    )

    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(temp_fig, use_container_width=True, theme=None)

    with col2:
        st.plotly_chart(rain_fig, use_container_width=True, theme=None)

    with st.expander("📊 View detailed weather data"):
        st.dataframe(weather, use_container_width=True, hide_index=True)

    
    # SATELLITE VEGETATION INTELLIGENCE

    st.subheader("🛰️ Satellite & Vegetation Intelligence", anchor="satellite-monitoring")

    with st.container(border=True):

        if sat:

            if sat.get("ndvi") is not None:

                ndvi = float(sat["ndvi"])

                col1, col2 = st.columns([1, 2])

                with col1:
                    st.metric(
                        "Sentinel-2 NDVI",
                        f"{ndvi:.3f}"
                    )

                with col2:
                    st.markdown("### Vegetation Index")
                    st.progress(
                        max(0.0, min(1.0, (ndvi + 1) / 2))
                    )
                    st.caption("NDVI scale: -1 to +1")

                st.divider()


                # SATELLITE SCENE INFORMATION

                st.markdown("### 🛰️ Satellite Scene Details")

                meta1, meta2, meta3 = st.columns(3)

                meta1.metric(
                    "📅 Scene Date",
                    sat.get("scene_date") or "Unavailable"
                )

                cloud = sat.get("cloud_cover")

                meta2.metric(
                    "☁️ Cloud Coverage",
                    f"{float(cloud):.1f}%" if cloud is not None else "Unavailable"
                )

                valid_pixels = sat.get("valid_pixels")

                meta3.metric(
                    "🖼️ Valid Pixels",
                    str(valid_pixels) if valid_pixels is not None else "—"
                )

                st.caption(
                    "Satellite scene metadata and valid pixels used "
                    "for the vegetation index calculation."
                )


            else:
                st.warning(
                    sat.get("message", "No NDVI data available.")
                )

        else:
            st.info(
                "Satellite lookup was not requested. "
                "Enable Sentinel-2 lookup in Farm Configuration."
            )

    
    # ==============================
    # WHOLE-FARM SATELLITE INTELLIGENCE
    # ==============================

    zone_satellite = st.session_state.get("zone_satellite")

    if zone_satellite is not None:

        st.divider()
        st.markdown("### 🛰️ Whole-Farm Satellite Intelligence")

        st.info(zone_satellite.get("message", "Satellite analysis completed."))

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Satellite Scene",
                zone_satellite.get("scene_date") or "Unavailable"
            )

        with col2:
            st.metric(
                "Spatial Coverage",
                f"{zone_satellite.get('farm_spatial_coverage_pct', 0):.1f}%"
            )

        with col3:
            st.metric(
                "Usable Coverage",
                f"{zone_satellite.get('farm_usable_coverage_pct', 0):.1f}%"
            )

        with col4:
            st.metric(
                "Tiles Used",
                len(zone_satellite.get("tiles_used", []))
            )

        st.caption(
            "Radiometric correction: "
            + (
                "Applied"
                if zone_satellite.get("reflectance_correction_applied")
                else "Not confirmed"
            )
            + " | Cloud and shadow masking: "
            + (
                "Applied"
                if zone_satellite.get("cloud_mask_applied")
                else "Not confirmed"
            )
        )

        if zone_satellite.get("zones"):

            st.markdown("#### Zone-wise Vegetation Analysis")

            satellite_table = pd.DataFrame([
                {
                    "Zone": zone["zone_id"],
                    "Area (ha)": zone["area_ha"],
                    "Median NDVI": zone["ndvi"],
                    "Mean NDVI": zone["mean_ndvi"],
                    "Usable Coverage (%)": zone["usable_coverage_pct"],
                    "Valid Pixels": zone["valid_pixels"],
                    "Red Reflectance": zone["median_red_reflectance"],
                    "NIR Reflectance": zone["median_nir_reflectance"],
                }
                for zone in zone_satellite["zones"]
            ])

            st.dataframe(
                satellite_table,
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                "NDVI indicates vegetation greenness, but does not "
                "diagnose drought or crop damage on its own. "
                "Always consider acquisition date and field conditions."
            )

        else:
            st.warning(
                "No usable zone-wise NDVI measurements are available."
            )

    # CLIMATE RISK ASSESSMENT

    st.subheader("🌦️ Climate Risk Assessment", anchor="climate-risk")

    risks, recommendations, metrics = assess_risks(weather, soil_value)

    # Summary metrics
    m1, m2, m3 = st.columns(3)

    m1.metric(
        "Forecast Rainfall",
        f"{metrics['forecast_rain_mm']} mm"
    )

    m2.metric(
        "Average Temperature",
        f"{metrics['mean_daily_max_temp_c']} °C"
    )

    m3.metric(
        "Rain Probability",
        f"{metrics['mean_rain_probability_pct']}%"
    )

    st.markdown("### Risk Analysis")

    # RISK SUMMARY

    high_count = sum(1 for _, level, _ in risks if level == "High")
    moderate_count = sum(1 for _, level, _ in risks if level == "Moderate")
    low_count = sum(1 for _, level, _ in risks if level == "Low")

    risk1, risk2, risk3 = st.columns(3)

    with risk1:
        st.metric("🔴 High Risks", high_count)

    with risk2:
        st.metric("🟠 Moderate Risks", moderate_count)

    with risk3:
        st.metric("🟢 Low Risks", low_count)

    st.divider()


    # Display risk cards
    risk_columns = st.columns(2)

    for index, (risk_name, level, explanation) in enumerate(risks):

        with risk_columns[index % 2]:

            with st.container(border=True):

                st.markdown(f"**{risk_name}**")

                if level == "High":
                    st.error(f"🔴 High Risk\n\n{explanation}")

                elif level == "Moderate":
                    st.warning(f"🟠 Moderate Risk\n\n{explanation}")

                else:
                    st.success(f"🟢 Low Risk\n\n{explanation}")

    # Recommendations
    st.markdown("### 💡 Smart Recommendations")

    # DISPLAY SMART RECOMMENDATIONS

    with st.container(border=True):

        if recommendations:

            for item in recommendations:
                st.markdown(f"🌱 {item}")

        else:
            st.info("No recommendations available right now.")

  
    # YIELD PREDICTION

    # ==============================
    # ZONE-WISE MICRO-LOCATION INTELLIGENCE
    # ==============================

    if farm_boundary is not None:

        zone_risks = assess_zone_risks(
            farm_boundary["zones"],
            sensor_summary,
            weather
        )

        zone_plans = generate_recommendations(
            zone_risks,
            satellite_data=st.session_state.get("zone_satellite")
        )

        st.divider()

        st.subheader("🧠 Zone-wise Micro-Location Intelligence")

        st.caption(
            "Each farm zone receives its own soil-moisture assessment "
            "and targeted recommendations. Heat and rainfall risks "
            "currently use the farm-level weather forecast."
        )

        for zone_result, zone_plan in zip(zone_risks, zone_plans):

            with st.container(border=True):

                st.markdown(
                    f"### 📍 {zone_result['zone_id']} "
                    f"— {zone_result['area_ha']:.2f} hectares"
                )

                # ZONE RISK ASSESSMENT

                st.markdown("**Climate Risk Assessment**")

                risk_columns = st.columns(
                    len(zone_result["risks"])
                )

                for index, risk in enumerate(zone_result["risks"]):

                    with risk_columns[index]:

                        level = risk["level"]

                        if level == "High":
                            st.error(
                                f"🔴 **{risk['risk']}**\n\n"
                                f"High Risk\n\n{risk['reason']}"
                            )

                        elif level == "Moderate":
                            st.warning(
                                f"🟠 **{risk['risk']}**\n\n"
                                f"Moderate Risk\n\n{risk['reason']}"
                            )

                        elif level == "Low":
                            st.success(
                                f"🟢 **{risk['risk']}**\n\n"
                                f"Low Risk\n\n{risk['reason']}"
                            )

                        else:
                            st.info(
                                f"⚪ **{risk['risk']}**\n\n"
                                f"Unknown\n\n{risk['reason']}"
                            )

                # DATA COVERAGE

                st.markdown("**Data Coverage**")

                st.caption(
                    " · ".join(zone_result["data_status"])
                )

                if zone_result["evidence"]:
                    st.write(
                        "**Measured evidence:** "
                        + " | ".join(zone_result["evidence"])
                    )

                # SMART RECOMMENDATIONS

                st.markdown("**💡 Recommended Actions**")

                for item in zone_plan["recommendations"]:

                    st.markdown(
                        f"**{item['priority']} Priority · "
                        f"{item['timing']}**"
                    )

                    st.write(item["action"])

                    st.caption(
                        f"Why: {item['reason']}"
                    )

                    st.caption(
                        f"Expected benefit: {item['expected_benefit']}"
                    )

                    st.caption(
                        f"Recommendation confidence: {item['confidence']}"
                    )

                    st.divider()


    st.subheader("🌾 Yield Prediction", anchor="yield-prediction")

    st.warning(
        "The default training data is synthetic. "
        "This prediction is for software demonstration only, "
        "not for actual farming decisions."
    )

    ndvi_value = (
        sat.get("ndvi")
        if sat and sat.get("ndvi") is not None
        else 0.55
    )

    with st.container(border=True):

        st.markdown("#### Prediction Inputs")

        col1, col2 = st.columns(2)

        with col1:

            seasonal_rain = st.number_input(
                "Expected seasonal rainfall (mm)",
                min_value=0.0,
                max_value=3000.0,
                value=650.0,
                step=25.0,
                key="yield_seasonal_rain"
            )

            mean_temp = st.number_input(
                "Expected seasonal mean temperature (°C)",
                min_value=0.0,
                max_value=50.0,
                value=28.0,
                step=0.5,
                key="yield_mean_temp"
            )

        with col2:

            moisture = st.number_input(
                "Soil moisture (fraction, e.g. 0.25)",
                min_value=0.0,
                max_value=1.0,
                value=float(
                    soil_value if soil_value is not None else 0.25
                ),
                step=0.01,
                key="yield_soil_moisture"
            )

            ndvi_input = st.number_input(
                "NDVI (0 to 1)",
                min_value=0.0,
                max_value=1.0,
                value=float(ndvi_value),
                step=0.01,
                key="yield_ndvi"
            )

        st.caption(
            "NDVI defaults to 0.55 when satellite data is unavailable."
        )

        if st.button(
            "🌱 Estimate Demo Yield",
            type="primary",
            use_container_width=True,
            key="estimate_demo_yield"
        ):

            try:

                features = {
                    "seasonal_rainfall_mm": seasonal_rain,
                    "mean_temp_c": mean_temp,
                    "soil_moisture": moisture,
                    "ndvi": ndvi_input,
                    "crop_encoded": crop_codes[
                        st.session_state.get("crop", "Rice")
                    ]
                }

                prediction = predict_yield(features)

                st.success("Demonstration prediction generated")

                st.metric(
                    "Demonstration Yield Estimate",
                    f"{prediction:.2f} tonnes/hectare"
                )

                st.caption(
                    "This is a software demonstration based on "
                    "synthetic training data, not an agronomic forecast."
                )

            except Exception as e:

                st.error(f"Yield model error: {e}")


    with st.expander("⚙️ Model Training — Historical Yield Data", expanded=False):
        st.markdown("### Train Your Agriculture Model")
        st.caption(
            "Upload historical crop-yield data to train and evaluate "
            "your machine learning model."
        )
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
