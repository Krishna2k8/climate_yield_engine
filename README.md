# Predictive Climate & Micro-Location Yield Engine

Buildathon MVP: weather + soil forecast variables + optional Sentinel-2 NDVI + climate-risk rules + yield-model demonstration.

## 1) Requirements
- Python 3.10 or newer
- Internet connection for weather and satellite requests
- macOS / Windows / Linux

## 2) Install on Mac
Open Terminal and enter:

```bash
cd climate_yield_engine
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `rasterio` installation fails on your Mac, you can still run the app without satellite lookup. Try installing the rest first:
```bash
pip install streamlit pandas numpy requests scikit-learn joblib plotly python-dotenv
```

## 3) Run
```bash
streamlit run app.py
```
A browser tab should open. If it does not, visit `http://localhost:8501`.

## 4) Try the demo
- Enter the latitude and longitude of a farm.
- Click **Run climate analysis**.
- Inspect temperature, precipitation, soil-moisture forecast values and risk flags.
- Optionally enable Sentinel-2 NDVI lookup. It is best-effort and may return no scene.
- The yield estimator uses a synthetic demonstration dataset unless you upload real labelled training data.

## 5) Real yield training data
Upload a CSV with these exact columns:
- `seasonal_rainfall_mm`
- `mean_temp_c`
- `soil_moisture`
- `ndvi`
- `crop_encoded` (0=Rice, 1=Wheat, 2=Maize, 3=Millet)
- `yield_tonnes_per_hectare`

Each row should represent a real crop-season-location observation. Keep units consistent and document the data source. Do not mix crops without encoding them correctly.

The included synthetic data is generated automatically for a demo only. It is not evidence of real-world model accuracy.

## 6) Pipeline architecture
1. Input: farm coordinates and crop
2. Weather API: Open-Meteo daily forecast + hourly soil variables
3. Satellite: optional Sentinel-2 L2A scene search through Microsoft Planetary Computer STAC; calculate NDVI from B04 and B08
4. Risk engine: transparent threshold-based indicators
5. Yield model: Random Forest trained on labelled yield records
6. UI: Streamlit dashboard

## 7) Important limitations / next improvements
- Open-Meteo forecast is short-range, not a seasonal climate forecast.
- Forecast soil moisture is modelled data, not a substitute for a calibrated in-field sensor.
- NDVI is a vegetation indicator, not a direct yield measurement. A single central pixel can be noisy; production work should use field polygons, cloud/shadow masks, and zonal statistics.
- Current risk thresholds are illustrative and should be calibrated for crop, growth stage, soil and region with agronomist input.
- Yield predictions are not decision-grade until trained and validated on real multi-year, multi-location yield data.
- For production, add authenticated Copernicus/ISRO feeds where available, sensor ingestion (MQTT/IoT), a database, user accounts, field boundaries and model monitoring.

## 8) Suggested team split
- Data/API: weather + satellite retrieval
- ML: yield dataset, model training and evaluation
- Backend: API and data pipeline orchestration
- Frontend: dashboard and visual storytelling
