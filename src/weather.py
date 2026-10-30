import requests
import pandas as pd

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

def get_weather(lat: float, lon: float, days: int = 7) -> pd.DataFrame:
    """Fetch forecast + soil variables from Open-Meteo."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "forecast_days": max(1, min(days, 16)),
        "timezone": "auto",
        "daily": [
            "temperature_2m_max", "temperature_2m_min",
            "temperature_2m_mean", "precipitation_sum",
            "precipitation_probability_max", "et0_fao_evapotranspiration"
        ],
        "hourly": [
            "soil_moisture_0_to_7cm", "soil_temperature_0cm",
            "relative_humidity_2m"
        ]
    }
    response = requests.get(FORECAST_URL, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()

    daily = pd.DataFrame(payload.get("daily", {}))
    if daily.empty:
        raise ValueError("The weather service returned no daily forecast data.")

    # Aggregate hourly soil moisture into daily averages where available.
    hourly = pd.DataFrame(payload.get("hourly", {}))
    if not hourly.empty and "time" in hourly:
        hourly["date"] = pd.to_datetime(hourly["time"]).dt.date.astype(str)
        agg_cols = [c for c in ["soil_moisture_0_to_7cm", "soil_temperature_0cm",
                                "relative_humidity_2m"] if c in hourly.columns]
        if agg_cols:
            soil_daily = hourly.groupby("date")[agg_cols].mean(numeric_only=True).reset_index()
            daily["date"] = pd.to_datetime(daily["time"]).dt.date.astype(str)
            daily = daily.merge(soil_daily, on="date", how="left")
    daily["time"] = pd.to_datetime(daily["time"])
    return daily
