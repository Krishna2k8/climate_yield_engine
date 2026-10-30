import pandas as pd
from typing import Optional

def assess_risks(weather: pd.DataFrame, soil_moisture: Optional[float] = None):
    """Transparent rule-based risk indicators; not a calibrated forecast model."""
    total_rain = float(weather["precipitation_sum"].fillna(0).sum()) if "precipitation_sum" in weather else 0
    mean_max_temp = float(weather["temperature_2m_max"].mean()) if "temperature_2m_max" in weather else 0
    mean_et0 = float(weather["et0_fao_evapotranspiration"].mean()) if "et0_fao_evapotranspiration" in weather else 0
    mean_rain_prob = float(weather["precipitation_probability_max"].mean()) if "precipitation_probability_max" in weather else 0

    risks = []
    if total_rain < 10 and mean_max_temp >= 30:
        risks.append(("Drought / moisture stress", "High",
                      "Forecast rainfall is low while temperatures are elevated. Check field moisture and plan irrigation carefully."))
    elif total_rain < 25:
        risks.append(("Dry spell", "Moderate",
                      "Rainfall over the forecast window looks limited. Monitor soil moisture and crop stress."))
    else:
        risks.append(("Dry spell", "Low",
                      "Forecast rainfall is not especially low over this window."))

    if mean_max_temp >= 38:
        heat_level = "High"
    elif mean_max_temp >= 34:
        heat_level = "Moderate"
    else:
        heat_level = "Low"
    risks.append(("Heat stress", heat_level,
                  "High daytime temperatures can stress crops. Consider locally appropriate irrigation timing and shade measures."))

    if mean_rain_prob >= 70 and total_rain >= 40:
        rain_level = "High"
    elif mean_rain_prob >= 50 or total_rain >= 25:
        rain_level = "Moderate"
    else:
        rain_level = "Low"
    risks.append(("Heavy rainfall", rain_level,
                  "Check field drainage and local alerts if heavy rainfall is forecast."))

    if soil_moisture is not None:
        if soil_moisture < 0.15:
            risks.append(("Surface soil dryness", "High",
                          "The reported top-layer soil moisture is low. Verify with an on-site sensor before irrigation decisions."))
        elif soil_moisture < 0.25:
            risks.append(("Surface soil dryness", "Moderate",
                          "Top-layer soil moisture is worth monitoring."))
        else:
            risks.append(("Surface soil dryness", "Low",
                          "Top-layer soil moisture is not flagged as low by this simple threshold."))

    recommendations = []
    if any(level == "High" for _, level, _ in risks):
        recommendations += [
            "Verify the forecast against local observations before taking costly action.",
            "Inspect crop condition and soil moisture in multiple parts of the field."
        ]
    if mean_max_temp >= 34:
        recommendations.append("Where agronomically suitable, irrigate during cooler hours to reduce evaporation.")
    if total_rain >= 25:
        recommendations.append("Check drainage channels and avoid field operations on waterlogged soil.")
    if not recommendations:
        recommendations.append("Continue routine monitoring; no high risk was flagged by these simple rules.")

    return risks, recommendations, {
        "forecast_rain_mm": round(total_rain, 1),
        "mean_daily_max_temp_c": round(mean_max_temp, 1),
        "mean_daily_et0_mm": round(mean_et0, 2),
        "mean_rain_probability_pct": round(mean_rain_prob, 1)
    }
