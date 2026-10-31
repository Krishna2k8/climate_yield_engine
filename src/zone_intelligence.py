import pandas as pd


def assess_zone_risks(zones, sensor_summary, weather):
    """
    Assess climate risks separately for each farm zone.

    Soil moisture is zone-specific when sensor data is available.
    Weather risks are farm-level because the current forecast
    represents one farm coordinate, not separate zone forecasts.
    """

    results = []

    # Farm-level weather indicators
    total_rain = (
        float(weather["precipitation_sum"].fillna(0).sum())
        if "precipitation_sum" in weather.columns
        else None
    )

    mean_max_temp = (
        float(weather["temperature_2m_max"].mean())
        if "temperature_2m_max" in weather.columns
        else None
    )

    # Prepare sensor summaries
    sensor_lookup = {}

    if sensor_summary is not None and not sensor_summary.empty:
        for _, row in sensor_summary.iterrows():
            sensor_lookup[row["zone_id"]] = row

    now = pd.Timestamp.now(tz="UTC")

    for zone in zones:
        zone_id = zone["zone_id"]
        sensor = sensor_lookup.get(zone_id)

        risks = []
        evidence = []
        data_status = []

        # --------------------------------
        # ZONE-SPECIFIC SOIL MOISTURE
        # --------------------------------

        if sensor is None:
            risks.append({
                "risk": "Soil moisture stress",
                "level": "Unknown",
                "reason": "No sensor readings are available for this zone."
            })
            data_status.append("No sensor data")

        else:
            moisture = float(sensor["soil_moisture_pct"])
            latest = pd.to_datetime(sensor["latest_reading"], utc=True)
            age_hours = (now - latest).total_seconds() / 3600

            if age_hours > 48:
                risks.append({
                    "risk": "Soil moisture stress",
                    "level": "Unknown",
                    "reason": "The latest sensor reading is older than 48 hours."
                })
                data_status.append("Sensor data is stale")

            else:
                evidence.append(
                    f"Soil moisture: {moisture:.1f}%"
                )
                data_status.append("Recent sensor data available")

                if moisture < 15:
                    level = "High"
                    reason = "Measured soil moisture is below the prototype threshold of 15%."

                elif moisture < 25:
                    level = "Moderate"
                    reason = "Measured soil moisture is below the prototype threshold of 25%."

                else:
                    level = "Low"
                    reason = "Measured soil moisture is not flagged by the prototype thresholds."

                risks.append({
                    "risk": "Soil moisture stress",
                    "level": level,
                    "reason": reason
                })

        # --------------------------------
        # FARM-LEVEL HEAT RISK
        # --------------------------------

        if mean_max_temp is not None and pd.notna(mean_max_temp):
            if mean_max_temp >= 38:
                heat_level = "High"
            elif mean_max_temp >= 34:
                heat_level = "Moderate"
            else:
                heat_level = "Low"

            risks.append({
                "risk": "Heat stress",
                "level": heat_level,
                "reason": (
                    f"Farm-level forecast average daily maximum "
                    f"temperature is {mean_max_temp:.1f}°C."
                )
            })
        else:
            risks.append({
                "risk": "Heat stress",
                "level": "Unknown",
                "reason": "Temperature forecast data is unavailable."
            })

        # --------------------------------
        # FARM-LEVEL RAINFALL RISK
        # --------------------------------

        if total_rain is not None:
            if total_rain >= 100:
                rain_level = "High"
            elif total_rain >= 40:
                rain_level = "Moderate"
            else:
                rain_level = "Low"

            risks.append({
                "risk": "Excess rainfall",
                "level": rain_level,
                "reason": (
                    f"Farm-level forecast rainfall over the "
                    f"forecast window is {total_rain:.1f} mm."
                )
            })
        else:
            risks.append({
                "risk": "Excess rainfall",
                "level": "Unknown",
                "reason": "Rainfall forecast data is unavailable."
            })

        results.append({
            "zone_id": zone_id,
            "area_ha": zone["area_ha"],
            "risks": risks,
            "evidence": evidence,
            "data_status": data_status
        })

    return results