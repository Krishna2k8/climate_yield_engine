
import pandas as pd
from io import BytesIO, StringIO


REQUIRED_COLUMNS = [
    "zone_id",
    "timestamp",
    "soil_moisture_pct",
]

OPTIONAL_COLUMNS = [
    "soil_temperature_c",
    "relative_humidity_pct",
]


def load_sensor_data(file_data, valid_zone_ids):
    """
    Load and validate uploaded farm sensor readings.

    Required CSV columns:
    zone_id, timestamp, soil_moisture_pct

    Optional columns:
    soil_temperature_c, relative_humidity_pct
    """

    try:
        if isinstance(file_data, bytes):
            df = pd.read_csv(BytesIO(file_data))
        else:
            df = pd.read_csv(file_data)

    except Exception as exc:
        raise ValueError(f"Could not read sensor CSV: {exc}")

    missing = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required CSV columns: {', '.join(missing)}"
        )

    if df.empty:
        raise ValueError("The uploaded sensor file is empty.")

    df = df.copy()

    df["zone_id"] = df["zone_id"].astype(str).str.strip()

    unknown_zones = sorted(
        set(df["zone_id"]) - set(valid_zone_ids)
    )

    if unknown_zones:
        raise ValueError(
            f"Unknown zone IDs: {', '.join(unknown_zones)}"
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
        utc=True
    )

    if df["timestamp"].isna().any():
        raise ValueError(
            "Some timestamps are invalid. Use ISO format, "
            "for example 2026-10-31T10:30:00Z."
        )

    numeric_columns = [
        "soil_moisture_pct",
        *OPTIONAL_COLUMNS
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    if df["soil_moisture_pct"].isna().any():
        raise ValueError("Soil moisture contains missing or invalid values.")

    if not df["soil_moisture_pct"].between(0, 100).all():
        raise ValueError("Soil moisture must be between 0 and 100%.")

    if "soil_temperature_c" in df.columns:
        if not df["soil_temperature_c"].dropna().between(-10, 70).all():
            raise ValueError("Soil temperature must be between -10 and 70°C.")

    if "relative_humidity_pct" in df.columns:
        if not df["relative_humidity_pct"].dropna().between(0, 100).all():
            raise ValueError("Relative humidity must be between 0 and 100%.")

    return df.sort_values("timestamp").reset_index(drop=True)


def summarize_sensor_data(df):
    """Create a per-zone summary from uploaded sensor readings."""

    summary = df.groupby("zone_id").agg(
        soil_moisture_pct=("soil_moisture_pct", "mean"),
        latest_reading=("timestamp", "max"),
        reading_count=("timestamp", "count"),
    ).reset_index()

    if "soil_temperature_c" in df.columns:
        temperature = df.groupby("zone_id")[
            "soil_temperature_c"
        ].mean().rename("soil_temperature_c")

        summary = summary.merge(
            temperature,
            on="zone_id",
            how="left"
        )

    if "relative_humidity_pct" in df.columns:
        humidity = df.groupby("zone_id")[
            "relative_humidity_pct"
        ].mean().rename("relative_humidity_pct")

        summary = summary.merge(
            humidity,
            on="zone_id",
            how="left"
        )

    summary["soil_moisture_pct"] = (
        summary["soil_moisture_pct"].round(1)
    )

    return summary
