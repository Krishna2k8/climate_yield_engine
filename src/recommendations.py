from datetime import date
from statistics import median

def prepare_satellite_context(satellite_data):
    """
    Prepare reliable zone-wise NDVI comparisons.

    Reject stale scenes, poor coverage and insufficient
    zone observations before generating recommendations.
    """

    if not satellite_data:
        return {}, None

    scene_date = satellite_data.get("scene_date")

    try:
        image_date = date.fromisoformat(str(scene_date)[:10])
        age_days = (date.today() - image_date).days
    except (ValueError, TypeError):
        return {}, None

    if age_days < 0 or age_days > 30:
        return {}, None

    farm_coverage = satellite_data.get(
        "farm_usable_coverage_pct", 0
    )

    if farm_coverage < 90:
        return {}, None

    lookup = {}

    for zone in satellite_data.get("zones", []):
        ndvi = zone.get("ndvi")

        if ndvi is None:
            continue

        try:
            ndvi = float(ndvi)
            coverage = float(zone.get("usable_coverage_pct", 0))
            pixels = int(zone.get("valid_pixels", 0))
        except (TypeError, ValueError):
            continue

        if (
            -1 <= ndvi <= 1
            and coverage >= 90
            and pixels >= 20
        ):
            lookup[zone["zone_id"]] = ndvi

    # Require at least three valid zones for comparison.
    if len(lookup) < 3:
        return {}, None

    farm_median = median(lookup.values())

    # Avoid interpreting uniformly low vegetation signals
    # as a zone-specific anomaly.
    if farm_median < 0.20:
        return {}, None

    return lookup, farm_median

def generate_recommendations(zone_results, satellite_data=None):
    """
    Generate explainable, zone-specific recommendations
    from the output of the zone risk engine.

    Recommendations are prototype guidance, not a
    substitute for local agronomic advice.
    """
    all_recommendations = []

    satellite_lookup, farm_ndvi_median = (
        prepare_satellite_context(satellite_data)
    )

    for zone in zone_results:
        zone_id = zone["zone_id"]
        recommendations = []

        for risk in zone["risks"]:
            risk_name = risk["risk"]
            level = risk["level"]
            reason = risk["reason"]

            # 1. SOIL MOISTURE STRESS
            if risk_name == "Soil moisture stress":

                if level == "High":
                    recommendations.append({
                        "priority": "High",
                        "action": (
                            "Verify soil moisture at 2–3 points "
                            "in this zone and inspect the irrigation system."
                        ),
                        "location": zone_id,
                        "timing": "As soon as possible",
                        "reason": reason,
                        "expected_benefit": (
                            "Confirm whether the zone is experiencing "
                            "water stress before deciding on irrigation."
                        ),
                        "confidence": "Moderate",
                    })

                elif level == "Moderate":
                    recommendations.append({
                        "priority": "Medium",
                        "action": (
                            "Monitor soil moisture and check the crop "
                            "for early signs of water stress."
                        ),
                        "location": zone_id,
                        "timing": "Within the next 24 hours",
                        "reason": reason,
                        "expected_benefit": (
                            "Identify worsening moisture conditions early."
                        ),
                        "confidence": "Moderate",
                    })

                elif level == "Unknown":
                    recommendations.append({
                        "priority": "High",
                        "action": (
                            "Collect a fresh soil moisture reading "
                            "from this zone."
                        ),
                        "location": zone_id,
                        "timing": "Before making irrigation decisions",
                        "reason": reason,
                        "expected_benefit": (
                            "Avoid making water-management decisions "
                            "without reliable soil data."
                        ),
                        "confidence": "High",
                    })

            # 2. HEAT STRESS
            elif risk_name == "Heat stress":

                if level == "High":
                    recommendations.append({
                        "priority": "High",
                        "action": (
                            "Inspect crops for heat-stress symptoms "
                            "and check whether water is available "
                            "for appropriate crop management."
                        ),
                        "location": zone_id,
                        "timing": "During the next hot period",
                        "reason": reason,
                        "expected_benefit": (
                            "Identify heat damage early and support "
                            "timely management decisions."
                        ),
                        "confidence": "Moderate",
                    })

                elif level == "Moderate":
                    recommendations.append({
                        "priority": "Medium",
                        "action": (
                            "Inspect crops during the hottest part "
                            "of the day and monitor for wilting."
                        ),
                        "location": zone_id,
                        "timing": "Daily during the forecast period",
                        "reason": reason,
                        "expected_benefit": (
                            "Detect developing heat stress before "
                            "visible damage becomes severe."
                        ),
                        "confidence": "Moderate",
                    })

            # 3. EXCESS RAINFALL
            elif risk_name == "Excess rainfall":

                if level == "High":
                    recommendations.append({
                        "priority": "High",
                        "action": (
                            "Inspect field drainage and check for "
                            "standing water after rainfall."
                        ),
                        "location": zone_id,
                        "timing": "Before and after forecast rainfall",
                        "reason": reason,
                        "expected_benefit": (
                            "Identify waterlogging risks early."
                        ),
                        "confidence": "Moderate",
                    })

                elif level == "Moderate":
                    recommendations.append({
                        "priority": "Medium",
                        "action": (
                            "Monitor drainage channels and check "
                            "low-lying parts of the zone."
                        ),
                        "location": zone_id,
                        "timing": "Before the expected rainfall",
                        "reason": reason,
                        "expected_benefit": (
                            "Improve preparedness for possible "
                            "water accumulation."
                        ),
                        "confidence": "Moderate",
                    })

        # 4. SATELLITE-BASED VEGETATION ANOMALY

        if (
            farm_ndvi_median is not None
            and zone_id in satellite_lookup
        ):
            zone_ndvi = satellite_lookup[zone_id]

            ndvi_difference = farm_ndvi_median - zone_ndvi

            # Flag zones substantially below the farm median.
            # This is an anomaly signal, not a crop-damage diagnosis.
            if ndvi_difference >= 0.15:

                scene_date = satellite_data.get(
                    "scene_date", "unknown date"
                )

                recommendations.append({
                    "priority": "Medium",
                    "action": (
                        "Inspect vegetation in this zone. Check for "
                        "uneven crop establishment, canopy gaps, "
                        "bare soil or differences in crop growth stage."
                    ),
                    "location": zone_id,
                    "timing": "During the next field inspection",
                    "reason": (
                        f"Zone NDVI is {zone_ndvi:.3f}, which is "
                        f"{ndvi_difference:.3f} below the farm's "
                        f"median NDVI of {farm_ndvi_median:.3f}. "
                        f"Satellite scene date: {scene_date}. "
                        "This is a relative vegetation anomaly, "
                        "not a confirmed crop-stress diagnosis."
                    ),
                    "expected_benefit": (
                        "Verify whether the vegetation difference "
                        "reflects a genuine crop-management issue "
                        "before choosing an intervention."
                    ),
                    "confidence": "Moderate for anomaly detection; cause unconfirmed",
                })

        # Fallback if no actionable risk was detected
        if not recommendations:
            recommendations.append({
                "priority": "Low",
                "action": (
                    "Continue routine monitoring and collect "
                    "updated field observations."
                ),
                "location": zone_id,
                "timing": "During the next monitoring cycle",
                "reason": "No actionable risk was flagged by the current prototype.",
                "expected_benefit": (
                    "Maintain awareness of changing farm conditions."
                ),
                "confidence": "Limited",
            })

        all_recommendations.append({
            "zone_id": zone_id,
            "recommendations": recommendations,
        })

    return all_recommendations

