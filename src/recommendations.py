def generate_recommendations(zone_results):
    """
    Generate explainable, zone-specific recommendations
    from the output of the zone risk engine.

    Recommendations are prototype guidance, not a
    substitute for local agronomic advice.
    """

    all_recommendations = []

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
