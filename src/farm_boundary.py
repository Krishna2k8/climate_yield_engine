import json

from pyproj import CRS, Geod, Transformer
from shapely.geometry import (
    shape,
    mapping,
    box,
    Polygon,
    MultiPolygon,
    GeometryCollection,
)
from shapely.validation import make_valid
from shapely.ops import transform


GEOD = Geod(ellps="WGS84")


def _get_polygonal_geometry(geometry):
    """Keep only polygon components from a repaired geometry."""
    if isinstance(geometry, (Polygon, MultiPolygon)):
        return geometry

    if isinstance(geometry, GeometryCollection):
        polygons = [
            item
            for item in geometry.geoms
            if isinstance(item, (Polygon, MultiPolygon))
        ]

        if not polygons:
            raise ValueError("The boundary does not contain a valid polygon.")

        from shapely.ops import unary_union
        return unary_union(polygons)

    raise ValueError("Please upload a Polygon or MultiPolygon boundary.")


def _calculate_area_ha(geometry):
    """Calculate geodesic area in hectares."""
    area, _ = GEOD.geometry_area_perimeter(geometry)
    return abs(area) / 10_000


def _make_utm_transformers(geometry):
    """Create local UTM transformers for accurate farm-zone operations."""
    centroid = geometry.centroid
    longitude = centroid.x
    latitude = centroid.y

    zone_number = int((longitude + 180) // 6) + 1
    epsg_code = (32600 if latitude >= 0 else 32700) + zone_number

    utm_crs = CRS.from_epsg(epsg_code)

    forward = Transformer.from_crs(
        "EPSG:4326", utm_crs, always_xy=True
    ).transform

    backward = Transformer.from_crs(
        utm_crs, "EPSG:4326", always_xy=True
    ).transform

    return forward, backward


def _split_in_half(geometry):
    """Split a projected polygon into two approximately equal-area parts."""
    min_x, min_y, max_x, max_y = geometry.bounds

    use_vertical_cut = (max_x - min_x) >= (max_y - min_y)

    if use_vertical_cut:
        low, high = min_x, max_x
    else:
        low, high = min_y, max_y

    target_area = geometry.area / 2

    for _ in range(45):
        middle = (low + high) / 2

        if use_vertical_cut:
            cutter = box(
                min_x - 1,
                min_y - 1,
                middle,
                max_y + 1,
            )
        else:
            cutter = box(
                min_x - 1,
                min_y - 1,
                max_x + 1,
                middle,
            )

        first_half = geometry.intersection(cutter)

        if first_half.area < target_area:
            low = middle
        else:
            high = middle

    middle = (low + high) / 2

    if use_vertical_cut:
        first_box = box(min_x - 1, min_y - 1, middle, max_y + 1)
        second_box = box(middle, min_y - 1, max_x + 1, max_y + 1)
    else:
        first_box = box(min_x - 1, min_y - 1, max_x + 1, middle)
        second_box = box(min_x - 1, middle, max_x + 1, max_y + 1)

    first = _get_polygonal_geometry(geometry.intersection(first_box))
    second = _get_polygonal_geometry(geometry.intersection(second_box))

    if first.is_empty or second.is_empty:
        raise ValueError("Could not divide this boundary into zones.")

    return first, second


def _create_zones(projected_geometry, number_of_zones):
    """Repeatedly divide the largest zone until the target count is reached."""
    zones = [projected_geometry]

    while len(zones) < number_of_zones:
        largest_index = max(
            range(len(zones)),
            key=lambda index: zones[index].area,
        )

        largest_zone = zones.pop(largest_index)
        first, second = _split_in_half(largest_zone)

        zones.extend([first, second])

    return zones


def load_farm_geojson(geojson_data, number_of_zones=4):
    """
    Validate a GeoJSON farm boundary and generate micro-location zones.

    Accepts:
    - A raw Polygon or MultiPolygon geometry
    - A GeoJSON Feature containing one polygon
    - A FeatureCollection containing exactly one polygon feature

    Coordinates must use WGS84 longitude and latitude.
    """
    if not 2 <= number_of_zones <= 12:
        raise ValueError("Choose between 2 and 12 zones.")

    if isinstance(geojson_data, str):
        try:
            geojson_data = json.loads(geojson_data)
        except json.JSONDecodeError as exc:
            raise ValueError("The uploaded file is not valid JSON.") from exc

    if not isinstance(geojson_data, dict):
        raise ValueError("Invalid GeoJSON structure.")

    geojson_type = geojson_data.get("type")

    if geojson_type == "FeatureCollection":
        features = geojson_data.get("features", [])

        if len(features) != 1:
            raise ValueError(
                "Please upload a GeoJSON containing exactly one farm boundary."
            )

        feature = features[0]
        geometry_data = feature.get("geometry") if feature else None

    elif geojson_type == "Feature":
        geometry_data = geojson_data.get("geometry")

    else:
        geometry_data = geojson_data

    if not geometry_data:
        raise ValueError("No farm boundary geometry was found.")

    try:
        geometry = shape(geometry_data)
    except Exception as exc:
        raise ValueError("Could not read the farm boundary geometry.") from exc

    if geometry.is_empty:
        raise ValueError("The farm boundary is empty.")

    geometry = _get_polygonal_geometry(make_valid(geometry))

    min_x, min_y, max_x, max_y = geometry.bounds

    if (
        min_x < -180 or max_x > 180
        or min_y < -90 or max_y > 90
    ):
        raise ValueError(
            "Coordinates must be longitude/latitude in WGS84 (EPSG:4326)."
        )

    if geometry.area == 0:
        raise ValueError("The farm boundary has zero area.")

    farm_area_ha = _calculate_area_ha(geometry)

    if farm_area_ha <= 0:
        raise ValueError("Could not calculate the farm's area.")

    forward, backward = _make_utm_transformers(geometry)

    projected_geometry = transform(forward, geometry)

    projected_zones = _create_zones(
        projected_geometry,
        number_of_zones,
    )

    # Sort zones from north to south, then west to east.
    projected_zones.sort(
        key=lambda zone: (
            -zone.centroid.y,
            zone.centroid.x,
        )
    )

    zones = []

    for index, projected_zone in enumerate(projected_zones, start=1):
        geographic_zone = transform(backward, projected_zone)

        zones.append({
            "zone_id": f"Z{index:02d}",
            "area_ha": round(_calculate_area_ha(geographic_zone), 3),
            "centroid": {
                "latitude": round(geographic_zone.centroid.y, 6),
                "longitude": round(geographic_zone.centroid.x, 6),
            },
            "geometry": mapping(geographic_zone),
        })

    centroid = geometry.centroid

    return {
        "area_ha": round(farm_area_ha, 3),
        "centroid": {
            "latitude": round(centroid.y, 6),
            "longitude": round(centroid.x, 6),
        },
        "geometry": mapping(geometry),
        "zones": zones,
        "zone_count": len(zones),
    }
