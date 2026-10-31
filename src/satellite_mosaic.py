
from collections import defaultdict
from datetime import date, timedelta
import math
import requests
import xml.etree.ElementTree as ET

import numpy as np
import rasterio
from pyproj import CRS
from pystac_client import Client
import planetary_computer
from rasterio.features import bounds as geometry_bounds, geometry_mask
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling, transform_geom
from shapely.geometry import shape


BAD_SCL_CLASSES = (0, 1, 2, 3, 7, 8, 9, 10, 11)


def _farm_target_crs(farm_boundary):
    """Choose a local UTM CRS for the farm mosaic grid."""
    centroid = shape(farm_boundary["geometry"]).centroid
    lon, lat = centroid.x, centroid.y

    if not -80 <= lat <= 84:
        raise ValueError(
            "This prototype's 10 m mosaic currently supports farms "
            "between 80°S and 84°N."
        )

    utm_zone = max(1, min(60, int((lon + 180) // 6) + 1))
    epsg = (32600 if lat >= 0 else 32700) + utm_zone
    return CRS.from_epsg(epsg)


def _zone_grid(zone, target_crs, resolution_m):
    """Build a common projected 10 m grid and a mask for one zone."""
    projected_geometry = transform_geom(
        "EPSG:4326",
        target_crs,
        zone["geometry"],
    )

    left, bottom, right, top = geometry_bounds(projected_geometry)
    width = max(1, math.ceil((right - left) / resolution_m))
    height = max(1, math.ceil((top - bottom) / resolution_m))

    # Avoid unexpectedly allocating very large arrays for huge boundaries.
    if width * height > 25_000_000:
        raise ValueError(
            f"{zone['zone_id']} is too large for this prototype's "
            "in-memory 10 m processing grid."
        )

    grid_transform = from_origin(
        left, top, resolution_m, resolution_m
    )
    inside = geometry_mask(
        [projected_geometry],
        out_shape=(height, width),
        transform=grid_transform,
        all_touched=False,
        invert=True,
    )

    return grid_transform, (height, width), inside


def _reproject_band(
    source,
    out_shape,
    dst_transform,
    dst_crs,
    resampling,
    dtype,
    dst_nodata,
):
    """Reproject a source band's pixels onto the common farm grid."""
    destination = np.full(
        out_shape,
        dst_nodata,
        dtype=dtype,
    )

    source_nodata = source.nodata
    if source_nodata is None:
        source_nodata = 0

    reproject(
        source=rasterio.band(source, 1),
        destination=destination,
        src_transform=source.transform,
        src_crs=source.crs,
        src_nodata=source_nodata,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        dst_nodata=dst_nodata,
        resampling=resampling,
        init_dest_nodata=True,
    )

    return destination


def _group_by_datatake(items):
    """
    Group adjacent Sentinel-2 tile items from the same acquisition.

    s2:datatake_id is preferred. A conservative metadata fallback is
    used only when that property is absent.
    """
    groups = defaultdict(list)

    for item in items:
        props = item.properties
        datatake_id = props.get("s2:datatake_id")

        if datatake_id:
            key = str(datatake_id)
        else:
            timestamp = (
                item.datetime.isoformat()
                if item.datetime is not None
                else "unknown-time"
            )
            key = "|".join([
                str(props.get("platform", "unknown-platform")),
                str(props.get("sat:relative_orbit", "unknown-orbit")),
                timestamp,
            ])

        groups[key].append(item)

    return list(groups.items())


def _boa_radiometry(item):
    """
    Read the actual Sentinel-2 BOA radiometric parameters
    from the product-metadata XML.

    B04 and B08 offsets are read independently.
    No assumed offset is used for newer products.
    """

    baseline_value = item.properties.get("s2:processing_baseline")

    if baseline_value is None:
        raise ValueError("Sentinel-2 processing baseline is missing.")

    try:
        parts = str(baseline_value).strip().lstrip("vV").split(".")
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
    except (TypeError, ValueError, IndexError) as exc:
        raise ValueError(
            f"Invalid processing baseline: {baseline_value}"
        ) from exc

    metadata_asset = item.assets.get("product-metadata")

    if metadata_asset is None:
        raise ValueError("Product metadata XML asset is unavailable.")

    signed_item = planetary_computer.sign(item)

    response = requests.get(
        signed_item.assets["product-metadata"].href,
        timeout=30
    )
    response.raise_for_status()

    root = ET.fromstring(response.content)

    def local_name(tag):
        return tag.rsplit("}", 1)[-1]

    quantification = None
    offsets = {}

    band_mapping = {
        "3": "B04",
        "7": "B08",
    }

    for element in root.iter():
        name = local_name(element.tag)

        if name == "BOA_QUANTIFICATION_VALUE":
            quantification = float(element.text)

        elif name == "BOA_ADD_OFFSET":
            band_id = element.attrib.get("band_id")

            if band_id in band_mapping:
                offsets[band_mapping[band_id]] = float(element.text)

    if quantification is None or quantification <= 0:
        raise ValueError(
            "Valid BOA quantification value missing from product metadata."
        )

    # Processing baselines before 04.00 did not use BOA_ADD_OFFSET.
    if (major, minor) < (4, 0):
        offsets = {
            "B04": 0.0,
            "B08": 0.0,
        }

    else:
        for band in ("B04", "B08"):
            if band not in offsets:
                raise ValueError(
                    f"BOA offset for {band} is missing from product metadata."
                )

    return str(baseline_value), offsets, quantification


def _evaluate_datatake(group_key, group_items, farm_zones, target_crs, resolution_m):
    """Mosaic all available tiles in one datatake and calculate zone NDVI."""
    contexts = []

    for zone in farm_zones:
        grid_transform, out_shape, inside = _zone_grid(
            zone, target_crs, resolution_m
        )

        contexts.append({
            "zone": zone,
            "transform": grid_transform,
            "shape": out_shape,
            "inside": inside,
            "red": np.full(out_shape, np.nan, dtype="float32"),
            "nir": np.full(out_shape, np.nan, dtype="float32"),
            "spatial": np.zeros(out_shape, dtype=bool),
        })

    used_tiles = set()
    cloud_values = []
    tile_errors = []
    cloud_mask_tiles = 0
    radiometry_by_tile = {}

    # Prefer clearer tiles where tile footprints overlap.
    ordered_items = sorted(
        group_items,
        key=lambda item: (
            float(item.properties["eo:cloud_cover"])
            if item.properties.get("eo:cloud_cover") is not None
            else 100.0
        ),
    )

    for item in ordered_items:
        if not all(name in item.assets for name in ("B04", "B08", "SCL")):
            tile_errors.append(
                f"{item.id}: missing B04, B08 or SCL asset"
            )
            continue

        try:
            baseline, band_offsets, quantification_value = _boa_radiometry(item)
            signed_item = planetary_computer.sign(item)

            with (
                rasterio.open(signed_item.assets["B04"].href) as red_src,
                rasterio.open(signed_item.assets["B08"].href) as nir_src,
                rasterio.open(signed_item.assets["SCL"].href) as scl_src,
            ):
                if red_src.crs is None or nir_src.crs is None or scl_src.crs is None:
                    raise ValueError("A raster asset has no CRS.")

                cloud_mask_tiles += 1
                tile_contributed = False

                for context in contexts:
                    red_dn = _reproject_band(
                        red_src,
                        context["shape"],
                        context["transform"],
                        target_crs,
                        Resampling.bilinear,
                        "float32",
                        np.nan,
                    )

                    nir_dn = _reproject_band(
                        nir_src,
                        context["shape"],
                        context["transform"],
                        target_crs,
                        Resampling.bilinear,
                        "float32",
                        np.nan,
                    )

                    raw_valid = (
                        context["inside"]
                        & np.isfinite(red_dn)
                        & np.isfinite(nir_dn)
                        & (red_dn > 0)
                        & (nir_dn > 0)
                    )

                    # Convert raw L2A digital numbers to BOA surface
                    # reflectance before calculating NDVI. The additive
                    # offset does NOT cancel in the NDVI ratio.
                    red = np.full(red_dn.shape, np.nan, dtype="float32")
                    nir = np.full(nir_dn.shape, np.nan, dtype="float32")
                    red[raw_valid] = (
                        red_dn[raw_valid] + band_offsets["B04"]
                    ) / quantification_value
                    nir[raw_valid] = (
                        nir_dn[raw_valid] + band_offsets["B08"]
                    ) / quantification_value

                    scl = _reproject_band(
                        scl_src,
                        context["shape"],
                        context["transform"],
                        target_crs,
                        Resampling.nearest,
                        "uint8",
                        0,
                    )

                    # Spatial coverage is based on available source-band
                    # data; usable coverage additionally requires positive
                    # corrected reflectance and a clear SCL class.
                    reflectance_valid = (
                        raw_valid
                        & np.isfinite(red)
                        & np.isfinite(nir)
                        & (red > 0)
                        & (nir > 0)
                    )

                    if raw_valid.any():
                        context["spatial"] |= raw_valid
                        tile_contributed = True

                    clear_pixels = ~np.isin(
                        scl, BAD_SCL_CLASSES
                    )

                    usable = reflectance_valid & clear_pixels

                    # Keep the first clear observation from this datatake.
                    # Tiles are ordered by cloud cover before mosaicking.
                    take = usable & ~np.isfinite(context["red"])

                    if take.any():
                        context["red"][take] = red[take]
                        context["nir"][take] = nir[take]
                        tile_contributed = True

                if tile_contributed:
                    tile_id = item.properties.get(
                        "s2:mgrs_tile", item.id
                    )
                    used_tiles.add(str(tile_id))
                    radiometry_by_tile[str(tile_id)] = {
                        "processing_baseline": baseline,
                        "boa_add_offsets_dn": {
                            "B04": band_offsets["B04"],
                            "B08": band_offsets["B08"],
                        },
                        "quantification_value": quantification_value,
                    }

                    cloud = item.properties.get("eo:cloud_cover")
                    if cloud is not None:
                        cloud_values.append(float(cloud))

        except Exception as exc:
            tile_errors.append(f"{item.id}: {exc}")

    zone_results = []
    total_zone_pixels = 0
    total_spatial_pixels = 0
    total_usable_pixels = 0

    for context in contexts:
        zone = context["zone"]
        inside_count = int(context["inside"].sum())

        red = context["red"]
        nir = context["nir"]
        denominator = red + nir

        valid = (
            context["inside"]
            & np.isfinite(red)
            & np.isfinite(nir)
            & (denominator > 0)
        )

        ndvi = np.full(red.shape, np.nan, dtype="float32")
        ndvi[valid] = (nir[valid] - red[valid]) / denominator[valid]

        valid &= np.isfinite(ndvi) & (ndvi >= -1) & (ndvi <= 1)
        values = ndvi[valid]

        spatial_pixels = int(context["spatial"].sum())
        usable_pixels = int(valid.sum())

        spatial_pct = (
            100 * spatial_pixels / inside_count
            if inside_count else 0.0
        )
        usable_pct = (
            100 * usable_pixels / inside_count
            if inside_count else 0.0
        )

        zone_results.append({
            "zone_id": zone["zone_id"],
            "area_ha": zone["area_ha"],
            "ndvi": round(float(np.median(values)), 4)
                if len(values) else None,
            "mean_ndvi": round(float(np.mean(values)), 4)
                if len(values) else None,
            "median_red_reflectance": round(
                float(np.median(red[valid])), 5
            ) if usable_pixels else None,
            "median_nir_reflectance": round(
                float(np.median(nir[valid])), 5
            ) if usable_pixels else None,
            "valid_pixels": usable_pixels,
            "zone_pixels": inside_count,
            "spatial_coverage_pct": round(spatial_pct, 1),
            "usable_coverage_pct": round(usable_pct, 1),
        })

        total_zone_pixels += inside_count
        total_spatial_pixels += spatial_pixels
        total_usable_pixels += usable_pixels

    farm_spatial_pct = (
        100 * total_spatial_pixels / total_zone_pixels
        if total_zone_pixels else 0.0
    )
    farm_usable_pct = (
        100 * total_usable_pixels / total_zone_pixels
        if total_zone_pixels else 0.0
    )

    dates = [
        item.datetime for item in group_items
        if item.datetime is not None
    ]
    scene_date = (
        max(dates).date().isoformat()
        if dates else None
    )

    return {
        "zones": zone_results,
        "scene_date": scene_date,
        "datatake_id": group_key,
        "cloud_cover": round(
            float(np.mean(cloud_values)), 3
        ) if cloud_values else None,
        "tiles_found": len(group_items),
        "tiles_used": sorted(used_tiles),
        "farm_spatial_coverage_pct": round(farm_spatial_pct, 1),
        "farm_usable_coverage_pct": round(farm_usable_pct, 1),
        "cloud_mask_applied": cloud_mask_tiles > 0,
        "radiometry_by_tile": radiometry_by_tile,
        "reflectance_correction_applied": bool(radiometry_by_tile),
        "tile_errors": tile_errors[:5],
    }


def get_farm_zone_ndvi(
    farm_boundary,
    days_back=45,
    resolution_m=10,
    max_datatakes=12,
    target_coverage_pct=95,
):
    """
    Build a same-acquisition Sentinel-2 mosaic across the complete farm.

    Adjacent tiles from one datatake are reprojected to a common local
    UTM grid. NDVI is calculated per farm zone after SCL cloud masking.

    Coverage percentages are based on pixels inside each zone polygon:
    - spatial_coverage_pct: pixels with valid red/NIR reflectance
    - usable_coverage_pct: pixels remaining after SCL quality masking
    """
    try:
        target_crs = _farm_target_crs(farm_boundary)

        end = date.today()
        start = end - timedelta(days=days_back)

        catalog = Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1"
        )

        search = catalog.search(
            collections=["sentinel-2-l2a"],
            intersects=farm_boundary["geometry"],
            datetime=f"{start.isoformat()}/{end.isoformat()}",
            max_items=500,
        )

        items = list(search.items())

        if not items:
            return {
                "zones": [],
                "message": (
                    "No Sentinel-2 scenes intersected the farm "
                    "during the selected date range."
                ),
            }

        groups = _group_by_datatake(items)

        def group_order(group):
            _, group_items = group
            timestamps = [
                item.datetime for item in group_items
                if item.datetime is not None
            ]
            newest = max(timestamps).isoformat() if timestamps else ""
            clouds = [
                float(item.properties["eo:cloud_cover"])
                for item in group_items
                if item.properties.get("eo:cloud_cover") is not None
            ]
            mean_cloud = float(np.mean(clouds)) if clouds else 100.0
            return newest, -mean_cloud

        groups.sort(key=group_order, reverse=True)

        best_result = None
        best_score = None

        for group_key, group_items in groups[:max_datatakes]:
            result = _evaluate_datatake(
                group_key,
                group_items,
                farm_boundary["zones"],
                target_crs,
                resolution_m,
            )

            score = (
                result["farm_usable_coverage_pct"],
                result["farm_spatial_coverage_pct"],
                result["scene_date"] or "",
            )

            if best_result is None or score > best_score:
                best_result = result
                best_score = score

            # Prefer the newest acquisition that reaches the target coverage.
            if result["farm_usable_coverage_pct"] >= target_coverage_pct:
                result["message"] = (
                    "A same-acquisition Sentinel-2 mosaic was created "
                    "with sufficient usable coverage across the farm."
                )
                return result

        if best_result is None:
            return {
                "zones": [],
                "message": (
                    "Satellite scenes were found, but no usable "
                    "same-acquisition mosaic could be created."
                ),
            }

        best_result["message"] = (
            "A partial same-acquisition mosaic was created. "
            "Some farm pixels may be outside the available tiles, "
            "cloud-masked, or missing. Check each zone's coverage."
        )

        return best_result

    except Exception as exc:
        return {
            "zones": [],
            "message": f"Whole-farm satellite analysis failed: {exc}",
        }
