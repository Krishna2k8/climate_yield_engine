"""Sentinel-2 NDVI lookup using Microsoft Planetary Computer."""

from datetime import date, timedelta
import numpy as np


def get_sentinel_ndvi(lat: float, lon: float, days_back: int = 45):
    """Return median NDVI from a Sentinel-2 tile that covers the point."""
    try:
        import planetary_computer
        import rasterio
        from rasterio.warp import transform as transform_coords
        from pystac_client import Client
    except Exception as exc:
        return {
            "ndvi": None,
            "scene_date": None,
            "cloud_cover": None,
            "message": f"Satellite dependencies unavailable: {exc}",
        }

    try:
        delta = 0.025
        bbox = [lon - delta, lat - delta, lon + delta, lat + delta]
        end = date.today()
        start = end - timedelta(days=days_back)

        catalog = Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1"
        )
        search = catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=bbox,
            datetime=f"{start.isoformat()}/{end.isoformat()}",
            query={"eo:cloud_cover": {"lt": 35}},
            max_items=50,
        )
        items = list(search.items())

        if not items:
            return {
                "ndvi": None,
                "scene_date": None,
                "cloud_cover": None,
                "message": "No suitable Sentinel-2 image found in the selected date range.",
            }

        # Try the clearest scenes first, but do not assume the first tile
        # actually covers the exact farm coordinate.
        items.sort(key=lambda item: item.properties.get("eo:cloud_cover", 100))
        last_problem = "No tile in the search results covered the selected coordinate."

        for candidate in items:
            item = planetary_computer.sign(candidate)
            scene_date = (
                item.datetime.date().isoformat() if item.datetime else None
            )
            cloud_cover = item.properties.get("eo:cloud_cover")

            try:
                red_href = item.assets["B04"].href
                nir_href = item.assets["B08"].href

                with rasterio.open(red_href) as red_src, rasterio.open(nir_href) as nir_src:
                    # Convert the input WGS84 point separately into each raster CRS.
                    red_x, red_y = transform_coords(
                        "EPSG:4326", red_src.crs, [lon], [lat]
                    )
                    nir_x, nir_y = transform_coords(
                        "EPSG:4326", nir_src.crs, [lon], [lat]
                    )

                    row, col = red_src.index(red_x[0], red_y[0])
                    nir_row, nir_col = nir_src.index(nir_x[0], nir_y[0])

                    red_inside = (
                        0 <= row < red_src.height
                        and 0 <= col < red_src.width
                    )
                    nir_inside = (
                        0 <= nir_row < nir_src.height
                        and 0 <= nir_col < nir_src.width
                    )

                    # This STAC item may intersect the search box without
                    # containing its centre point. Try the next item instead.
                    if not (red_inside and nir_inside):
                        last_problem = "This image tile does not cover the exact farm coordinate."
                        continue

                    row_start = max(0, row - 2)
                    row_end = min(red_src.height, row + 3)
                    col_start = max(0, col - 2)
                    col_end = min(red_src.width, col + 3)

                    red_coordinates = [
                        red_src.xy(r, c)
                        for r in range(row_start, row_end)
                        for c in range(col_start, col_end)
                    ]
                    sample_x = [point[0] for point in red_coordinates]
                    sample_y = [point[1] for point in red_coordinates]

                    # Rasterio sample coordinates must be expressed in the
                    # CRS of the dataset being sampled.
                    if red_src.crs != nir_src.crs:
                        sample_x, sample_y = transform_coords(
                            red_src.crs, nir_src.crs, sample_x, sample_y
                        )
                        nir_coordinates = list(zip(sample_x, sample_y))
                    else:
                        nir_coordinates = red_coordinates

                    red_samples = list(
                        red_src.sample(red_coordinates, masked=True)
                    )
                    nir_samples = list(
                        nir_src.sample(nir_coordinates, masked=True)
                    )

                    ndvi_values = []
                    for red_pixel, nir_pixel in zip(red_samples, nir_samples):
                        red_value = red_pixel[0]
                        nir_value = nir_pixel[0]

                        if np.ma.is_masked(red_value) or np.ma.is_masked(nir_value):
                            continue

                        red_value = float(red_value)
                        nir_value = float(nir_value)
                        denominator = nir_value + red_value

                        if (
                            not np.isfinite(red_value)
                            or not np.isfinite(nir_value)
                            or red_value <= 0
                            or nir_value <= 0
                            or denominator <= 0
                        ):
                            continue

                        value = (nir_value - red_value) / denominator
                        if -1 <= value <= 1:
                            ndvi_values.append(value)

                if ndvi_values:
                    return {
                        "ndvi": float(np.median(ndvi_values)),
                        "scene_date": scene_date,
                        "cloud_cover": cloud_cover,
                        "valid_pixels": len(ndvi_values),
                        "message": "NDVI successfully calculated.",
                    }

                last_problem = "The tile covers the point, but nearby pixels contain no valid reflectance data."

            except Exception as item_exc:
                last_problem = f"Could not read one satellite tile: {item_exc}"
                continue

        return {
            "ndvi": None,
            "scene_date": None,
            "cloud_cover": None,
            "message": f"Satellite scenes were found, but no usable tile was available for this point. {last_problem}",
        }

    except Exception as exc:
        return {
            "ndvi": None,
            "scene_date": None,
            "cloud_cover": None,
            "message": f"Satellite lookup failed: {exc}",
        }

def get_zone_ndvi(farm_boundary, days_back=45, max_cloud_cover=35):
    """
    Calculate Sentinel-2 NDVI separately for each farm zone.

    Uses the same satellite scene for all zones where possible,
    making zone-to-zone comparisons more meaningful.

    Returns zone-level median NDVI, mean NDVI and valid pixel count.
    """

    try:
        import planetary_computer
        import rasterio
        import numpy as np

        from datetime import date, timedelta
        from pystac_client import Client
        from rasterio.mask import mask
        from rasterio.warp import transform_geom, reproject, Resampling

    except Exception as exc:
        return {
            "zones": [],
            "message": f"Satellite dependencies unavailable: {exc}"
        }

    try:
        zones = farm_boundary["zones"]
        farm_geometry = farm_boundary["geometry"]

        end = date.today()
        start = end - timedelta(days=days_back)

        catalog = Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1"
        )

        search = catalog.search(
            collections=["sentinel-2-l2a"],
            intersects=farm_geometry,
            datetime=f"{start.isoformat()}/{end.isoformat()}",
            query={
                "eo:cloud_cover": {"lt": max_cloud_cover}
            },
            max_items=30,
        )

        items = list(search.items())

        if not items:
            return {
                "zones": [],
                "message": (
                    "No suitable Sentinel-2 scenes were found "
                    "for this farm and date range."
                )
            }

        items.sort(
            key=lambda item: item.properties.get(
                "eo:cloud_cover", 100
            )
        )

        best_partial = None
        best_pixel_count = -1

        for candidate in items:

            item = planetary_computer.sign(candidate)

            try:
                with (
                    rasterio.open(item.assets["B04"].href) as red_src,
                    rasterio.open(item.assets["B08"].href) as nir_src
                ):

                    if red_src.crs is None or nir_src.crs is None:
                        continue

                    zone_results = []

                    for zone in zones:

                        # Convert zone geometry into raster coordinates.
                        red_geometry = transform_geom(
                            "EPSG:4326",
                            red_src.crs,
                            zone["geometry"]
                        )

                        nir_geometry = transform_geom(
                            "EPSG:4326",
                            nir_src.crs,
                            zone["geometry"]
                        )

                        # Extract pixels inside each zone.
                        red_clip, red_transform = mask(
                            red_src,
                            [red_geometry],
                            crop=True,
                            filled=False
                        )

                        nir_clip, nir_transform = mask(
                            nir_src,
                            [nir_geometry],
                            crop=True,
                            filled=False
                        )

                        red = red_clip[0].astype(
                            "float32"
                        ).filled(np.nan)

                        nir = nir_clip[0].astype(
                            "float32"
                        ).filled(np.nan)

                        # Align NIR pixels to the red-band grid
                        # if the two raster grids differ.
                        if (
                            red.shape != nir.shape
                            or red_transform != nir_transform
                            or red_src.crs != nir_src.crs
                        ):

                            aligned_nir = np.full(
                                red.shape,
                                np.nan,
                                dtype="float32"
                            )

                            reproject(
                                source=nir,
                                destination=aligned_nir,
                                src_transform=nir_transform,
                                src_crs=nir_src.crs,
                                src_nodata=np.nan,
                                dst_transform=red_transform,
                                dst_crs=red_src.crs,
                                dst_nodata=np.nan,
                                resampling=Resampling.bilinear,
                            )

                            nir = aligned_nir

                        denominator = nir + red

                        valid = (
                            np.isfinite(red)
                            & np.isfinite(nir)
                            & (red > 0)
                            & (nir > 0)
                            & (denominator > 0)
                        )

                        ndvi = np.full(
                            red.shape,
                            np.nan,
                            dtype="float32"
                        )

                        ndvi[valid] = (
                            (nir[valid] - red[valid])
                            / denominator[valid]
                        )

                        valid = (
                            np.isfinite(ndvi)
                            & (ndvi >= -1)
                            & (ndvi <= 1)
                        )

                        values = ndvi[valid]

                        if len(values) > 0:
                            zone_results.append({
                                "zone_id": zone["zone_id"],
                                "area_ha": zone["area_ha"],
                                "ndvi": round(
                                    float(np.median(values)), 4
                                ),
                                "mean_ndvi": round(
                                    float(np.mean(values)), 4
                                ),
                                "valid_pixels": int(len(values)),
                            })

                        else:
                            zone_results.append({
                                "zone_id": zone["zone_id"],
                                "area_ha": zone["area_ha"],
                                "ndvi": None,
                                "mean_ndvi": None,
                                "valid_pixels": 0,
                            })

                valid_count = sum(
                    result["valid_pixels"]
                    for result in zone_results
                )

                if valid_count > best_pixel_count:
                    best_pixel_count = valid_count

                    best_partial = {
                        "zones": zone_results,
                        "scene_date": (
                            item.datetime.date().isoformat()
                            if item.datetime else None
                        ),
                        "cloud_cover": item.properties.get(
                            "eo:cloud_cover"
                        ),
                    }

                # Prefer a scene with valid pixels in every zone.
                if all(
                    result["ndvi"] is not None
                    for result in zone_results
                ):
                    return {
                        **best_partial,
                        "message": (
                            "Zone-wise NDVI successfully calculated."
                        ),
                    }

            except Exception:
                continue

        if best_partial is not None:
            return {
                **best_partial,
                "message": (
                    "Partial zone-wise NDVI calculated. "
                    "Some zones had no valid pixels."
                ),
            }

        return {
            "zones": [],
            "message": (
                "Satellite scenes were found, but no usable "
                "zone-level NDVI data could be calculated."
            ),
        }

    except Exception as exc:
        return {
            "zones": [],
            "message": f"Zone-wise satellite analysis failed: {exc}"
        }
