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
