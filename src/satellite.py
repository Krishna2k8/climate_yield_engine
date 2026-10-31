
"""Sentinel-2 NDVI lookup using Microsoft Planetary Computer."""

from datetime import date, timedelta
import numpy as np


def get_sentinel_ndvi(lat: float, lon: float, days_back: int = 45):

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
            "message": f"Satellite dependencies unavailable: {exc}"
        }

    try:
        # Search for satellite images around the selected location
        delta = 0.025
        bbox = [lon-delta, lat-delta, lon+delta, lat+delta]

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
            max_items=20
        )

        items = list(search.items())

        if not items:
            return {
                "ndvi": None,
                "scene_date": None,
                "cloud_cover": None,
                "message": "No suitable Sentinel-2 image found. Try another location or date range."
            }

        # Select the scene with the lowest cloud cover
        items.sort(
            key=lambda x: x.properties.get("eo:cloud_cover", 100)
        )

        item = planetary_computer.sign(items[0])

        scene_date = (
            item.datetime.date().isoformat()
            if item.datetime else None
        )

        cloud_cover = item.properties.get("eo:cloud_cover")

        red_href = item.assets["B04"].href
        nir_href = item.assets["B08"].href

        ndvi_values = []

        # Check a 5 x 5 pixel area around the selected coordinate
        with rasterio.open(red_href) as red_src, \
             rasterio.open(nir_href) as nir_src:

            x, y = transform_coords(
                "EPSG:4326",
                red_src.crs,
                [lon],
                [lat]
            )

            row, col = red_src.index(x[0], y[0])

            if not (
                0 <= row < red_src.height
                and 0 <= col < red_src.width
            ):
                return {
                    "ndvi": None,
                    "scene_date": scene_date,
                    "cloud_cover": cloud_cover,
                    "message": "Selected coordinates are outside the satellite image."
                }

            row_start = max(0, row - 2)
            row_end = min(red_src.height, row + 3)

            col_start = max(0, col - 2)
            col_end = min(red_src.width, col + 3)

            coordinates = []

            for r in range(row_start, row_end):
                for c in range(col_start, col_end):
                    x, y = red_src.xy(r, c)
                    coordinates.append((x, y))

            red_samples = list(
                red_src.sample(coordinates, masked=True)
            )

            nir_samples = list(
                nir_src.sample(coordinates, masked=True)
            )

            for red_pixel, nir_pixel in zip(
                red_samples, nir_samples
            ):

                red_value = red_pixel[0]
                nir_value = nir_pixel[0]

                if (
                    np.ma.is_masked(red_value)
                    or np.ma.is_masked(nir_value)
                ):
                    continue

                red_value = float(red_value)
                nir_value = float(nir_value)

                if (
                    not np.isfinite(red_value)
                    or not np.isfinite(nir_value)
                    or red_value <= 0
                    or nir_value <= 0
                ):
                    continue

                denominator = nir_value + red_value

                if denominator == 0:
                    continue

                value = (
                    (nir_value - red_value) / denominator
                )

                if -1 <= value <= 1:
                    ndvi_values.append(value)

        if not ndvi_values:
            return {
                "ndvi": None,
                "scene_date": scene_date,
                "cloud_cover": cloud_cover,
                "message": (
                    "No valid satellite pixels found near these "
                    "coordinates. Try another field location."
                )
            }

        # Median reduces the effect of an unusual individual pixel
        ndvi = float(np.median(ndvi_values))

        return {
            "ndvi": ndvi,
            "scene_date": scene_date,
            "cloud_cover": cloud_cover,
            "valid_pixels": len(ndvi_values),
            "message": "NDVI successfully calculated."
        }

    except Exception as exc:
        return {
            "ndvi": None,
            "scene_date": None,
            "cloud_cover": None,
            "message": f"Satellite lookup failed: {exc}"
        }

