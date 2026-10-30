"""Optional Sentinel-2 NDVI lookup through Microsoft Planetary Computer STAC.

This is best-effort: satellite scenes may be unavailable for a location/date,
and cloud cover or raster access can prevent a result. A missing NDVI is not zero.
"""
from datetime import date, timedelta
import numpy as np

def get_sentinel_ndvi(lat: float, lon: float, days_back: int = 45):
    try:
        import planetary_computer
        import rasterio
        from pystac_client import Client
        from rasterio.windows import from_bounds
    except Exception as exc:
        return {"ndvi": None, "scene_date": None, "cloud_cover": None,
                "message": f"Satellite dependencies unavailable: {exc}"}

    try:
        # Small search area around the selected coordinate.
        delta = 0.025
        bbox = [lon-delta, lat-delta, lon+delta, lat+delta]
        end = date.today()
        start = end - timedelta(days=days_back)
        catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
        search = catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=bbox,
            datetime=f"{start.isoformat()}/{end.isoformat()}",
            query={"eo:cloud_cover": {"lt": 35}},
            max_items=20
        )
        items = list(search.items())
        if not items:
            return {"ndvi": None, "scene_date": None, "cloud_cover": None,
                    "message": "No suitable Sentinel-2 scene found in the selected date window."}

        items.sort(key=lambda x: x.properties.get("eo:cloud_cover", 100))
        item = planetary_computer.sign(items[0])
        # Read the central pixel from the red and near-infrared COGs.
        red_href = item.assets["B04"].href
        nir_href = item.assets["B08"].href
        with rasterio.open(red_href) as red_src, rasterio.open(nir_href) as nir_src:
            r = next(red_src.sample([(lon, lat)]))[0].astype(float)
            n = next(nir_src.sample([(lon, lat)]))[0].astype(float)
        if r + n == 0:
            ndvi = None
        else:
            ndvi = float((n-r)/(n+r))
        return {
            "ndvi": ndvi,
            "scene_date": str(item.datetime.date()) if item.datetime else None,
            "cloud_cover": item.properties.get("eo:cloud_cover"),
            "message": "NDVI estimated from the nearest selected Sentinel-2 scene."
        }
    except Exception as exc:
        return {"ndvi": None, "scene_date": None, "cloud_cover": None,
                "message": f"Satellite lookup failed (weather analysis can still run): {exc}"}
