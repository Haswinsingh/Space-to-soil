import os
import json
import logging
import datetime
import requests
import ee
from google.oauth2.credentials import Credentials

logger = logging.getLogger("uvicorn.error")

# Caching dictionary to prevent repeatedly fetching the same coordinates/image
_gee_cache = {}

def initialize_gee():
    try:
        # Try default credentials/project first
        ee.Initialize()
        logger.info("Connected to Earth Engine (Default credentials)")
        print("Connected to Earth Engine (Default credentials)")
        return True
    except Exception as e:
        logger.warning(f"Default Earth Engine connection failed: {e}. Attempting manual path.")
        try:
            creds_path = os.path.expanduser("~/.config/earthengine/credentials")
            if os.path.exists(creds_path):
                with open(creds_path, "r") as f:
                    creds_data = json.load(f)
                creds = Credentials(
                    token=None,
                    refresh_token=creds_data["refresh_token"],
                    client_id=creds_data["client_id"],
                    client_secret=creds_data["client_secret"],
                    token_uri="https://oauth2.googleapis.com/token",
                    scopes=creds_data["scopes"]
                )
                project = creds_data.get("project")
                ee.Initialize(credentials=creds, project=project)
                logger.info(f"Connected to Earth Engine (Manual project: {project})")
                print(f"Connected to Earth Engine (Manual project: {project})")
                return True
            else:
                logger.error("No Earth Engine credentials file found.")
                return False
        except Exception as manual_err:
            logger.exception(f"Manual Earth Engine connection failed: {manual_err}")
            return False

# Initialize on import
initialize_gee()

def analyze_field(latitude: float, longitude: float, polygon_coords: list = None, use_landsat: bool = False):
    """
    Queries Sentinel-2 (or Landsat-8) for a given coordinate/polygon, computes indices,
    calculates stats, generates map tile URLs, and downloads a thumbnail image for ML.
    """
    # Create GEE Geometry
    if polygon_coords:
        # Expected polygon format: list of [lng, lat]
        geometry = ee.Geometry.Polygon(polygon_coords)
    else:
        # fallback to point buffer (500m)
        geometry = ee.Geometry.Point([longitude, latitude]).buffer(500)

    # Caching key based on geometry bounds
    bounds_coords = geometry.bounds().getInfo()['coordinates'][0]
    cache_key = str(bounds_coords) + ("_landsat" if use_landsat else "_sentinel")

    if cache_key in _gee_cache:
        logger.info("Returning cached Earth Engine analysis.")
        return _gee_cache[cache_key]

    # Date range: past 12 months to guarantee cloud-free imagery
    now = datetime.datetime.utcnow()
    one_year_ago = now - datetime.timedelta(days=365)
    start_date = one_year_ago.strftime("%Y-%m-%d")
    end_date = now.strftime("%Y-%m-%d")

    collection_name = "LANDSAT/LC08/C02/T1_L2" if use_landsat else "COPERNICUS/S2_SR_HARMONIZED"
    cloud_prop = "CLOUD_COVER" if use_landsat else "CLOUDY_PIXEL_PERCENTAGE"

    # Load collection
    collection = ee.ImageCollection(collection_name) \
        .filterBounds(geometry) \
        .filterDate(start_date, end_date)

    # Sort by cloud cover and get the best image
    image = collection.sort(cloud_prop).first()
    
    # Check if we got any image
    if image.getInfo() is None:
        raise ValueError("No satellite imagery found for the specified region and date range.")

    logger.info("Loaded Sentinel image" if not use_landsat else "Loaded Landsat image")
    print("Loaded Sentinel image" if not use_landsat else "Loaded Landsat image")

    # Get actual image date and cloud cover
    img_info = image.getInfo()
    img_properties = img_info.get("properties", {})
    acq_date = img_properties.get("system:time_start", 0) / 1000.0
    acq_date_str = datetime.datetime.utcfromtimestamp(acq_date).strftime("%Y-%m-%d")
    cloud_pct = img_properties.get(cloud_prop, 0.0)

    # Band mapping
    if use_landsat:
        # Landsat 8 Band Mapping (Scale by 0.0000275 and add -0.2 to get surface reflectance)
        nir = image.select('SR_B5').multiply(0.0000275).add(-0.2)
        red = image.select('SR_B4').multiply(0.0000275).add(-0.2)
        green = image.select('SR_B3').multiply(0.0000275).add(-0.2)
        blue = image.select('SR_B2').multiply(0.0000275).add(-0.2)
        swir = image.select('SR_B6').multiply(0.0000275).add(-0.2)
    else:
        # Sentinel-2 Band Mapping (Scale by 0.0001 to get reflectance)
        nir = image.select('B8').multiply(0.0001)
        red = image.select('B4').multiply(0.0001)
        green = image.select('B3').multiply(0.0001)
        blue = image.select('B2').multiply(0.0001)
        swir = image.select('B11').multiply(0.0001)

    # Compute Indices
    ndvi = nir.subtract(red).divide(nir.add(red)).rename('ndvi')
    print("Computed NDVI")

    # EVI
    evi = ee.Image(2.5).multiply(
        nir.subtract(red).divide(
            nir.add(ee.Image(6.0).multiply(red)).subtract(ee.Image(7.5).multiply(blue)).add(ee.Image(1.0))
        )
    ).rename('evi')

    # SAVI
    savi = nir.subtract(red).multiply(1.5).divide(nir.add(red).add(0.5)).rename('savi')

    # NDWI (Water Index - McFeeters)
    ndwi = green.subtract(nir).divide(green.add(nir)).rename('ndwi')

    # GNDVI
    gndvi = nir.subtract(green).divide(nir.add(green)).rename('gndvi')

    # MSAVI
    msavi = ee.Image(2.0).multiply(nir).add(1.0).subtract(
        ee.Image(2.0).multiply(nir).add(1.0).pow(2.0).subtract(
            ee.Image(8.0).multiply(nir.subtract(red))
        ).sqrt()
    ).divide(2.0).rename('msavi')

    # Chlorophyll Index (CI)
    ci = nir.divide(green).subtract(1.0).rename('ci')

    # Combine indices into a single image
    indices_img = ee.Image.cat([ndvi, evi, savi, ndwi, gndvi, msavi, ci])

    # Compute Statistics
    reducers = ee.Reducer.mean() \
        .combine(ee.Reducer.stdDev(), sharedInputs=True) \
        .combine(ee.Reducer.minMax(), sharedInputs=True) \
        .combine(ee.Reducer.percentile([25, 75]), sharedInputs=True)

    stats = indices_img.reduceRegion(
        reducer=reducers,
        geometry=geometry,
        scale=10,
        maxPixels=1e9
    ).getInfo()

    # Calculate Area
    area_ha = geometry.area().divide(10000).getInfo()

    # Map stats keys to tabular model keys (converting from float64 to float for serialization)
    def clean_val(k, default=0.0):
        val = stats.get(k)
        if val is None:
            return default
        return float(val)

    features = {
        "ndvi_mean": clean_val("ndvi_mean"),
        "ndvi_std": clean_val("ndvi_stdDev"),
        "ndvi_min": clean_val("ndvi_min"),
        "ndvi_max": clean_val("ndvi_max"),
        "ndvi_q25": clean_val("ndvi_p25"),
        "ndvi_q75": clean_val("ndvi_p75"),

        "evi_mean": clean_val("evi_mean"),
        "evi_std": clean_val("evi_stdDev"),
        "evi_min": clean_val("evi_min"),
        "evi_max": clean_val("evi_max"),
        "evi_q25": clean_val("evi_p25"),
        "evi_q75": clean_val("evi_p75"),

        "savi_mean": clean_val("savi_mean"),
        "savi_std": clean_val("savi_stdDev"),
        "savi_min": clean_val("savi_min"),
        "savi_max": clean_val("savi_max"),
        "savi_q25": clean_val("savi_p25"),
        "savi_q75": clean_val("savi_p75"),

        "ndwi_mean": clean_val("ndwi_mean"),
        "ndwi_std": clean_val("ndwi_stdDev"),
        "ndwi_min": clean_val("ndwi_min"),
        "ndwi_max": clean_val("ndwi_max"),
        "ndwi_q25": clean_val("ndwi_p25"),
        "ndwi_q75": clean_val("ndwi_p75"),

        "ci_mean": clean_val("ci_mean"),
        "ci_std": clean_val("ci_stdDev"),
        "ci_min": clean_val("ci_min"),
        "ci_max": clean_val("ci_max"),
        "ci_q25": clean_val("ci_p25"),
        "ci_q75": clean_val("ci_p75")
    }

    # Generate Map Tile URLs
    # Visualizations
    palette_veg = ['FFFFFF', 'CE7E45', 'DF923D', 'F1B555', 'FCD163', '99B718', '74A408', '4A7E03', '325B02', '244002']
    palette_water = ['FFFFFF', 'A5F2F3', '4AC5F2', '1489F2', '0830F2']

    def get_tile_url(img, min_val, max_val, palette=None, bands=None):
        vis = {'min': min_val, 'max': max_val}
        if palette:
            vis['palette'] = palette
        if bands:
            vis['bands'] = bands
        try:
            map_id = img.getMapId(vis)
            url = map_id['tile_fetcher'].url_format
            print("Returned tile URL")
            return url
        except Exception as tile_err:
            logger.error(f"Failed to generate tile URL: {tile_err}")
            return ""

    tile_urls = {
        "ndvi": get_tile_url(ndvi, -0.1, 0.9, palette_veg),
        "evi": get_tile_url(evi, -0.1, 0.9, palette_veg),
        "savi": get_tile_url(savi, -0.1, 0.9, palette_veg),
        "ndwi": get_tile_url(ndwi, -0.5, 0.5, palette_water),
        "ci": get_tile_url(ci, 0.0, 5.0, palette_veg),
        "true_color": get_tile_url(image, 0.0, 3000.0, bands=['B4', 'B3', 'B2']),
        "false_color": get_tile_url(image, 0.0, 4000.0, bands=['B8', 'B4', 'B3'])
    }

    # Prepare Download of RGB Thumbnail for ML predictions
    import uuid
    file_id = str(uuid.uuid4())
    uploads_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "uploads")
    os.makedirs(uploads_dir, exist_ok=True)
    raw_filename = f"raw_gee_{file_id}.png"
    raw_path = os.path.join(uploads_dir, raw_filename)

    try:
        bounds = geometry.bounds()
        # Scale to max 3000 reflectance range for visualization
        thumb_url = image.select(['B4', 'B3', 'B2']).getThumbURL({
            'min': 0,
            'max': 3000,
            'region': bounds,
            'dimensions': 256,
            'format': 'png'
        })
        resp = requests.get(thumb_url, timeout=15)
        if resp.status_code == 200:
            with open(raw_path, "wb") as f:
                f.write(resp.content)
            logger.info(f"Downloaded field thumbnail to {raw_path}")
        else:
            logger.error(f"Failed to fetch GEE thumbnail image: {resp.status_code}")
            raw_path = None
    except Exception as thumb_err:
        logger.error(f"Error fetching GEE field thumbnail: {thumb_err}")
        raw_path = None

    result = {
        "success": True,
        "file_id": file_id,
        "image_path": raw_path,
        "image_url": f"/uploads/{raw_filename}" if raw_path else None,
        "acquisition_date": acq_date_str,
        "cloud_cover": float(cloud_pct),
        "area_ha": float(area_ha),
        "features": features,
        "extra_indices": {
            "gndvi_mean": clean_val("gndvi_mean"),
            "msavi_mean": clean_val("msavi_mean")
        },
        "tile_urls": tile_urls
    }

    _gee_cache[cache_key] = result
    return result
