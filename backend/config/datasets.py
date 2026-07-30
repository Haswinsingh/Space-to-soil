import os
from backend.utils.config import BASE_DIR

DATASETS = {
    "EuroSAT": os.path.join(BASE_DIR, "datasets", "images", "EuroSAT"),
    "PlantVillage": os.path.join(BASE_DIR, "datasets", "images", "PlantVillage"),
    "Sentinel2": os.path.join(BASE_DIR, "datasets", "images", "Sentinel2")
}
