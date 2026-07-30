import os
from dotenv import load_dotenv

load_dotenv()

# App Configurations
APP_NAME = "QuantumCrop AI"
SECRET_KEY = os.getenv("SECRET_KEY", "quantum_crop_ai_super_secret_jwt_key_2026")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

# Database Configuration
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/spacetosoil")
DATABASE_NAME = os.getenv("DATABASE_NAME", "spacetosoil")

# Base and Subdirectory Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

# New Dataset & Model Path Preferences
DATASET_IMAGE_PATH = os.path.join(BASE_DIR, "datasets", "images")
DATASET_CSV_PATH = os.path.join(BASE_DIR, "datasets", "csv")
MODEL_PATH = os.path.join(BASE_DIR, "models")
UPLOAD_PATH = UPLOAD_DIR

# Automatically create all required directories
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(DATASET_IMAGE_PATH, exist_ok=True)
os.makedirs(DATASET_CSV_PATH, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)

# Pre-generate subfolders for requested image dataset types
for dataset_subfolder in ["EuroSAT", "PlantVillage", "RadiantEarth", "Sentinel2"]:
    os.makedirs(os.path.join(DATASET_IMAGE_PATH, dataset_subfolder), exist_ok=True)

# Training Modes & Optimizations (Task 8 Requirements)
TRAIN_MODE = os.getenv("TRAIN_MODE", "development")  # "development" or "production"
DATASET_PERCENTAGE = float(os.getenv("DATASET_PERCENTAGE", "0.2"))  # Dev mode default 20%
EPOCHS = int(os.getenv("EPOCHS", "2"))                              # Dev mode default 2 epochs
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "64"))                    # Dev mode default 64 batch size
