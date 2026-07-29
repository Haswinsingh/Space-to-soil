import os
import pandas as pd
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from backend.utils import config

def detect_dataset_type():
    """
    Scans image and csv dataset paths to detect the active dataset.
    Returns:
        type: "image" or "csv" or None
        path: Path to the dataset or None
    """
    # 1. Check for CSV datasets first in the designated folder
    csv_dir = config.DATASET_CSV_PATH
    if os.path.exists(csv_dir):
        csv_files = [f for f in os.listdir(csv_dir) if f.lower().endswith('.csv')]
        if csv_files:
            # Prioritize the default generated dataset if present
            target_csv = "agricultural_data.csv" if "agricultural_data.csv" in csv_files else csv_files[0]
            return "csv", os.path.join(csv_dir, target_csv)
            
    # 2. Check for image datasets (EuroSAT, RadiantEarth, Sentinel2, PlantVillage)
    image_dir = config.DATASET_IMAGE_PATH
    if os.path.exists(image_dir):
        subfolders = ["EuroSAT", "PlantVillage", "RadiantEarth", "Sentinel2"]
        for sf in subfolders:
            sf_path = os.path.join(image_dir, sf)
            if os.path.isdir(sf_path):
                img_count = 0
                for root, _, files in os.walk(sf_path):
                    for f in files:
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                            img_count += 1
                if img_count > 0:
                    return "image", sf_path

        # If subfolders didn't match, scan any other subdirectory directly
        other_dirs = [d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))]
        for od in other_dirs:
            od_path = os.path.join(image_dir, od)
            img_count = 0
            for root, _, files in os.walk(od_path):
                for f in files:
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                        img_count += 1
            if img_count > 0:
                return "image", od_path
                
    # Fallback to the root datasets directory if any
    fallback_csv = os.path.join(config.BASE_DIR, "datasets", "agricultural_data.csv")
    if os.path.exists(fallback_csv):
        return "csv", fallback_csv

    return None, None

def load_image_dataset(dataset_path, batch_size=32):
    print(f"[DATASET LOADING] Initializing EuroSAT image dataset loader from: {dataset_path}")
    print("[PREPROCESSING] Applying transforms: Resize(64,64), RandomHorizontalFlip, RandomRotation(15), ToTensor, Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])")
    
    transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert('RGB') if hasattr(img, 'convert') else img),
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = ImageFolder(root=dataset_path, transform=transform)
    if len(dataset) == 0:
        print("[DATASET ERROR] Dataset contains no samples!")
        raise ValueError("Dataset contains no samples.")
        
    total_len = len(dataset)
    train_size = int(0.8 * total_len)
    val_size = int(0.1 * total_len)
    test_size = total_len - train_size - val_size
    
    print(f"[DATASET LOADING] Detected {len(dataset.classes)} classes: {dataset.classes}")
    print(f"[DATASET SPLIT] Split configurations: Train={train_size} (80%), Val={val_size} (10%), Test={test_size} (10%)")
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, total_len, len(dataset.classes), dataset.classes

def load_csv_dataset(csv_path):
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        raise ValueError("Dataset contains no samples.")
        
    # Auto-detect target column
    potential_targets = ["label", "class", "target", "crop_health", "health"]
    target_col = None
    for col in potential_targets:
        if col in df.columns:
            target_col = col
            break
    if not target_col:
        target_col = df.columns[-1]
        
    # Feature columns (all numeric/encoded columns except label and yield)
    exclude_cols = [target_col, "label_name", "yield", "id", "file_id", "created_at"]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Missing values imputation
    for col in df.columns:
        if df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].mean())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])
                
    # Encode categorical columns in features
    categorical_cols = [col for col in feature_cols if not pd.api.types.is_numeric_dtype(df[col])]
    if categorical_cols:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
    # Encode target labels
    le = LabelEncoder()
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        df[target_col] = le.fit_transform(df[target_col].astype(str))
        classes = [str(c) for c in le.classes_]
    else:
        unique_classes = sorted(df[target_col].unique())
        classes = [f"Class {c}" for c in unique_classes]
        
    X = df[feature_cols].values
    y = df[target_col].values
    
    # Extract yield values if present, else fallback
    y_yield = df["yield"].values if "yield" in df.columns else np.random.uniform(2.0, 10.0, size=len(df))
    
    X_train, X_test, y_train, y_test, y_yield_train, y_yield_test = train_test_split(
        X, y, y_yield, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, y_yield_train, y_yield_test, len(classes), classes, scaler, feature_cols
