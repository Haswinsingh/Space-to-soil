import os
import pandas as pd
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from PIL import Image
from backend.utils import config

# Task 2: CachedDataset wrapper to cache decoded image tensors in memory
class CachedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.cache = {}
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        if idx not in self.cache:
            self.cache[idx] = self.dataset[idx]
        return self.cache[idx]

class TransformSubset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
        
    def __len__(self):
        return len(self.indices)
        
    def __getitem__(self, idx):
        path, target = self.dataset.samples[self.indices[idx]]
        img = self.dataset.loader(path)
        if self.transform:
            img = self.transform(img)
        return img, target

def detect_original_image_size(dataset_path):
    """Scans the dataset path to find the first image and returns its size (width, height)."""
    for root, _, files in os.walk(dataset_path):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.geotiff')):
                img_path = os.path.join(root, f)
                try:
                    with Image.open(img_path) as img:
                        return img.size
                except Exception:
                    pass
    return (64, 64)  # Default fallback

def detect_dataset_type():
    """
    Scans image and csv dataset paths to detect the active dataset.
    Returns:
        type: "image" or "csv" or None
        path: Path to the dataset or None
    """
    csv_dir = config.DATASET_CSV_PATH
    if os.path.exists(csv_dir):
        csv_files = [f for f in os.listdir(csv_dir) if f.lower().endswith('.csv')]
        if csv_files:
            target_csv = "agricultural_data.csv" if "agricultural_data.csv" in csv_files else csv_files[0]
            return "csv", os.path.join(csv_dir, target_csv)
            
    image_dir = config.DATASET_IMAGE_PATH
    if os.path.exists(image_dir):
        subfolders = ["EuroSAT", "PlantVillage", "Sentinel2"]
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
                
    fallback_csv = os.path.join(config.BASE_DIR, "datasets", "agricultural_data.csv")
    if os.path.exists(fallback_csv):
        return "csv", fallback_csv

    return None, None

class DatasetLoader:
    def __init__(self, batch_size=None):
        # Allow dynamic override, else fall back to config
        self.batch_size = batch_size or config.BATCH_SIZE

    def load_dataset(self, dataset_name):
        from backend.config.datasets import DATASETS
        dataset_path = DATASETS.get(dataset_name)
        if not dataset_path or not os.path.exists(dataset_path):
            if os.path.exists(dataset_name):
                dataset_path = dataset_name
                dataset_name = os.path.basename(dataset_name)
            else:
                raise ValueError(f"Dataset path for {dataset_name} not found or doesn't exist.")

        # Automatically detect original image size
        original_size = detect_original_image_size(dataset_path)
        print(f"[DATASET LOADING] Detected original image size for {dataset_name}: {original_size[0]}x{original_size[1]}")

        # Transforms with augmentation
        transform = transforms.Compose([
            transforms.Lambda(lambda img: img.convert('RGB') if hasattr(img, 'convert') else img),
            transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomCrop(64, padding=4, padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        val_transform = transforms.Compose([
            transforms.Lambda(lambda img: img.convert('RGB') if hasattr(img, 'convert') else img),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        subdirs = [d.lower() for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
        has_split_folders = any(s in ["train", "val", "validation", "test"] for s in subdirs)

        if has_split_folders:
            train_dir = None
            val_dir = None
            test_dir = None
            for d in os.listdir(dataset_path):
                dl = d.lower()
                if dl == "train":
                    train_dir = os.path.join(dataset_path, d)
                elif dl in ["val", "validation"]:
                    val_dir = os.path.join(dataset_path, d)
                elif dl == "test":
                    test_dir = os.path.join(dataset_path, d)

            if train_dir:
                train_dataset = ImageFolder(root=train_dir, transform=transform)
                classes = train_dataset.classes
            else:
                raise ValueError("Train directory missing in split dataset folders.")

            if val_dir:
                val_dataset = ImageFolder(root=val_dir, transform=val_transform)
            else:
                train_len = len(train_dataset)
                val_len = int(0.1 * train_len)
                train_dataset, val_dataset = random_split(train_dataset, [train_len - val_len, val_len])

            if test_dir:
                test_dataset = ImageFolder(root=test_dir, transform=val_transform)
            else:
                train_len = len(train_dataset)
                test_len = int(0.1 * train_len)
                train_dataset, test_dataset = random_split(train_dataset, [train_len - test_len, test_len])
                
            # Task 1: Development Mode Subsetting for split folder datasets
            if config.TRAIN_MODE == "development":
                print(f"[DEV MODE] Subsetting training dataset to {config.DATASET_PERCENTAGE*100}% and validation to 10%...")
                from torch.utils.data import Subset
                
                # Subset training dataset
                train_len = len(train_dataset)
                subset_train_len = int(config.DATASET_PERCENTAGE * train_len)
                train_indices = np.random.choice(train_len, size=subset_train_len, replace=False)
                train_dataset = Subset(train_dataset, train_indices)
                
                # Subset validation dataset to 10%
                val_len = len(val_dataset)
                subset_val_len = max(1, int(0.1 * val_len))
                val_indices = np.random.choice(val_len, size=subset_val_len, replace=False)
                val_dataset = Subset(val_dataset, val_indices)
        else:
            dataset = ImageFolder(root=dataset_path, transform=transform)
            classes = dataset.classes
            total_len = len(dataset)
            
            # Task 1: Development Mode Subsetting for single folder datasets
            if config.TRAIN_MODE == "development":
                print(f"[DEV MODE] Subsetting training dataset to {config.DATASET_PERCENTAGE*100}% and validation to 10%...")
                train_size = int(config.DATASET_PERCENTAGE * 0.8 * total_len)
                val_size = max(1, int(0.1 * 0.1 * total_len))
                test_size = int(0.1 * total_len)
            else:
                train_size = int(0.8 * total_len)
                val_size = int(0.1 * total_len)
                test_size = total_len - train_size - val_size
            
            indices = list(range(total_len))
            np.random.seed(42)
            np.random.shuffle(indices)
            
            train_idx = indices[:train_size]
            val_idx = indices[train_size:train_size+val_size]
            test_idx = indices[train_size+val_size:train_size+val_size+test_size]
            
            train_dataset = TransformSubset(dataset, train_idx, transform)
            val_dataset = TransformSubset(dataset, val_idx, val_transform)
            test_dataset = TransformSubset(dataset, test_idx, val_transform)

        # Wrap with CachedDataset for high performance caching
        train_dataset = CachedDataset(train_dataset)
        val_dataset = CachedDataset(val_dataset)
        test_dataset = CachedDataset(test_dataset)

        # Task 1: Optimized DataLoader configuration for development & production modes
        cpu_count = os.cpu_count() or 1
        num_workers = min(4, cpu_count)
        cuda_available = torch.cuda.is_available()
        pin_memory = cuda_available
        persistent_workers = num_workers > 0

        # For Windows compatibility, override parameters to disable multiprocessing & pickling issues
        if os.name == "nt":
            num_workers = 0
            persistent_workers = False
            pin_memory = False

        print(f"[DATALOADER CONFIG] Mode: {config.TRAIN_MODE} | Batch Size: {self.batch_size} | num_workers: {num_workers} | pin_memory: {pin_memory} | persistent_workers: {persistent_workers}")

        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers
        )

        total_len = len(train_dataset) + len(val_dataset) + len(test_dataset)
        num_classes = len(classes)

        print(f"[DATASET LOADING] Detected {num_classes} classes: {classes}")
        print(f"[DATASET SPLIT] Split configurations: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

        return train_loader, val_loader, test_loader, total_len, num_classes, classes

    def load_eurosat(self):
        return self.load_dataset("EuroSAT")

    def load_plantvillage(self):
        return self.load_dataset("PlantVillage")

    def load_sentinel2(self):
        return self.load_dataset("Sentinel2")

def load_image_dataset(dataset_path, batch_size=32):
    """Backward compatibility wrapper that loads dataset dynamically using DatasetLoader."""
    dataset_name = os.path.basename(dataset_path)
    loader = DatasetLoader(batch_size=batch_size)
    try:
        return loader.load_dataset(dataset_name)
    except Exception:
        # Fallback to load directly via ImageFolder from the path
        transform = transforms.Compose([
            transforms.Lambda(lambda img: img.convert('RGB') if hasattr(img, 'convert') else img),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        dataset = ImageFolder(root=dataset_path, transform=transform)
        total_len = len(dataset)
        train_size = int(0.8 * total_len)
        val_size = int(0.1 * total_len)
        test_size = total_len - train_size - val_size
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, 
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        train_dataset = CachedDataset(train_dataset)
        val_dataset = CachedDataset(val_dataset)
        test_dataset = CachedDataset(test_dataset)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        return train_loader, val_loader, test_loader, total_len, len(dataset.classes), dataset.classes

def load_csv_dataset(csv_path):
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        raise ValueError("Dataset contains no samples.")
        
    potential_targets = ["label", "class", "target", "crop_health", "health"]
    target_col = None
    for col in potential_targets:
        if col in df.columns:
            target_col = col
            break
    if not target_col:
        target_col = df.columns[-1]
        
    exclude_cols = [target_col, "label_name", "yield", "id", "file_id", "created_at"]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    for col in df.columns:
        if df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].mean())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])
                
    categorical_cols = [col for col in feature_cols if not pd.api.types.is_numeric_dtype(df[col])]
    if categorical_cols:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
    le = LabelEncoder()
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        df[target_col] = le.fit_transform(df[target_col].astype(str))
        classes = [str(c) for c in le.classes_]
    else:
        unique_classes = sorted(df[target_col].unique())
        classes = [f"Class {c}" for c in unique_classes]
        
    X = df[feature_cols].values
    y = df[target_col].values
    
    y_yield = df["yield"].values if "yield" in df.columns else np.random.uniform(2.0, 10.0, size=len(df))
    
    X_train, X_test, y_train, y_test, y_yield_train, y_yield_test = train_test_split(
        X, y, y_yield, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, y_yield_train, y_yield_test, len(classes), classes, scaler, feature_cols
