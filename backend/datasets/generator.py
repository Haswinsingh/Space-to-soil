import os
import pandas as pd
import numpy as np

def generate_agricultural_dataset(num_samples=200):
    """
    Generates a synthetic agricultural dataset containing spectral index features,
    crop health labels, and yield values based on agricultural physics.
    
    Classes:
    0: Healthy
    1: Water Stress
    2: Nitrogen Deficiency
    3: Disease
    4: Severe Stress
    """
    np.random.seed(42)
    
    data = []
    
    classes = {
        0: {"name": "Healthy", "ndvi": (0.65, 0.85), "ndwi": (0.1, 0.4), "ci": (4.0, 7.0), "yield": (8.0, 12.0)},
        1: {"name": "Water Stress", "ndvi": (0.45, 0.6), "ndwi": (-0.4, -0.1), "ci": (2.5, 4.5), "yield": (4.5, 7.0)},
        2: {"name": "Nitrogen Deficiency", "ndvi": (0.4, 0.55), "ndwi": (0.0, 0.3), "ci": (0.8, 1.8), "yield": (5.0, 7.5)},
        3: {"name": "Disease", "ndvi": (0.35, 0.55), "ndwi": (0.1, 0.3), "ci": (2.0, 4.0), "yield": (3.5, 6.0)},
        4: {"name": "Severe Stress", "ndvi": (0.12, 0.3), "ndwi": (-0.5, 0.0), "ci": (0.2, 1.0), "yield": (1.0, 3.2)}
    }
    
    for i in range(num_samples):
        # Even class distribution
        cls_id = i % 5
        params = classes[cls_id]
        
        # Spectral indices
        ndvi = np.random.uniform(*params["ndvi"])
        ndwi = np.random.uniform(*params["ndwi"])
        ci = np.random.uniform(*params["ci"])
        
        # Correlated indices
        evi = ndvi * 0.8 + np.random.uniform(-0.05, 0.05)
        savi = ndvi * 0.9 + np.random.uniform(-0.03, 0.03)
        
        # Standard deviations (simulating spatial variance in field)
        ndvi_std = np.random.uniform(0.02, 0.05) if cls_id != 3 else np.random.uniform(0.06, 0.12) # high variance in disease
        evi_std = ndvi_std * 0.8
        savi_std = ndvi_std * 0.9
        ndwi_std = np.random.uniform(0.02, 0.06)
        ci_std = np.random.uniform(0.1, 0.4)
        
        # Min / Max / Quantiles
        features = {
            "ndvi_mean": ndvi,
            "ndvi_std": ndvi_std,
            "ndvi_min": ndvi - 2 * ndvi_std,
            "ndvi_max": ndvi + 2 * ndvi_std,
            "ndvi_q25": ndvi - 0.67 * ndvi_std,
            "ndvi_q75": ndvi + 0.67 * ndvi_std,
            
            "evi_mean": evi,
            "evi_std": evi_std,
            "evi_min": evi - 2 * evi_std,
            "evi_max": evi + 2 * evi_std,
            "evi_q25": evi - 0.67 * evi_std,
            "evi_q75": evi + 0.67 * evi_std,
            
            "savi_mean": savi,
            "savi_std": savi_std,
            "savi_min": savi - 2 * savi_std,
            "savi_max": savi + 2 * savi_std,
            "savi_q25": savi - 0.67 * savi_std,
            "savi_q75": savi + 0.67 * savi_std,
            
            "ndwi_mean": ndwi,
            "ndwi_std": ndwi_std,
            "ndwi_min": ndwi - 2 * ndwi_std,
            "ndwi_max": ndwi + 2 * ndwi_std,
            "ndwi_q25": ndwi - 0.67 * ndwi_std,
            "ndwi_q75": ndwi + 0.67 * ndwi_std,
            
            "ci_mean": ci,
            "ci_std": ci_std,
            "ci_min": max(0.0, ci - 2 * ci_std),
            "ci_max": ci + 2 * ci_std,
            "ci_q25": max(0.0, ci - 0.67 * ci_std),
            "ci_q75": ci + 0.67 * ci_std,
            
            "label": cls_id,
            "label_name": params["name"],
            "yield": float(np.random.uniform(*params["yield"]))
        }
        
        data.append(features)
        
    df = pd.DataFrame(data)
    return df

def setup_dataset_files(base_dir):
    """
    Creates and saves the dataset CSVs in the backend workspace.
    """
    datasets_dir = os.path.join(base_dir, "datasets")
    os.makedirs(datasets_dir, exist_ok=True)
    
    csv_path = os.path.join(datasets_dir, "agricultural_data.csv")
    if not os.path.exists(csv_path):
        df = generate_agricultural_dataset(250)
        df.to_csv(csv_path, index=False)
        print(f"Generated synthetic agricultural dataset at {csv_path}")
    return csv_path
