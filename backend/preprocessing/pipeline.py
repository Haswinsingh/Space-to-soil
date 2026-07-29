import os
import cv2
import numpy as np

def load_multispectral_image(filepath: str):
    """
    Loads an image file (JPEG, PNG, TIFF, GeoTIFF) and returns Red, Green, Blue, and NIR bands.
    If the image is standard RGB (3 channels), it synthesizes a realistic NIR band based on
    vegetation spectral signatures.
    """
    # Check extension
    _, ext = os.path.splitext(filepath.lower())
    
    # Try using Rasterio if available and it's a TIFF/TIF
    if ext in ['.tif', '.tiff', '.geotiff']:
        try:
            import rasterio
            with rasterio.open(filepath) as src:
                # Read bands
                # Standard Sentinel-2/Landsat mapping: Band 2=Blue, Band 3=Green, Band 4=Red, Band 8/5=NIR
                count = src.count
                if count >= 4:
                    blue = src.read(1).astype(np.float32)
                    green = src.read(2).astype(np.float32)
                    red = src.read(3).astype(np.float32)
                    nir = src.read(4).astype(np.float32)
                    
                    # Normalize to 0-1
                    def norm(b):
                        b_min, b_max = b.min(), b.max()
                        if b_max > b_min:
                            return (b - b_min) / (b_max - b_min)
                        return b
                    
                    return norm(red), norm(green), norm(blue), norm(nir), f"{count}-Band GeoTIFF"
        except Exception as e:
            # Fallback to OpenCV if rasterio fails
            pass

    # Read image using OpenCV
    img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not open image file: {filepath}")
    
    # Handle based on channels
    if len(img.shape) == 2:
        # Grayscale
        gray = img.astype(np.float32) / 255.0
        red = green = blue = gray
        # Synthesize NIR (lower reflection on soil/shadow, higher on plants)
        nir = gray * 1.2
        nir = np.clip(nir, 0.0, 1.0)
        source = "Grayscale Image (NIR Synthesized)"
    elif img.shape[2] == 3:
        # BGR (OpenCV Default)
        bgr = img.astype(np.float32) / 255.0
        blue, green, red = bgr[:, :, 0], bgr[:, :, 1], bgr[:, :, 2]
        # Synthesize NIR band: healthy plants reflect NIR strongly, which correlates with Green
        # and has low absorption compared to Red (which is absorbed by chlorophyll)
        # Synthetic NIR = Green * 1.3 - Red * 0.3 + 0.1
        nir = green * 1.3 - red * 0.3 + 0.1
        nir = np.clip(nir, 0.01, 1.0)
        source = "RGB Image (NIR Synthesized)"
    elif img.shape[2] >= 4:
        # Multi-band or BGRA
        bands = img.astype(np.float32) / 255.0
        blue, green, red, nir = bands[:, :, 0], bands[:, :, 1], bands[:, :, 2], bands[:, :, 3]
        source = f"{img.shape[2]}-Band Multi-Channel Image"
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")
        
    return red, green, blue, nir, source

def preprocess_bands(red, green, blue, nir):
    """
    Performs resize, noise reduction, and contrast enhancement on individual bands.
    """
    # 1. Resize to a consistent size (e.g. 512x512) for uniformity in processing
    target_size = (512, 512)
    red_r = cv2.resize(red, target_size, interpolation=cv2.INTER_LINEAR)
    green_r = cv2.resize(green, target_size, interpolation=cv2.INTER_LINEAR)
    blue_r = cv2.resize(blue, target_size, interpolation=cv2.INTER_LINEAR)
    nir_r = cv2.resize(nir, target_size, interpolation=cv2.INTER_LINEAR)
    
    # 2. Noise Reduction: Apply Gaussian Blur
    kernel_size = (3, 3)
    red_blur = cv2.GaussianBlur(red_r, kernel_size, 0)
    green_blur = cv2.GaussianBlur(green_r, kernel_size, 0)
    blue_blur = cv2.GaussianBlur(blue_r, kernel_size, 0)
    nir_blur = cv2.GaussianBlur(nir_r, kernel_size, 0)
    
    # 3. Contrast Enhancement: Contrast stretching (CLAHE equivalent for normalized floats)
    def enhance_contrast(band):
        p2, p98 = np.percentile(band, (2, 98))
        if p98 > p2:
            enhanced = (band - p2) / (p98 - p2)
            return np.clip(enhanced, 0.0, 1.0)
        return band

    red_enh = enhance_contrast(red_blur)
    green_enh = enhance_contrast(green_blur)
    blue_enh = enhance_contrast(blue_blur)
    nir_enh = enhance_contrast(nir_blur)
    
    return red_enh, green_enh, blue_enh, nir_enh

def calculate_indices(red, green, blue, nir):
    """
    Calculates Vegetation and Water Indices: NDVI, EVI, SAVI, NDWI, Chlorophyll Index.
    Safe division is used to avoid NaNs.
    """
    eps = 1e-6
    
    # 1. NDVI (Normalized Difference Vegetation Index)
    # NDVI = (NIR - Red) / (NIR + Red)
    ndvi = (nir - red) / (nir + red + eps)
    ndvi = np.clip(ndvi, -1.0, 1.0)
    
    # 2. EVI (Enhanced Vegetation Index)
    # EVI = 2.5 * ((NIR - Red) / (NIR + 6 * Red - 7.5 * Blue + 1))
    evi = 2.5 * (nir - red) / (nir + 6.0 * red - 7.5 * blue + 1.0 + eps)
    evi = np.clip(evi, -1.0, 1.0)
    
    # 3. SAVI (Soil Adjusted Vegetation Index)
    # SAVI = ((NIR - Red) / (NIR + Red + 0.5)) * 1.5
    savi = ((nir - red) / (nir + red + 0.5 + eps)) * 1.5
    savi = np.clip(savi, -1.0, 1.0)
    
    # 4. NDWI (Normalized Difference Water Index)
    # NDWI = (Green - NIR) / (Green + NIR)
    ndwi = (green - nir) / (green + nir + eps)
    ndwi = np.clip(ndwi, -1.0, 1.0)
    
    # 5. Chlorophyll Index (CI_green)
    # CI_green = (NIR / Green) - 1
    ci = (nir / (green + eps)) - 1.0
    ci = np.clip(ci, -2.0, 10.0)  # CI can have larger positive values
    
    return ndvi, evi, savi, ndwi, ci

def extract_features(ndvi, evi, savi, ndwi, ci):
    """
    Extracts statistical features from computed indices to build the ML/QML feature vector.
    """
    features = {}
    for name, index_arr in [("ndvi", ndvi), ("evi", evi), ("savi", savi), ("ndwi", ndwi), ("ci", ci)]:
        features[f"{name}_mean"] = float(np.mean(index_arr))
        features[f"{name}_std"] = float(np.std(index_arr))
        features[f"{name}_min"] = float(np.min(index_arr))
        features[f"{name}_max"] = float(np.max(index_arr))
        features[f"{name}_q25"] = float(np.percentile(index_arr, 25))
        features[f"{name}_q75"] = float(np.percentile(index_arr, 75))
        
    return features

def save_color_mapped_index(index_arr, min_val, max_val, cmap_code, output_path):
    """
    Saves an index array as a color-mapped visualization image.
    """
    # Normalize index array to 0-255
    norm_arr = (index_arr - min_val) / (max_val - min_val + 1e-8)
    norm_arr = np.clip(norm_arr, 0.0, 1.0)
    gray = (norm_arr * 255).astype(np.uint8)
    
    # Apply colormap
    color_mapped = cv2.applyColorMap(gray, cmap_code)
    cv2.imwrite(output_path, color_mapped)

def run_preprocessing_pipeline(filepath: str, output_dir: str, prefix: str = ""):
    """
    Executes the entire preprocessing pipeline and saves visualizations.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load bands
    red, green, blue, nir, source = load_multispectral_image(filepath)
    
    # Preprocess (Resize, denoise, enhance)
    red_p, green_p, blue_p, nir_p = preprocess_bands(red, green, blue, nir)
    
    # Calculate Indices
    ndvi, evi, savi, ndwi, ci = calculate_indices(red_p, green_p, blue_p, nir_p)
    
    # Save visualizations
    # 1. RGB original (rescaled for display)
    rgb_display = np.stack([blue_p, green_p, red_p], axis=-1)  # BGR
    rgb_path = os.path.join(output_dir, f"{prefix}rgb_enhanced.png")
    cv2.imwrite(rgb_path, (rgb_display * 255).astype(np.uint8))
    
    # 2. Save individual bands
    cv2.imwrite(os.path.join(output_dir, f"{prefix}band_red.png"), (red_p * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(output_dir, f"{prefix}band_green.png"), (green_p * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(output_dir, f"{prefix}band_blue.png"), (blue_p * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(output_dir, f"{prefix}band_nir.png"), (nir_p * 255).astype(np.uint8))
    
    # 3. Save Colorized Indices
    # NDVI: Green represents high index, Red represents low. We use COLORMAP_JET or COLORMAP_SUMMER.
    # Jet has nice red-to-blue. Let's use COLORMAP_SPEED or COLORMAP_WINTER or Jet.
    # We will use Jet for general indices since it shows stress vs health nicely (Red=low, Yellow=mid, Green=high).
    save_color_mapped_index(ndvi, -0.2, 1.0, cv2.COLORMAP_JET, os.path.join(output_dir, f"{prefix}index_ndvi.png"))
    save_color_mapped_index(evi, -0.2, 1.0, cv2.COLORMAP_JET, os.path.join(output_dir, f"{prefix}index_evi.png"))
    save_color_mapped_index(savi, -0.2, 1.0, cv2.COLORMAP_JET, os.path.join(output_dir, f"{prefix}index_savi.png"))
    save_color_mapped_index(ndwi, -1.0, 1.0, cv2.COLORMAP_WINTER, os.path.join(output_dir, f"{prefix}index_ndwi.png"))  # Winter is blue-green
    save_color_mapped_index(ci, 0.0, 6.0, cv2.COLORMAP_SUMMER, os.path.join(output_dir, f"{prefix}index_ci.png"))
    
    # Extract features
    features = extract_features(ndvi, evi, savi, ndwi, ci)
    
    # Return results
    return {
        "source": source,
        "features": features,
        "visualizations": {
            "rgb": f"/uploads/processed/{prefix}rgb_enhanced.png",
            "red": f"/uploads/processed/{prefix}band_red.png",
            "green": f"/uploads/processed/{prefix}band_green.png",
            "blue": f"/uploads/processed/{prefix}band_blue.png",
            "nir": f"/uploads/processed/{prefix}band_nir.png",
            "ndvi": f"/uploads/processed/{prefix}index_ndvi.png",
            "evi": f"/uploads/processed/{prefix}index_evi.png",
            "savi": f"/uploads/processed/{prefix}index_savi.png",
            "ndwi": f"/uploads/processed/{prefix}index_ndwi.png",
            "ci": f"/uploads/processed/{prefix}index_ci.png"
        }
    }
