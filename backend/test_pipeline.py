import os
import numpy as np
import cv2
from backend.preprocessing import pipeline
from backend.models.classical.classifiers import ClassicalPipeline
from backend.models.quantum.classifiers import QuantumPipeline
from backend.datasets.generator import setup_dataset_files
from backend.reports.generator import generate_crop_report
from backend.utils import config

def run_tests():
    print("=== STARTING BACKEND PIPELINE TESTS ===")
    
    # 1. Setup dataset files
    print("\n[STEP 1] Generating mock dataset...")
    csv_path = setup_dataset_files(config.BASE_DIR)
    assert os.path.exists(csv_path), "Dataset CSV generation failed!"
    print("✓ Dataset CSV created at:", csv_path)
    
    # 2. Test Multi-spectral image loading and index calculation
    print("\n[STEP 2] Simulating multispectral image upload...")
    test_img = np.zeros((256, 256, 4), dtype=np.uint8)
    # Fill bands with healthy-looking agricultural profiles (Red high absorption, NIR high reflection)
    test_img[:, :, 0] = 50   # Blue
    test_img[:, :, 1] = 120  # Green
    test_img[:, :, 2] = 40   # Red
    test_img[:, :, 3] = 220  # NIR
    
    test_img_path = os.path.join(config.UPLOAD_DIR, "test_input.png")
    cv2.imwrite(test_img_path, test_img)
    print("✓ Test multispectral input image created at:", test_img_path)
    
    print("\n[STEP 3] Running image preprocessing pipeline...")
    processed_dir = os.path.join(config.UPLOAD_DIR, "test_processed")
    res = pipeline.run_preprocessing_pipeline(test_img_path, processed_dir)
    
    assert "ndvi" in res["visualizations"], "Pipeline failed to produce NDVI mapping!"
    print("✓ Preprocessing pipeline executed successfully.")
    print("✓ Visualizations created:", list(res["visualizations"].keys()))
    print("✓ Extracted NDVI mean:", res["features"]["ndvi_mean"])
    
    # 3. Test Classical models training and inference
    print("\n[STEP 4] Testing Classical ML Models training...")
    classical = ClassicalPipeline(config.BASE_DIR)
    eurosat_path = os.path.join(config.BASE_DIR, "datasets", "images", "EuroSAT")
    train_res = classical.train_models("image", eurosat_path)
    print("✓ Classical RF accuracy:", train_res["random_forest"]["accuracy"])
    
    print("\n[STEP 5] Testing Classical ML inference...")
    pred_res = classical.predict(res["features"])
    print("✓ Crop Health classification:", pred_res["health_predictions"]["svm"]["class_name"])
    print("✓ Crop Yield prediction:", pred_res["yield_t_ha"], "Tons/Hectare")
    
    # 4. Test Quantum models training and circuit drawing
    print("\n[STEP 6] Testing Quantum ML Models training & circuit visualizations...")
    quantum = QuantumPipeline(config.BASE_DIR)
    quant_res = quantum.train_all_quantum_models("image", eurosat_path)
    print("✓ QSVM accuracy:", quant_res["qsvm"]["accuracy"])
    assert os.path.exists(os.path.join(config.BASE_DIR, "models", "quantum", "circuits", "qsvm_circuit.png")), "QSVM circuit visualization not drawn!"
    print("✓ Quantum Feature map drawn successfully.")
    
    # 5. Test PDF Report Generation
    print("\n[STEP 7] Generating sample PDF report...")
    pdf_out = os.path.join(config.REPORTS_DIR, "test_report.pdf")
    # Mock parameters
    class_results_pdf = {
        "random_forest": {"class_name": "Healthy", "confidence": 0.92, "benchmark_acc": 0.89},
        "svm": {"class_name": "Healthy", "confidence": 0.88, "benchmark_acc": 0.86}
    }
    quant_results_pdf = {
        "qsvm": {"class_name": "Healthy", "confidence": 0.85, "benchmark_acc": 0.85}
    }
    recs = ["Optimal Health: Crop shows high vegetation reflection.", "Maintain current watering schedules."]
    
    generate_crop_report(
        pdf_out,
        test_img_path,
        os.path.join(processed_dir, "index_ndvi.png"),
        "Healthy",
        0.88,
        10.5,
        class_results_pdf,
        quant_results_pdf,
        recs
    )
    
    assert os.path.exists(pdf_out), "PDF Report generation failed!"
    print("✓ PDF Report successfully generated at:", pdf_out)
    
    print("\n=== ALL BACKEND PIPELINE TESTS PASSED SUCCESSFUL ===")

if __name__ == "__main__":
    run_tests()
