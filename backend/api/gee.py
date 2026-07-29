import os
import logging
import datetime
import uuid
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from backend.api.auth import get_current_user
from backend.utils import config
from backend.utils.database import get_collection, serialize_document
from backend.services.gee_service import analyze_field
from backend.models.classical.classifiers import ClassicalPipeline
from backend.models.quantum.classifiers import QuantumPipeline

logger = logging.getLogger("uvicorn.error")

router = APIRouter(prefix="/gee", tags=["Google Earth Engine"])

class AnalyzeRequest(BaseModel):
    latitude: float
    longitude: float
    polygon: Optional[List[List[float]]] = None
    geojson: Optional[Dict[str, Any]] = None
    use_landsat: Optional[bool] = False

@router.get("/map")
async def get_gee_map(lat: float, lng: float, use_landsat: bool = False):
    try:
        result = analyze_field(lat, lng, use_landsat=use_landsat)
        return {"success": True, "tile_urls": result["tile_urls"]}
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/ndvi")
async def get_gee_ndvi(lat: float, lng: float, use_landsat: bool = False):
    try:
        result = analyze_field(lat, lng, use_landsat=use_landsat)
        return {"success": True, "ndvi_url": result["tile_urls"]["ndvi"]}
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/evi")
async def get_gee_evi(lat: float, lng: float, use_landsat: bool = False):
    try:
        result = analyze_field(lat, lng, use_landsat=use_landsat)
        return {"success": True, "evi_url": result["tile_urls"]["evi"]}
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/savi")
async def get_gee_savi(lat: float, lng: float, use_landsat: bool = False):
    try:
        result = analyze_field(lat, lng, use_landsat=use_landsat)
        return {"success": True, "savi_url": result["tile_urls"]["savi"]}
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/ndwi")
async def get_gee_ndwi(lat: float, lng: float, use_landsat: bool = False):
    try:
        result = analyze_field(lat, lng, use_landsat=use_landsat)
        return {"success": True, "ndwi_url": result["tile_urls"]["ndwi"]}
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze")
async def post_gee_analyze(req: AnalyzeRequest, current_user: dict = Depends(get_current_user)):
    try:
        # Determine polygon coords
        coords = None
        if req.geojson and "geometry" in req.geojson:
            geom = req.geojson["geometry"]
            if geom and "coordinates" in geom:
                coords = geom["coordinates"][0]
        elif req.polygon:
            coords = req.polygon

        # Run Earth Engine index computation & telemetry extraction
        gee_res = analyze_field(req.latitude, req.longitude, coords, req.use_landsat)
        
        file_id = gee_res["file_id"]
        features = gee_res["features"]
        raw_path = gee_res["image_path"]

        # Run Classical & Quantum predictions (Task 14 / 15)
        try:
            classical = ClassicalPipeline(config.BASE_DIR)
            classical_res = classical.predict(features, image_path=raw_path)
        except Exception as clf_err:
            logger.exception(clf_err)
            raise HTTPException(status_code=500, detail=f"Classical prediction model error: {str(clf_err)}")

        try:
            quantum = QuantumPipeline(config.BASE_DIR)
            quantum_res = quantum.predict(features, image_path=raw_path)
        except Exception as q_err:
            logger.exception(q_err)
            raise HTTPException(status_code=500, detail=f"Quantum prediction model error: {str(q_err)}")

        # Extract prediction values
        health_status = classical_res["health_predictions"]["svm"]["class_name"]
        confidence = classical_res["health_predictions"]["svm"]["confidence"]
        predicted_yield = classical_res["yield_t_ha"]
        quantum_class = quantum_res.get("qsvm", {}).get("class_name", "")
        quantum_confidence = quantum_res.get("qsvm", {}).get("confidence", 1.0)
        
        svm_probs = classical_res["health_predictions"]["svm"].get("probabilities", [])
        disease_prob = svm_probs[3] if len(svm_probs) > 3 else 0.0

        ndvi_mean = features.get("ndvi_mean", 0.5)
        evi_mean = features.get("evi_mean", 0.5)
        savi_mean = features.get("savi_mean", 0.5)
        ndwi_mean = features.get("ndwi_mean", 0.0)
        ci_mean = features.get("ci_mean", 3.0)

        # Generate Recommendations (Task 15)
        recommendations = []

        # Rule 1: NDVI Vegetation Density
        if ndvi_mean > 0.7:
            recommendations.append({
                "severity": "low",
                "title": "Healthy Vegetation",
                "message": "Excellent crop density and chlorophyll activity detected. Maintain current practices."
            })
        elif 0.4 <= ndvi_mean <= 0.7:
            recommendations.append({
                "severity": "medium",
                "title": "Moderate Vegetation",
                "message": "Crop density is moderate. Monitor growth and vegetative development closely."
            })
        else:
            recommendations.append({
                "severity": "high",
                "title": "Low Vegetation",
                "message": "Warning: Very low vegetation density. Indicates potential crop stress or sparse growth."
            })

        # Rule 2: Chlorophyll / Nitrogen Deficiency
        if ci_mean < 1.5:
            recommendations.append({
                "severity": "high",
                "title": "Nitrogen Deficiency",
                "message": "Extremely low Chlorophyll Index detected. Apply urea or targeted nitrogen fertilizers within 5 days."
            })
        elif ci_mean < 2.5:
            recommendations.append({
                "severity": "medium",
                "title": "Nitrogen Warning",
                "message": "Chlorophyll levels are slightly depressed. Consider early nitrogen fertilizer application."
            })

        # Rule 3: NDWI / Irrigation
        if ndwi_mean < -0.2:
            recommendations.append({
                "severity": "high",
                "title": "Irrigation Recommendation",
                "message": "Critical water stress detected. Soil moisture is severely depleted; increase irrigation frequency immediately."
            })
        elif ndwi_mean < -0.05:
            recommendations.append({
                "severity": "medium",
                "title": "Moisture Stress",
                "message": "Moderate moisture stress detected. Ensure regular watering schedules are maintained."
            })

        # Rule 4: Disease
        if disease_prob > 0.6:
            recommendations.append({
                "severity": "high",
                "title": "Disease Inspection",
                "message": "Warning: High disease probability detected. Arrange an immediate field check for fungal or pest damage."
            })
        elif disease_prob > 0.3:
            recommendations.append({
                "severity": "medium",
                "title": "Foliar Disease Risk",
                "message": "Elevated disease probability. Inspect patchy vegetation sections for early pathogen indicators."
            })

        # Rule 5: Yield Improvement
        if predicted_yield < 6.0:
            recommendations.append({
                "severity": "high",
                "title": "Yield Improvement",
                "message": "Yield forecast is below target threshold. Apply soil conditioners and optimize macronutrients."
            })

        # Rule 6: Confidence
        if confidence < 0.6:
            recommendations.append({
                "severity": "medium",
                "title": "Data Quality Warning",
                "message": "Prediction confidence is low. Recommend collecting more high-resolution drone imagery."
            })

        # Rule 7: Model Disagreement
        is_quantum_veg = quantum_class in ["AnnualCrop", "Forest", "HerbaceousVegetation", "PermanentCrop", "Pasture"]
        if not is_quantum_veg:
            recommendations.append({
                "severity": "medium",
                "title": "Model Disagreement",
                "message": f"Quantum model classifies area as {quantum_class} while Classical model predicts crop health state {health_status}. Recommend manual land use inspection."
            })

        # Rule 8: If no alerts or recommendations, add default optimal health
        if not recommendations:
            recommendations.append({
                "severity": "low",
                "title": "Optimal Health",
                "message": "Crop shows high vegetation reflection and moisture. Maintain current irrigation and fertilizer schedules."
            })

        # Logging (Task 21)
        print("\n=== GOOGLE EARTH ENGINE PREDICTION COMPLETED ===")
        print(f"NDVI Mean: {ndvi_mean:.4f}")
        print(f"EVI Mean: {evi_mean:.4f}")
        print(f"NDWI Mean: {ndwi_mean:.4f}")
        print(f"CI Mean: {ci_mean:.4f}")
        print(f"Disease score: {disease_prob:.4f}")
        print(f"Yield: {predicted_yield:.2f}")
        print(f"Generated recommendations: {recommendations}")
        print("================================================\n")

        prediction_results = {
            "crop_health": health_status,
            "confidence": confidence,
            "yield_t_ha": predicted_yield,
            "recommendations": recommendations,
            "classical_details": classical_res["health_predictions"],
            "quantum_details": quantum_res
        }

        # Save to MongoDB uploads_history collection
        history_col = get_collection("uploads_history")
        raw_url = gee_res["image_url"]
        history_item = {
            "file_id": file_id,
            "filename": f"GEE_{file_id}.png",
            "uploaded_by": current_user["email"],
            "file_size_kb": 0.0,
            "resolution": "Sentinel-2 Multi-Spectral (GEE)",
            "bands_count": 10,
            "source_type": "gee",
            "features": features,
            "paths": {
                "raw": raw_url,
                "processed": {
                    "ndvi": raw_url, # Fallback to true-color download if processed not locally computed
                    "evi": raw_url,
                    "savi": raw_url,
                    "ndwi": raw_url,
                    "ci": raw_url
                }
            },
            "predictions": prediction_results,
            "created_at": datetime.datetime.utcnow().isoformat()
        }
        history_col.insert_one(serialize_document(history_item))

        # Return full payload
        return {
            "success": True,
            "file_id": file_id,
            "area_ha": gee_res["area_ha"],
            "cloud_cover": gee_res["cloud_cover"],
            "acquisition_date": gee_res["acquisition_date"],
            "indices": features,
            "extra_indices": gee_res["extra_indices"],
            "tile_urls": gee_res["tile_urls"],
            "prediction": health_status,
            "crop_health": health_status,
            "confidence": confidence,
            "yield_t_ha": predicted_yield,
            "disease_probability": disease_prob,
            "quantum_confidence": quantum_confidence,
            "predicted_crop": quantum_class,
            "recommendations": recommendations,
            "image_url": raw_url
        }

    except ValueError as val_err:
        logger.error(f"Validation error: {val_err}")
        raise HTTPException(status_code=400, detail=str(val_err))
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))
