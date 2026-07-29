import os
import shutil
import uuid
import logging
import zipfile
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from backend.api.auth import get_current_user
from backend.preprocessing.pipeline import run_preprocessing_pipeline
from backend.utils import config
from backend.utils.database import get_collection
from backend.utils.mongo import serialize_document, make_response
from PIL import Image as PILImage

router = APIRouter(prefix="/processing", tags=["Image Processing"])
logger = logging.getLogger("ProcessingAPI")

@router.post("/upload")
async def upload_and_process_image(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    try:
        # Verify file extensions (expanded to support zip and csv dataset uploads)
        allowed_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".geotiff", ".zip", ".csv"}
        _, ext = os.path.splitext(file.filename.lower())
        if ext not in allowed_exts:
            raise HTTPException(status_code=400, detail=f"Unsupported file format {ext}")
            
        # Generate unique filenames
        file_id = str(uuid.uuid4())
        filename = f"{file_id}{ext}"
        raw_path = os.path.join(config.UPLOAD_DIR, filename)
        
        # Save uploaded file
        try:
            with open(raw_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {str(e)}")
            
        # 1. Handle ZIP dataset archives
        if ext == ".zip":
            try:
                with zipfile.ZipFile(raw_path, 'r') as zip_ref:
                    zip_ref.extractall(config.DATASET_IMAGE_PATH)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to extract ZIP dataset: {str(e)}")
            
            resp_data = {
                "message": f"Successfully extracted dataset ZIP archive to {config.DATASET_IMAGE_PATH}",
                "filename": file.filename,
                "dataset_type": "image",
                "file_id": file_id
            }
            return make_response(data=resp_data)
            
        # 2. Handle CSV dataset files
        elif ext == ".csv":
            csv_dest = os.path.join(config.DATASET_CSV_PATH, file.filename)
            try:
                shutil.copyfile(raw_path, csv_dest)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to save CSV dataset: {str(e)}")
                
            resp_data = {
                "message": f"Successfully saved CSV dataset file to {csv_dest}",
                "filename": file.filename,
                "dataset_type": "csv",
                "file_id": file_id
            }
            return make_response(data=resp_data)
            
        # 3. Handle normal satellite / agricultural imagery
        file_size_kb = round(os.path.getsize(raw_path) / 1024, 2)
        resolution = "Unknown"
        bands_count = 3
        
        # Try reading dimensions
        try:
            if ext in [".tif", ".tiff", ".geotiff"]:
                try:
                    import rasterio
                    with rasterio.open(raw_path) as src:
                        resolution = f"{src.width} x {src.height}"
                        bands_count = src.count
                except Exception:
                    with PILImage.open(raw_path) as img:
                        resolution = f"{img.width} x {img.height}"
                        bands_count = len(img.getbands())
            else:
                with PILImage.open(raw_path) as img:
                    resolution = f"{img.width} x {img.height}"
                    bands_count = len(img.getbands())
        except Exception:
            pass
            
        # Run the image processing pipeline
        processed_dir_name = f"processed_{file_id}"
        processed_abs_path = os.path.join(config.UPLOAD_DIR, processed_dir_name)
        os.makedirs(processed_abs_path, exist_ok=True)
        
        try:
            pipeline_results = run_preprocessing_pipeline(raw_path, processed_abs_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error in image processing pipeline: {str(e)}")
            
        web_processed_paths = {}
        for name, relative_path in pipeline_results["visualizations"].items():
            file_basename = os.path.basename(relative_path)
            web_processed_paths[name] = f"/uploads/{processed_dir_name}/{file_basename}"
            
        # Log upload history in database
        history_item = {
            "file_id": file_id,
            "filename": file.filename,
            "uploaded_by": current_user["email"],
            "file_size_kb": file_size_kb,
            "resolution": resolution,
            "bands_count": bands_count,
            "source_type": pipeline_results["source"],
            "features": pipeline_results["features"],
            "paths": {
                "raw": f"/uploads/{filename}",
                "processed": web_processed_paths
            },
            "created_at": str(uuid.uuid4())
        }
        
        serialized_history = serialize_document(history_item)
        get_collection("uploads_history").insert_one(serialized_history)
        
        response_data = {
            "file_id": file_id,
            "filename": file.filename,
            "metadata": {
                "size_kb": file_size_kb,
                "resolution": resolution,
                "bands": bands_count,
                "type": pipeline_results["source"]
            },
            "features": pipeline_results["features"],
            "paths": {
                "raw": f"/uploads/{filename}",
                "processed": web_processed_paths
            }
        }
        
        serialized_response = serialize_document(response_data)
        return make_response(data=serialized_response)
    except HTTPException as http_err:
        return make_response(success=False, message=http_err.detail)
    except Exception as e:
        logger.exception(e)
        return make_response(success=False, message=str(e))
