import logging
from fastapi import APIRouter, Depends, HTTPException
from backend.api.auth import get_current_user
from backend.utils.database import get_collection
from backend.utils.mongo import serialize_documents, make_response

router = APIRouter(prefix="/admin", tags=["Admin Operations"])
logger = logging.getLogger("AdminAPI")

@router.get("/stats")
async def get_system_stats(current_user: dict = Depends(get_current_user)):
    try:
        # 1. Total users
        users_col = get_collection("users")
        users_count = users_col.count_documents({})
        
        # 2. Total processed files
        history_col = get_collection("uploads_history")
        uploads_count = history_col.count_documents({})
        
        history_list = list(history_col.find())
        
        # Calculate crop stress statistics
        stress_stats = {
            "Healthy": 0,
            "Water Stress": 0,
            "Nitrogen Deficiency": 0,
            "Disease": 0,
            "Severe Stress": 0
        }
        
        for record in history_list:
            if "predictions" in record and record["predictions"]:
                health = record["predictions"].get("crop_health", "Healthy")
                if health in stress_stats:
                    stress_stats[health] += 1
                    
        # 3. Model benchmarks count
        benchmarks_col = get_collection("benchmarks")
        benchmarks_count = benchmarks_col.count_documents({})
        
        # Get recent 10 uploads and serialize them
        recent = history_list[-10:] if history_list else []
        recent_serialized = serialize_documents(recent)
        
        data = {
            "users_count": users_count,
            "uploads_count": uploads_count,
            "benchmarks_count": benchmarks_count,
            "crop_health_distribution": stress_stats,
            "recent_uploads": recent_serialized
        }
        
        return make_response(data=data)
    except Exception as e:
        logger.exception(e)
        return make_response(success=False, message=str(e))
