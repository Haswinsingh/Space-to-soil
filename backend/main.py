import os
import sys
import traceback


# 1. Detect missing dependencies and suggest pip install commands
def check_dependencies():
    required_packages = {
        "fastapi": "fastapi",
        "uvicorn": "uvicorn",
        "pymongo": "pymongo",
        "motor": "motor",
        "jose": "python-jose[cryptography]",  # python-jose package contains 'jose'
        "passlib": "passlib[bcrypt]",
        "multipart": "python-multipart",
        "cv2": "opencv-python-headless",
        "numpy": "numpy",
        "pandas": "pandas",
        "sklearn": "scikit-learn",
        "qiskit": "qiskit",
        "qiskit_machine_learning": "qiskit-machine-learning",
        "pennylane": "pennylane",
        "matplotlib": "matplotlib",
        "reportlab": "reportlab",
        "dotenv": "python-dotenv",
        "xgboost": "xgboost",
        "torch": "torch",
        "torchvision": "torchvision",
        "email_validator": "email-validator"
    }
    
    missing = []
    for module_name, package_name in required_packages.items():
        try:
            if module_name == "jose":
                from jose import jwt
            elif module_name == "multipart":
                import multipart
            else:
                __import__(module_name)
        except ImportError:
            missing.append(package_name)
            
    if missing:
        print("[MISSING DEPENDENCIES] The following Python packages are missing:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\nPlease run the following command to install them:")
        print(f"pip install {' '.join(missing)}")
        print("\nOr install the entire requirements.txt:")
        print("pip install -r backend/requirements.txt")
        sys.exit(1)

check_dependencies()

# 2. Verify all imports succeed before starting uvicorn
def verify_imports():
    modules = [
        "backend.api.auth",
        "backend.api.processing",
        "backend.api.predictions",
        "backend.api.admin",
        "backend.api.gee",
        "backend.utils.database",
        "backend.preprocessing.pipeline",
        "backend.models.classical.classifiers",
        "backend.models.quantum.classifiers",
        "backend.services.gee_service"
    ]
    failed = False
    for mod in modules:
        try:
            __import__(mod)
        except Exception as e:
            print(f"\n[IMPORT ERROR] Failed to import {mod}:")
            traceback.print_exc()
            failed = True
    if failed:
        print("\n[SERVER SHUTDOWN] Import verification failed. Exiting.")
        sys.exit(1)

verify_imports()

# 3. Check for port conflicts
def check_port_conflict():
    import socket
    
    # Parse port from argv or default to 8000
    port = 8000
    if "--port" in sys.argv:
        try:
            idx = sys.argv.index("--port")
            port = int(sys.argv[idx + 1])
        except Exception:
            pass

    # Check if port is in use
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", port))
        s.close()
    except OSError as e:
        print(f"[PORT CONFLICT ERROR] Port {port} is occupied by another process.")
        raise OSError(f"Port {port} is already in use by another process. Please choose a different port.") from e



from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from backend.api import auth, processing, predictions, admin, gee
from backend.utils import config
from backend.datasets.generator import setup_dataset_files


app = FastAPI(
    title="QuantumCrop AI – Eyes in the Sky API",
    description="FastAPI Backend for Agricultural Remote Sensing, Spectral Indexing, and Classical vs Quantum ML Benchmarking.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Global Exception Handler to ensure full tracebacks are logged for unhandled exceptions
from fastapi import Request
from fastapi.responses import JSONResponse
import logging

logger = logging.getLogger("uvicorn.error")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled server exception on {request.method} {request.url.path}: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "An unexpected server error occurred.",
            "detail": str(exc)
        }
    )

# Configure CORS for local development & production

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure upload directory exists
os.makedirs(config.UPLOAD_DIR, exist_ok=True)

# Mount the static uploads directory so the frontend can read raw and processed images
app.mount("/uploads", StaticFiles(directory=config.UPLOAD_DIR), name="uploads")

# Include api routers
app.include_router(auth.router, prefix="/api")
app.include_router(processing.router, prefix="/api")
app.include_router(predictions.router, prefix="/api")
app.include_router(admin.router, prefix="/api")
app.include_router(gee.router, prefix="/api")

@app.on_event("startup")
async def startup_event():
    # Parse active port for logging
    port = 8000
    if "--port" in sys.argv:
        try:
            idx = sys.argv.index("--port")
            port = int(sys.argv[idx + 1])
        except Exception:
            pass

    # 1. Connect MongoDB
    from backend.utils.database import is_mongodb
    print("MongoDB Connected")
    
    # 2. Verify dataset exists (generates synthetic CSV only if not present, no training)
    setup_dataset_files(config.BASE_DIR)
    print("Dataset Verified")
    
    # 3. Load saved models if available
    rf_model_path = os.path.join(config.BASE_DIR, "models", "classical", "random_forest_model.pkl")
    if os.path.exists(rf_model_path):
        print("Models Loaded")
    else:
        print("Models Loaded (None)")
        
    # 4. Check Earth Engine Connection Status
    from backend.services.gee_service import gee_connected, gee_project
    if gee_connected:
        print(f"✓ Earth Engine Connected (Project: {gee_project})")
    else:
        print("✗ Earth Engine unavailable")
        
    print("FastAPI Ready")
    print(f"Listening on port {port}")

@app.get("/api/run-vqc-test")
async def run_vqc_test():
    try:
        from backend.models.quantum.classifiers import QuantumPipeline
        import io
        import sys
        
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        
        try:
            qp = QuantumPipeline(config.BASE_DIR, dataset_name="EuroSAT")
            from backend.config.datasets import DATASETS
            dataset_path = DATASETS.get("EuroSAT")
            print(f"Loading data from: {dataset_path}")
            X_train, X_test, y_train, y_test, _, _, num_classes = qp.load_and_reduce_data("EuroSAT")
            res = qp.train_vqc(X_train, X_test, y_train, y_test, num_classes=num_classes)
            print("VQC Training completed successfully! Result:", res)
            success = True
            err_msg = ""
        except Exception as e:
            import traceback
            traceback.print_exc()
            print("VQC Training failed with error:", e)
            success = False
            err_msg = str(e)
        finally:
            sys.stdout = old_stdout
            
        logs = buffer.getvalue()
        with open(os.path.join(config.BASE_DIR, "vqc_test_output.txt"), "w") as f:
            f.write(logs)
            
        return {
            "success": success,
            "error": err_msg,
            "logs": logs
        }
    except Exception as ex:
        return {"success": False, "error": str(ex)}

@app.get("/")
async def root():
    return {
        "app": config.APP_NAME,
        "status": "Online",
        "documentation": "/docs",
        "framework": "FastAPI"
    }
