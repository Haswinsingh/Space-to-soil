# Implementation Plan - QuantumCrop AI: Eyes in the Sky

QuantumCrop AI is a Quantum Machine Learning platform designed to analyze remote sensing agricultural data. It compares Classical Machine Learning models (Random Forest, SVM, XGBoost, CNN) with Quantum Machine Learning models (QSVM, Variational Quantum Classifier (VQC), Hybrid QCNN) for classifying crop health, predicting yield, and detecting crop stress.

---

## User Review Required

> [!IMPORTANT]
> **Quantum Machine Learning Environment & Simulator Choice**
> - The QML pipeline will implement **Qiskit** and **PennyLane** with local simulators (`default.qubit` for PennyLane and `qasm_simulator` for Qiskit). This ensures high performance and prevents credential friction (no API keys required for MVP execution, though config is provided for IBM Quantum).
> - Since QML models scale exponentially in training time with qubits, we will downsample features using PCA / Autoencoders and train QML models on a reduced feature space (e.g., 4 to 8 features/qubits) for real-time dashboard execution.

> [!WARNING]
> **MongoDB Database Fallback**
> - To make the platform "plug-and-play" for evaluation, we will implement a dual-mode database engine. It will attempt to connect to MongoDB (via `motor`), but fall back gracefully to a file-based JSON database (`JsonDB`) if no MongoDB instance is running, ensuring the backend runs seamlessly on any machine.

---

## Proposed Architecture & Directory Structure

We will structure the project in the workspace root `d:\space to soil` as follows:

```
d:\space to soil/
├── frontend/                     # React.js (Vite + TypeScript)
│   ├── public/
│   └── src/
│       ├── components/           # UI components (Sidebar, Charts, Maps)
│       ├── pages/                # Landing, Auth, Dashboard, Predictors
│       ├── utils/                # API client, PDF generators
│       ├── App.tsx
│       ├── index.css             # Tailwind style configuration
│       └── main.tsx
└── backend/                      # FastAPI Backend
    ├── api/                      # Routing & Controllers
    │   ├── auth.py               # User registration and JWT session
    │   ├── processing.py         # Image upload & preprocessing endpoints
    │   ├── predictions.py        # Model inference and benchmarking
    │   └── admin.py              # User logs and system administration
    ├── preprocessing/            # Satellite image bands & VI calculation
    │   └── pipeline.py           # CV2 / Rasterio computations
    ├── models/                   # ML & QML Models
    │   ├── classical/            # Random Forest, XGBoost, CNN, SVM
    │   │   └── classifiers.py
    │   └── quantum/              # QSVM, VQC, Hybrid QCNN, Circuit Visualization
    │       └── classifiers.py
    ├── datasets/                 # Synthetic generator for EuroSAT / GeoTIFF
    │   └── generator.py
    ├── reports/                  # PDF Report Generator
    │   └── generator.py
    ├── utils/                    # Config, DB connections, fallback DB
    │   ├── config.py
    │   └── database.py
    ├── requirements.txt          # Python dependencies
    └── main.py                   # Entry point
```

---

## Proposed Changes

### Backend Component (`backend/`)

#### [NEW] [requirements.txt](file:///d:/space/to/soil/backend/requirements.txt)
- Define python libraries: `fastapi`, `uvicorn`, `pymongo`, `motor`, `pyjwt`, `passlib[bcrypt]`, `python-multipart`, `opencv-python-headless`, `numpy`, `pandas`, `scikit-learn`, `qiskit`, `qiskit-machine-learning`, `pennylane`, `matplotlib`, `reportlab`, `rasterio` (optional with cv2 fallback if binary compile issues on Windows), `python-dotenv`.

#### [NEW] [database.py](file:///d:/space/to/soil/backend/utils/database.py)
- Establish connection to MongoDB. Include a local SQLite or JSON file fallback if connection fails.

#### [NEW] [pipeline.py](file:///d:/space/to/soil/backend/preprocessing/pipeline.py)
- Code to load satellite GeoTIFF / PNG / JPG.
- Standardize sizing and perform noise reduction / normalization.
- Calculate Vegetation Indices:
  - **NDVI** = (NIR - Red) / (NIR + Red)
  - **EVI** = 2.5 * ((NIR - Red) / (NIR + 6 * Red - 7.5 * Blue + 1))
  - **SAVI** = ((NIR - Red) / (NIR + Red + 0.5)) * 1.5
  - **NDWI** = (Green - NIR) / (Green + NIR)
  - **Chlorophyll Index (CI)** = (NIR / Green) - 1
- Export band images and combined index heatmaps.

#### [NEW] [classical/classifiers.py](file:///d:/space/to/soil/backend/models/classical/classifiers.py)
- Train & execute Random Forest, SVM, XGBoost, and a simple 2D CNN (using PyTorch or TensorFlow, or scikit-learn MLP if TensorFlow takes too long to load).
- Standard scikit-learn pipelines with performance evaluations (Accuracy, F1, Precision, Recall).

#### [NEW] [quantum/classifiers.py](file:///d:/space/to/soil/backend/models/quantum/classifiers.py)
- Implement **QSVM** using Qiskit's `ZZFeatureMap` and `QuantumKernel`.
- Implement **VQC** (Variational Quantum Classifier) using PennyLane or Qiskit.
- Implement **Hybrid QCNN** (Quantum-Classical CNN) using PennyLane layer wrapped in a PyTorch module.
- Auto-generate Quantum Circuit diagrams using Matplotlib.
- Measure training time, inference time, and memory footprint.

#### [NEW] [reports/generator.py](file:///d:/space/to/soil/backend/reports/generator.py)
- Create a PDF generator using `reportlab`. It will gather predictions, classical vs quantum benchmark statistics, and insert generated images (NDVI maps, confusion matrices).

#### [NEW] [main.py](file:///d:/space/to/soil/backend/main.py)
- FastAPI routes setup, CORS middleware, JWT auth routes, processing routes, predictions routes.

---

### Frontend Component (`frontend/`)

#### [NEW] [package.json](file:///d:/space/to/soil/frontend/package.json)
- Define React Vite project with tailwind, lucide-react, framer-motion, chart.js, react-chartjs-2, react-leaflet, react-router-dom, axios.

#### [NEW] [tailwind.config.js](file:///d:/space/to/soil/frontend/tailwind.config.js)
- Dark mode theme with cyan/emerald/blue glassmorphism values.

#### [NEW] [pages/LandingPage.tsx](file:///d:/space/to/soil/frontend/src/pages/LandingPage.tsx)
- Immersive UI with particle-animated background, floating quantum circuits, and feature callouts.

#### [NEW] [pages/Dashboard.tsx](file:///d:/space/to/soil/frontend/src/pages/Dashboard.tsx)
- Professional UI featuring sidebar, responsive stats, interactive Leaflet field maps, drag-and-drop satellite imagery upload, and vegetation index inspection.

#### [NEW] [pages/Comparison.tsx](file:///d:/space/to/soil/frontend/src/pages/Comparison.tsx)
- Side-by-side graphs and performance parameters contrasting QML vs Classical ML. Shows accuracy-vs-data-size (quantum advantage demonstration).

#### [NEW] [components/QuantumCircuit.tsx](file:///d:/space/to/soil/frontend/src/components/QuantumCircuit.tsx)
- Visually renders the quantum feature maps, ansatz layers, and measurement symbols dynamically or displays backend Qiskit-rendered circuits.

---

## Verification Plan

### Automated Tests & Benchmarks
- We will write a validation script `backend/test_pipeline.py` to:
  1. Generate synthetic multispectral GeoTIFF files (4 bands: Red, Green, Blue, NIR).
  2. Run the preprocessing pipeline and verify correct index bounds (e.g. NDVI in [-1, 1]).
  3. Validate Classical and Quantum classifier pipelines.
  4. Measure accuracy and speed benchmarks.

### Manual Verification
- Start backend: `python -m uvicorn backend.main:app --reload`
- Start frontend: `npm run dev`
- Load the app in browser, complete mock user login, upload synthetic imagery, inspect outputs, compare models, and export PDF.
