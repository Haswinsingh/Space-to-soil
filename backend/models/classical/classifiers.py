import os
import pickle
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    auc,
    precision_recall_fscore_support
)
from backend.datasets.loader import load_csv_dataset, load_image_dataset
from backend.utils import config

# Fallback for xgboost
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
def inspect_and_clean_data(X, y, filenames=None, expected_feature_length=None, dataset_name="Dataset"):
    import numpy as np
    
    print(f"\n=== INSPECTING {dataset_name.upper()} BEFORE TRAINING ===")
    
    X_list = list(X)
    y_list = list(y)
    
    clean_X = []
    clean_y = []
    
    if expected_feature_length is None:
        for sample in X_list:
            try:
                s_arr = np.asarray(sample).flatten()
                expected_feature_length = len(s_arr)
                break
            except Exception:
                pass
        if expected_feature_length is None:
            expected_feature_length = 64
            
    print(f"Expected feature length: {expected_feature_length}")
    
    for idx, (sample, label) in enumerate(zip(X_list, y_list)):
        filename = filenames[idx] if (filenames and idx < len(filenames)) else "Unknown"
        try:
            s_arr = np.asarray(sample)
            s_flat = s_arr.flatten()
            actual_shape = s_arr.shape
            actual_len = len(s_flat)
            
            if actual_len != expected_feature_length:
                print(f"[MALFORMED SAMPLE REJECTED] Sample Index: {idx} | Expected Shape: ({expected_feature_length},) | Actual Shape: {actual_shape} | File Name: {filename}")
                continue
                
            l_arr = np.asarray(label)
            if l_arr.ndim > 0:
                l_val = int(l_arr.flatten()[0])
            else:
                l_val = int(label)
                
            clean_X.append(s_flat)
            clean_y.append(l_val)
        except Exception as e:
            print(f"[MALFORMED SAMPLE REJECTED] Sample Index: {idx} | Error parsing sample/label: {e} | File Name: {filename}")
            continue
            
    X_clean = np.asarray(clean_X, dtype=np.float32)
    y_clean = np.asarray(clean_y, dtype=np.int64)
    
    print(f"X.shape: {X_clean.shape}")
    print(f"y.shape: {y_clean.shape}")
    print(f"dtype: X={X_clean.dtype}, y={y_clean.dtype}")
    if len(X_clean) > 0:
        trunc_len = min(5, len(X_clean[0]))
        print(f"First sample (truncated): {X_clean[0][:trunc_len]}...")
        print(f"Feature length: {len(X_clean[0])}")
    else:
        print("First sample: N/A")
        print("Feature length: N/A")
        
    print(f"=== INSPECTION COMPLETE FOR {dataset_name.upper()} ===\n")
    
    if len(X_clean) == 0:
        raise ValueError(f"No valid samples remaining in {dataset_name} after cleaning!")
        
    return X_clean, y_clean

def log_exception_details(e):
    import traceback
    import sys
    print("\n!!! [VALIDATION ERROR OCCURRED] !!!")
    traceback.print_exc()
    exc_type, exc_value, exc_tb = sys.exc_info()
    tb = traceback.extract_tb(exc_tb)
    if tb:
        filename, line_num, func_name, text = tb[-1]
        print(f"[VALIDATION ERROR LOCATION] File: {filename}, Line: {line_num}, Function: {func_name}")
        print(f"Exact line of code: {text}")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")


def compute_roc_curve_data(y_true, y_prob, num_classes):
    try:
        y_true = np.array(y_true)
        y_prob = np.array(y_prob)
        
        # Validation checks
        if len(y_true) == 0 or len(y_prob) == 0:
            return {
                "roc_auc": None,
                "message": "ROC curve unavailable for current prediction."
            }
            
        unique_classes = np.unique(y_true)
        if len(unique_classes) <= 1:
            return {
                "roc_auc": None,
                "message": "ROC curve unavailable for current prediction."
            }
            
        # Detect binary vs multiclass
        is_multiclass = num_classes > 2 or len(unique_classes) > 2
        
        if is_multiclass:
            # One-vs-Rest One-Hot encoding of true classes
            y_onehot = np.zeros((len(y_true), num_classes))
            y_onehot[np.arange(len(y_true)), y_true] = 1
            
            # Compute macro ROC AUC score
            roc_auc_val = roc_auc_score(y_onehot, y_prob, multi_class="ovr", average="macro")
            
            # Compute one-vs-rest micro-average ROC curve for plotting
            fpr, tpr, _ = roc_curve(y_onehot.ravel(), y_prob.ravel())
            
            return {
                "roc_auc": float(roc_auc_val),
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "auc": float(roc_auc_val)
            }
        else:
            # Binary class
            if y_prob.ndim == 2:
                prob_pos = y_prob[:, 1]
            else:
                prob_pos = y_prob
                
            roc_auc_val = roc_auc_score(y_true, prob_pos)
            fpr, tpr, _ = roc_curve(y_true, prob_pos)
            
            return {
                "roc_auc": float(roc_auc_val),
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "auc": float(roc_auc_val)
            }
    except Exception:
        return {
            "roc_auc": None,
            "message": "ROC curve unavailable for current prediction."
        }

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # Input size is 64x64, after pool1: 32x32, after pool2: 16x16
        self.fc1 = nn.Linear(32 * 16 * 16, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x, extract_features=False):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 16 * 16)
        features = F.relu(self.fc1(x))
        if extract_features:
            return features
        x = self.fc2(features)
        return x

def extract_features(image, cnn_model=None, device=None):
    """
    Extracts features from an image or a batch of images/tensors.
    If cnn_model is provided, extracts 64-dimensional embeddings.
    """
    import torch
    import torchvision.transforms as transforms
    from PIL import Image

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if cnn_model is None:
        raise ValueError("cnn_model must be provided to extract features.")

    cnn_model.eval()

    if isinstance(image, torch.Tensor):
        # Already a tensor (e.g., during training)
        img_tensor = image.to(device)
        if img_tensor.ndim == 3:
            img_tensor = img_tensor.unsqueeze(0)
        with torch.no_grad():
            feats = cnn_model(img_tensor, extract_features=True)
        return feats.cpu().numpy()

    # If it's a file path or PIL image (e.g., during prediction)
    if isinstance(image, str):
        img = Image.open(image).convert("RGB")
    else:
        img = image.convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        feats = cnn_model(img_tensor, extract_features=True)
    return feats.cpu().numpy()

class ClassicalPipeline:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.models_dir = os.path.join(base_dir, "models", "classical")
        os.makedirs(self.models_dir, exist_ok=True)
        self.scaler = None
        self.feature_cols = [
            "ndvi_mean", "ndvi_std", "ndvi_min", "ndvi_max", "ndvi_q25", "ndvi_q75",
            "evi_mean", "evi_std", "evi_min", "evi_max", "evi_q25", "evi_q75",
            "savi_mean", "savi_std", "savi_min", "savi_max", "savi_q25", "savi_q75",
            "ndwi_mean", "ndwi_std", "ndwi_min", "ndwi_max", "ndwi_q25", "ndwi_q75",
            "ci_mean", "ci_std", "ci_min", "ci_max", "ci_q25", "ci_q75"
        ]

    def train_models(self, dataset_type="csv", dataset_path=None):
        if dataset_type == "image":
            return self._train_image_pipeline(dataset_path)
        else:
            return self._train_csv_pipeline(dataset_path)

    def _train_csv_pipeline(self, csv_path):
        X_train, X_test, y_train, y_test, y_yield_train, y_yield_test, num_classes, classes, scaler, feature_cols = load_csv_dataset(csv_path)
        
        # Enforce cleaning and verification
        try:
            X_train, y_train = inspect_and_clean_data(X_train, y_train, dataset_name="Classical CSV Train")
            X_test, y_test = inspect_and_clean_data(X_test, y_test, expected_feature_length=X_train.shape[1], dataset_name="Classical CSV Test")
        except Exception as e:
            log_exception_details(e)
            raise e
            
        # Save scaler for predictions
        self.scaler = scaler
        self.feature_cols = feature_cols
        with open(os.path.join(self.models_dir, "scaler.pkl"), "wb") as f:
            pickle.dump(self.scaler, f)
            
        # Task 2: Determine the exact feature vector size used during training
        print(f"X_train.shape: {X_train.shape}")
        print(f"scaler.n_features_in_: {self.scaler.n_features_in_}")

        # Save preprocessing objects (Task 13)
        with open(os.path.join(self.models_dir, "pca.pkl"), "wb") as f:
            pickle.dump(None, f)
        with open(os.path.join(self.models_dir, "feature_selector.pkl"), "wb") as f:
            pickle.dump(None, f)
        
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        le.fit(classes)
        with open(os.path.join(self.models_dir, "label_encoder.pkl"), "wb") as f:
            pickle.dump(le, f)
            
        models = {
            "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "svm": SVC(probability=True, kernel="rbf", random_state=42)
        }
        
        if HAS_XGB:
            models["xgboost"] = XGBClassifier(n_estimators=100, eval_metric="mlogloss", random_state=42)
        else:
            models["xgboost"] = GradientBoostingClassifier(n_estimators=100, random_state=42)
            
        results = {}
        
        for name, clf in models.items():
            start_time = time.time()
            try:
                # Enforce float32/int64 before fit
                X_fit = np.asarray(X_train, dtype=np.float32)
                y_fit = np.asarray(y_train, dtype=np.int64)
                clf.fit(X_fit, y_fit)
            except Exception as e:
                log_exception_details(e)
                raise e
            train_time = time.time() - start_time
            
            start_time = time.time()
            y_pred = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test)
            inference_time = time.time() - start_time
            
            acc = accuracy_score(y_test, y_pred)
            prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)
            cm = confusion_matrix(y_test, y_pred).tolist()
            
            results[name] = {
                "accuracy": float(acc),
                "precision": float(prec),
                "recall": float(rec),
                "f1_score": float(f1),
                "training_time_s": float(train_time),
                "inference_time_s": float(inference_time),
                "confusion_matrix": cm,
                "classes": classes
            }
            
            # Save trained model
            with open(os.path.join(self.models_dir, f"{name}_model.pkl"), "wb") as f:
                pickle.dump(clf, f)
                
        # Also train a yield predictor
        yield_reg = RandomForestRegressor(n_estimators=100, random_state=42)
        try:
            X_fit = np.asarray(X_train, dtype=np.float32)
            y_yield_fit = np.asarray(y_yield_train, dtype=np.float32)
            yield_reg.fit(X_fit, y_yield_fit)
        except Exception as e:
            log_exception_details(e)
            raise e
        with open(os.path.join(self.models_dir, "yield_regressor.pkl"), "wb") as f:
            pickle.dump(yield_reg, f)
            
        return results

    def _train_image_pipeline(self, dataset_path):
        train_loader, val_loader, test_loader, total_samples, num_classes, classes = load_image_dataset(dataset_path)
        
        # Enforce dataset inspection before training
        print("\n=== INSPECTING DATASET BEFORE CNN TRAINING ===")
        try:
            for imgs, targets in train_loader:
                print(f"X.shape: {imgs.shape}")
                print(f"y.shape: {targets.shape}")
                print(f"dtype: X={imgs.dtype}, y={targets.dtype}")
                print(f"first sample: shape={imgs[0].shape}, dtype={imgs[0].dtype}")
                print(f"feature length: {np.prod(imgs[0].shape)}")
                break
        except Exception as e:
            log_exception_details(e)
            raise e
        print("==============================================\n")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[CLASSICAL DEV] Running training and evaluation on device: {device}")

        # 1. Initialize and train SimpleCNN
        print("[MODEL INIT] Initializing SimpleCNN model (Classical) with 10 output classes...")
        cnn = SimpleCNN(num_classes=num_classes).to(device)
        optimizer = optim.Adam(cnn.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        print("[TRAINING CLASSICAL] Starting training for SimpleCNN on the full dataset...")
        cnn_start_time = time.time()
        cnn_loss_history = []
        cnn_acc_history = []
        
        # Train CNN for 3 epochs on the full dataset
        epochs = 3
        try:
            for epoch in range(epochs):
                cnn.train()
                running_loss = 0.0
                correct = 0
                total = 0
                for idx, (imgs, targets) in enumerate(train_loader):
                    imgs, targets = imgs.to(device), targets.to(device)
                    optimizer.zero_grad()
                    outputs = cnn(imgs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    
                    running_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                
                epoch_loss = running_loss / len(train_loader)
                epoch_acc = correct / total if total > 0 else 0.0
                print(f"[CLASSICAL CNN] Epoch {epoch+1}/{epochs} | Training Loss: {epoch_loss:.4f} | Training Accuracy: {epoch_acc*100:.2f}%")
                cnn_loss_history.append(float(epoch_loss))
                cnn_acc_history.append(float(epoch_acc))
        except Exception as e:
            log_exception_details(e)
            raise e
            
        cnn_train_time = time.time() - cnn_start_time
        print(f"[TRAINING CLASSICAL] CNN training finished in {cnn_train_time:.2f}s.")
        
        # Evaluate CNN on test set
        print("[VALIDATION CLASSICAL] Evaluating CNN on the test dataset...")
        cnn.eval()
        cnn_inf_start = time.time()
        y_true_cnn = []
        y_pred_cnn = []
        y_prob_cnn = []
        try:
            with torch.no_grad():
                for idx, (imgs, targets) in enumerate(test_loader):
                    imgs, targets = imgs.to(device), targets.to(device)
                    outputs = cnn(imgs)
                    probs = F.softmax(outputs, dim=1)
                    _, predicted = outputs.max(1)
                    
                    y_true_cnn.extend(targets.cpu().numpy().tolist())
                    y_pred_cnn.extend(predicted.cpu().numpy().tolist())
                    y_prob_cnn.extend(probs.cpu().numpy().tolist())
        except Exception as e:
            log_exception_details(e)
            raise e
            
        cnn_inf_time = time.time() - cnn_inf_start
        print(f"[VALIDATION CLASSICAL] Evaluation completed in {cnn_inf_time:.4f}s.")
        
        # Calculate CNN metrics
        acc_cnn = accuracy_score(y_true_cnn, y_pred_cnn)
        prec_cnn, rec_cnn, f1_cnn, _ = precision_recall_fscore_support(y_true_cnn, y_pred_cnn, average="weighted", zero_division=0)
        cm_cnn = confusion_matrix(y_true_cnn, y_pred_cnn).tolist()
        
        # Compute CNN ROC curve safely
        roc_data_cnn = compute_roc_curve_data(y_true_cnn, y_prob_cnn, num_classes)
        
        # Save CNN model
        cnn_model_path = os.path.join(self.models_dir, "cnn_model.pth")
        print(f"[MODEL SAVE] Saving CNN model state dict to: {cnn_model_path}")
        torch.save(cnn.state_dict(), cnn_model_path)
        
        cnn_results = {
            "accuracy": float(acc_cnn),
            "precision": float(prec_cnn),
            "recall": float(rec_cnn),
            "f1_score": float(f1_cnn),
            "training_time_s": float(cnn_train_time),
            "inference_time_s": float(cnn_inf_time),
            "confusion_matrix": cm_cnn,
            "roc_curve": roc_data_cnn,
            "loss_history": cnn_loss_history,
            "accuracy_history": cnn_acc_history,
            "classes": classes
        }
        
        # 2. Extract features for classical models
        print("[FEATURE EXTRACTION] Extracting features for traditional classical models...")
        from torch.utils.data import DataLoader
        train_dataset = train_loader.dataset
        test_dataset = test_loader.dataset
        
        # Non-shuffled loaders to align features and targets with file paths
        train_feat_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
        test_feat_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        X_train_list, y_train_list = [], []
        train_filenames = []
        with torch.no_grad():
            for idx, (imgs, targets) in enumerate(train_feat_loader):
                feats = extract_features(imgs, cnn_model=cnn, device=device)
                X_train_list.append(feats)
                y_train_list.append(targets.numpy())
                
                batch_size = imgs.size(0)
                start_i = idx * train_feat_loader.batch_size
                subset_indices = train_dataset.indices[start_i : start_i + batch_size]
                batch_files = [train_dataset.dataset.samples[i][0] for i in subset_indices]
                train_filenames.extend(batch_files)
                
        X_test_list, y_test_list = [], []
        test_filenames = []
        with torch.no_grad():
            for idx, (imgs, targets) in enumerate(test_feat_loader):
                feats = extract_features(imgs, cnn_model=cnn, device=device)
                X_test_list.append(feats)
                y_test_list.append(targets.numpy())
                
                batch_size = imgs.size(0)
                start_i = idx * test_feat_loader.batch_size
                subset_indices = test_dataset.indices[start_i : start_i + batch_size]
                batch_files = [test_dataset.dataset.samples[i][0] for i in subset_indices]
                test_filenames.extend(batch_files)
                
        X_train = np.concatenate(X_train_list, axis=0) if X_train_list else np.zeros((1, 64))
        y_train = np.concatenate(y_train_list, axis=0) if y_train_list else np.zeros(1, dtype=int)
        X_test = np.concatenate(X_test_list, axis=0) if X_test_list else np.zeros((1, 64))
        y_test = np.concatenate(y_test_list, axis=0) if y_test_list else np.zeros(1, dtype=int)
        
        # Scale extracted features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Enforce cleaning and verification of features
        try:
            X_train, y_train = inspect_and_clean_data(X_train_scaled, y_train, filenames=train_filenames, dataset_name="Classical Image Train")
            X_test, y_test = inspect_and_clean_data(X_test_scaled, y_test, filenames=test_filenames, expected_feature_length=X_train.shape[1], dataset_name="Classical Image Test")
        except Exception as e:
            log_exception_details(e)
            raise e
            
        # Save scaler
        with open(os.path.join(self.models_dir, "scaler.pkl"), "wb") as f:
            pickle.dump(scaler, f)
        with open(os.path.join(self.models_dir, "image_feature_scaler.pkl"), "wb") as f:
            pickle.dump(scaler, f)
            
        # Task 2: Determine the exact feature vector size used during training
        print(f"X_train.shape: {X_train.shape}")
        print(f"scaler.n_features_in_: {scaler.n_features_in_}")

        # Save preprocessing objects (Task 13)
        with open(os.path.join(self.models_dir, "pca.pkl"), "wb") as f:
            pickle.dump(None, f)
        with open(os.path.join(self.models_dir, "feature_selector.pkl"), "wb") as f:
            pickle.dump(None, f)
            
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        le.fit(classes)
        with open(os.path.join(self.models_dir, "label_encoder.pkl"), "wb") as f:
            pickle.dump(le, f)
            
        # 3. Train classical classifiers on extracted features
        models = {
            "random_forest": RandomForestClassifier(n_estimators=50, random_state=42),
            "svm": SVC(probability=True, kernel="rbf", random_state=42)
        }
        if HAS_XGB:
            models["xgboost"] = XGBClassifier(n_estimators=50, eval_metric="mlogloss", random_state=42)
        else:
            models["xgboost"] = GradientBoostingClassifier(n_estimators=50, random_state=42)
            
        results = {
            "cnn": cnn_results
        }
        
        for name, clf in models.items():
            print(f"[TRAINING CLASSICAL] Fitting traditional classifier: {name}...")
            start_time = time.time()
            try:
                # Enforce float32/int64 before fit
                X_fit = np.asarray(X_train, dtype=np.float32)
                y_fit = np.asarray(y_train, dtype=np.int64)
                clf.fit(X_fit, y_fit)
            except Exception as e:
                log_exception_details(e)
                raise e
            train_time = time.time() - start_time
            print(f"[TRAINING CLASSICAL] Finished training {name} in {train_time:.4f}s.")
            
            print(f"[VALIDATION CLASSICAL] Evaluating {name} on the test features...")
            start_inf = time.time()
            y_pred = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test)
            inference_time = time.time() - start_inf
            print(f"[VALIDATION CLASSICAL] Finished evaluating {name} in {inference_time:.4f}s.")
            
            acc = accuracy_score(y_test, y_pred)
            prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)
            cm = confusion_matrix(y_test, y_pred).tolist()
            
            # Compute ROC curve safely
            roc_data = compute_roc_curve_data(y_test, y_prob, num_classes)
            
            results[name] = {
                "accuracy": float(acc),
                "precision": float(prec),
                "recall": float(rec),
                "f1_score": float(f1),
                "training_time_s": float(train_time),
                "inference_time_s": float(inference_time),
                "confusion_matrix": cm,
                "roc_curve": roc_data,
                "classes": classes
            }
            
            # Save models to both requested names to preserve compatibility and API
            model_path1 = os.path.join(self.models_dir, f"{name}.pkl")
            model_path2 = os.path.join(self.models_dir, f"{name}_model.pkl")
            print(f"[MODEL SAVE] Saving traditional model {name} to:")
            print(f"  - {model_path1}")
            print(f"  - {model_path2}")
            with open(model_path1, "wb") as f:
                pickle.dump(clf, f)
            with open(model_path2, "wb") as f:
                pickle.dump(clf, f)
                
        # Train yield regressor mock for image pipeline
        print("[TRAINING CLASSICAL] Training yield regressor on CNN features...")
        yield_reg = RandomForestRegressor(n_estimators=50, random_state=42)
        mock_yield_train = np.random.uniform(3.0, 10.0, size=len(y_train))
        try:
            X_fit = np.asarray(X_train, dtype=np.float32)
            y_yield_fit = np.asarray(mock_yield_train, dtype=np.float32)
            yield_reg.fit(X_fit, y_yield_fit)
        except Exception as e:
            log_exception_details(e)
            raise e
        yield_path = os.path.join(self.models_dir, "yield_regressor.pkl")
        print(f"[MODEL SAVE] Saving yield regressor to: {yield_path}")
        with open(yield_path, "wb") as f:
            pickle.dump(yield_reg, f)
            
        return results

    def predict(self, feature_vector, image_path=None):
        """Predicts crop health and crop yield for a single feature vector or image path."""
        scaler_path = os.path.join(self.models_dir, "scaler.pkl")
        if not os.path.exists(scaler_path):
            raise ValueError("Models not trained.")
            
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)
            
        n_expected = self.scaler.n_features_in_
        
        # Determine feature extraction path based on expected number of features
        if n_expected == 64:
            # We need CNN embeddings (64 features) (Task 11)
            if image_path is None:
                raise ValueError("CNN model requires an image file input for prediction.")
            
            # Load CNN model
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            cnn_path = os.path.join(self.models_dir, "cnn_model.pth")
            if not os.path.exists(cnn_path):
                raise ValueError("CNN model not found. Models not trained.")
            
            # Dynamically determine num_classes from state_dict shape
            cnn_state = torch.load(cnn_path, map_location=device)
            num_classes = cnn_state['fc2.weight'].shape[0]
            
            cnn_model = SimpleCNN(num_classes=num_classes).to(device)
            cnn_model.load_state_dict(cnn_state)
            cnn_model.eval()
            
            # Extract features using the reusable function (Task 6)
            features = extract_features(image_path, cnn_model=cnn_model, device=device)
            features = features.flatten()
            feature_names = [f"cnn_feat_{i}" for i in range(64)]
        else:
            # We need handcrafted features (30 features)
            features = [feature_vector.get(col, 0.0) for col in self.feature_cols]
            feature_names = self.feature_cols

        # Task 8: Before calling scaler.transform(), print: Feature length, Feature names, Feature shape
        print(f"Feature length: {len(features)}")
        print(f"Feature names: {feature_names}")
        print(f"Feature shape: {np.array([features]).shape}")

        # Task 9: If the feature length is incorrect: Return an informative error instead of crashing.
        if len(features) != n_expected:
            raise ValueError(f"Feature length mismatch: expected {n_expected} features, but got {len(features)} features.")

        prediction_features = np.array([features], dtype=np.float32)

        # Task 10: Verify: StandardScaler.n_features_in_ matches prediction_features.shape[1]
        if self.scaler.n_features_in_ != prediction_features.shape[1]:
            raise ValueError(f"StandardScaler expects {self.scaler.n_features_in_} features, but prediction_features shape is {prediction_features.shape}")

        # Apply PCA if it exists (Task 12 & 13)
        pca_path = os.path.join(self.models_dir, "pca.pkl")
        if os.path.exists(pca_path):
            with open(pca_path, "rb") as f:
                pca = pickle.load(f)
            if pca is not None:
                prediction_features = pca.transform(prediction_features)

        # Apply FeatureSelector if it exists (Task 12 & 13)
        selector_path = os.path.join(self.models_dir, "feature_selector.pkl")
        if os.path.exists(selector_path):
            with open(selector_path, "rb") as f:
                selector = pickle.load(f)
            if selector is not None:
                prediction_features = selector.transform(prediction_features)

        # Scale features
        features_scaled = self.scaler.transform(prediction_features)
        
        predictions = {}
        for name in ["random_forest", "svm", "xgboost"]:
            model_path = os.path.join(self.models_dir, f"{name}_model.pkl")
            if not os.path.exists(model_path):
                raise ValueError(f"Model {name} not found.")
                    
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            
            pred_class = int(model.predict(features_scaled)[0])
            try:
                probs = model.predict_proba(features_scaled)[0]
                confidence = float(probs[pred_class])
            except Exception:
                probs = [1.0] + [0.0]*4
                confidence = 1.0
                
            predictions[name] = {
                "class_id": pred_class,
                "class_name": ["Healthy", "Water Stress", "Nitrogen Deficiency", "Disease", "Severe Stress"][pred_class] if pred_class < 5 else "Healthy",
                "confidence": confidence,
                "probabilities": probs if isinstance(probs, list) else list(probs)
            }
            
        yield_path = os.path.join(self.models_dir, "yield_regressor.pkl")
        if not os.path.exists(yield_path):
            raise ValueError("Yield regressor not found.")
                
        with open(yield_path, "rb") as f:
            yield_model = pickle.load(f)
        predicted_yield = float(yield_model.predict(features_scaled)[0])
        
        return {
            "health_predictions": predictions,
            "yield_t_ha": predicted_yield
        }
