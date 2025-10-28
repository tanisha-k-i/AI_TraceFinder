# app.py
import streamlit as st
import os
import pickle
import joblib # For loading baseline .pkl models
import numpy as np
import tensorflow as tf
import pywt # For Hybrid Model
import cv2 # For Hybrid Model
from skimage.feature import local_binary_pattern as sk_lbp # For Hybrid Model
from scipy.fft import fft2, fftshift # For Hybrid Model
from skimage.filters import sobel
from scipy.stats import skew, kurtosis, entropy
import pandas as pd
from PIL import Image
import io
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import tempfile # For saving temp file for baseline prediction

# --- Configuration: Hybrid Model (For "Hybrid Prediction" Page) ---
HYBRID_MODEL_PATH = os.path.join(".", "AI Tracefinder Hybrid 14 Final.keras")
HYBRID_LE_PATH = os.path.join(".", "Hybrid Label Encoder.pkl")
HYBRID_SCALER_PATH = os.path.join(".", "AI Tracefinder Hybrid Feature Scaler.pkl")
HYBRID_FP_PATH = os.path.join(".", "Fingerprints 14.pkl")
HYBRID_ORDER_NPY = os.path.join(".", "AI Tracefinder Keys.npy")
HYBRID_HISTORY_PKL_PATH = os.path.join(".", "hybrid_training_history.pkl")
HYBRID_EVAL_CSV_PATH = os.path.join(".", "evaluation_hybrid_14_test_split.csv")
EDA_CSV_PATH = os.path.join(".", "Metadata Features (1).csv") 

# --- Configuration: Baseline Models (For "Baseline Prediction" & "Baseline Performance") ---
BASELINE_CSV_PATH = os.path.join(".", "Baseline Models Metadata Features.csv")
BASELINE_SCALER_PATH = os.path.join("models", "scaler.pkl")
BASELINE_SVM_PATH = os.path.join("models", "svm.pkl")
BASELINE_RF_PATH = os.path.join("models", "random_forest.pkl")

# --- Configuration: Page Images ---
# !! THESE LINES ARE UPDATED !!
RF_IMAGE_PATH = os.path.join(".", "Random_Forest_confusion_matrix.png")
SVM_IMAGE_PATH = os.path.join(".", "SVM_confusion_matrix.png")

# --- Hybrid Model: Logic Configuration ---
EXPECTED_CLASSES = 11
EXPECTED_KEYS = 11
FEATURE_DIM = 27 # 11 (corr) + 6 (fft) + 10 (lbp)
IMG_SIZE = (256, 256) # For Hybrid model preprocessing

# --- Hybrid Model: Load Main Artifacts ---
@st.cache_resource
def load_hybrid_model_artifacts():
    """Loads the main HYBRID TF model, label encoder, scaler, and fingerprints."""
    required_hybrid_files = [HYBRID_MODEL_PATH, HYBRID_LE_PATH, HYBRID_SCALER_PATH, HYBRID_FP_PATH, HYBRID_ORDER_NPY]
    missing_files = [f for f in required_hybrid_files if not os.path.exists(f)]
    if missing_files:
        st.error(f"Error: Missing required HYBRID model files: {', '.join(missing_files)}. Cannot run predictions.")
        return None, None, None, None, None
    try:
        model = tf.keras.models.load_model(HYBRID_MODEL_PATH)
        with open(HYBRID_LE_PATH, "rb") as f: label_encoder = pickle.load(f)
        with open(HYBRID_SCALER_PATH, "rb") as f: scaler = pickle.load(f)
        with open(HYBRID_FP_PATH, "rb") as f: scanner_fps = pickle.load(f)
        fp_keys = np.load(HYBRID_ORDER_NPY, allow_pickle=True).tolist()
        return model, label_encoder, scaler, scanner_fps, fp_keys
    except Exception as e:
        st.error(f"Error loading hybrid model artifacts: {e}")
        return None, None, None, None, None

model, le, scaler, scanner_fps, fp_keys = load_hybrid_model_artifacts()

# --- Baseline Model: Load Artifacts ---
@st.cache_resource
def load_baseline_models():
    """Loads the BASELINE scaler, SVM, and RF models."""
    required_baseline_files = [BASELINE_SCALER_PATH, BASELINE_SVM_PATH, BASELINE_RF_PATH]
    missing_files = [f for f in required_baseline_files if not os.path.exists(f)]
    if missing_files:
        st.error(f"Error: Missing required BASELINE model files: {', '.join(missing_files)}. Cannot run Baseline Prediction.")
        return None, None, None
    try:
        baseline_scaler = joblib.load(BASELINE_SCALER_PATH)
        svm_model = joblib.load(BASELINE_SVM_PATH)
        rf_model = joblib.load(BASELINE_RF_PATH)
        return baseline_scaler, svm_model, rf_model
    except Exception as e:
        st.error(f"Error loading baseline models: {e}")
        return None, None, None

baseline_scaler, svm_model, rf_model = load_baseline_models()


# --- Hybrid Model: Preprocessing & Prediction Functions ---
def preprocess_residual_pywt(image_bytes):
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        if img is None: raise ValueError("Cannot decode image bytes.")
        if img.ndim == 3: img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)
        img_float = img.astype(np.float32) / 255.0
        coeffs = pywt.dwt2(img_float, 'haar')
        cA, (cH, cV, cD) = coeffs
        cH.fill(0); cV.fill(0); cD.fill(0)
        denoised = pywt.idwt2((cA, (cH, cV, cD)), 'haar')
        denoised = cv2.resize(denoised, IMG_SIZE, interpolation=cv2.INTER_AREA)
        residual = img_float - denoised
        return residual.astype(np.float32)
    except Exception as e:
        st.error(f"Error during image preprocessing: {e}"); return None

def corr2d(a, b):
    a = a.astype(np.float32).ravel(); b = b.astype(np.float32).ravel()
    a -= a.mean(); b -= b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float((a @ b) / denom) if denom != 0 else 0.0

def fft_radial_energy(img, K=6):
    try:
        f = fftshift(fft2(img)); mag = np.abs(f)
        h, w = mag.shape; cy, cx = h // 2, w // 2
        yy, xx = np.ogrid[:h, :w]; r = np.sqrt((yy - cy)**2 + (xx - cx)**2)
        rmax = r.max() + 1e-6; bins = np.linspace(0, rmax, K + 1)
        feats = []
        for i in range(K):
            mask = (r >= bins[i]) & (r < bins[i+1])
            feats.append(float(mag[mask].mean() if mask.any() else 0.0))
        return feats
    except Exception as e:
        st.error(f"Error calculating FFT features: {e}"); return [0.0] * K

def lbp_hist_safe(img, P=8, R=1.0):
    try:
        n_bins = P + 2; img_min = np.min(img); img_max = np.max(img)
        if img_max - img_min < 1e-6: img_norm = np.zeros_like(img, dtype=np.uint8)
        else: img_norm = ((img - img_min) / (img_max - img_min + 1e-9) * 255).astype(np.uint8)
        lbp_codes = sk_lbp(img_norm, P=P, R=R, method='uniform')
        hist, _ = np.histogram(lbp_codes.ravel(), bins=np.arange(0, n_bins + 1), range=(0, n_bins), density=True)
        return hist.astype(np.float32).tolist()
    except Exception as e:
        st.error(f"Error calculating LBP features: {e}"); return [0.0] * n_bins

def make_feats_from_res(residual, scanner_fingerprints, fingerprint_keys, feature_scaler):
    try:
        v_corr = [corr2d(residual, scanner_fingerprints[k]) for k in fingerprint_keys] # 11 features
        v_fft = fft_radial_energy(residual, K=6) # 6 features
        v_lbp = lbp_hist_safe(residual, P=8, R=1.0) # 10 features
        v = np.array(v_corr + v_fft + v_lbp, dtype=np.float32).reshape(1, -1) # 27 features
        if v.shape[1] != FEATURE_DIM:
            st.error(f"Feature extraction produced {v.shape[1]} features, but expected {FEATURE_DIM}.")
            return None
        v_scaled = feature_scaler.transform(v)
        return v_scaled
    except KeyError as e:
         st.error(f"Error during feature extraction: Key '{e}' not found in fingerprints."); return None
    except Exception as e:
        st.error(f"Error during feature extraction: {e}"); return None

def predict_scanner_hybrid(image_bytes):
    """Predicts using the main HYBRID model."""
    if model is None: st.error("Hybrid model is not loaded."); return None, None, None
    residual = preprocess_residual_pywt(image_bytes)
    if residual is None: return None, None, None
    handcrafted_features = make_feats_from_res(residual, scanner_fps, fp_keys, scaler)
    if handcrafted_features is None: return None, None, None 
    residual_img_input = np.expand_dims(residual, axis=(0, -1)) 
    try:
        predictions = model.predict([residual_img_input, handcrafted_features])
        probabilities = predictions[0]
        top1_index = np.argmax(probabilities)
        top1_label = le.classes_[top1_index]
        top1_confidence = probabilities[top1_index] * 100
        top_k_indices = np.argsort(probabilities)[::-1]
        top_k_results = [(le.classes_[i], probabilities[i] * 100) for i in top_k_indices]
        return top1_label, top1_confidence, top_k_results
    except Exception as e:
        st.error(f"Error during model prediction: {e}"); return None, None, None
# --- End of Hybrid Model Functions ---


# --- Baseline Model: Preprocessing & Prediction Functions ---
def load_and_preprocess_baseline(img_path, size=(512, 512)):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f" Could not load image: {img_path}")
    img = img.astype(np.float32) / 255.0
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

def compute_metadata_features(img, file_path):
    h, w = img.shape
    aspect_ratio = w / h
    file_size_kb = os.path.getsize(file_path) / 1024
    pixels = img.flatten()
    mean_intensity = np.mean(pixels)
    std_intensity = np.std(pixels)
    skewness = skew(pixels)
    kurt = kurtosis(pixels)
    ent = entropy(np.histogram(pixels, bins=256, range=(0,1))[0] + 1e-6)
    edges = sobel(img)
    edge_density = np.mean(edges > 0.1)
    return {
        "width": w, "height": h, "aspect_ratio": aspect_ratio,
        "file_size_kb": file_size_kb, "mean_intensity": mean_intensity,
        "std_intensity": std_intensity, "skewness": skewness,
        "kurtosis": kurt, "entropy": ent, "edge_density": edge_density
    }

def predict_scanner_baseline(image_path):
    """Predicts using the BASELINE models (SVM, RF)."""
    if baseline_scaler is None or svm_model is None or rf_model is None:
        st.error("Baseline models are not loaded. Cannot perform prediction.")
        return None, None
    
    try:
        img = load_and_preprocess_baseline(image_path)
        features = compute_metadata_features(img, image_path)
        df = pd.DataFrame([features])
        X_scaled = baseline_scaler.transform(df)

        # RF Prediction
        rf_pred = rf_model.predict(X_scaled)[0]
        rf_prob = rf_model.predict_proba(X_scaled)[0]
        rf_top1_conf = rf_prob.max() * 100
        rf_results = (rf_pred, rf_top1_conf, rf_model.classes_, rf_prob)

        # SVM Prediction
        svm_pred = svm_model.predict(X_scaled)[0]
        try:
            svm_prob = svm_model.predict_proba(X_scaled)[0]
            svm_top1_conf = svm_prob.max() * 100
        except AttributeError: 
            svm_prob = None
            svm_top1_conf = 100.0 
        
        svm_results = (svm_pred, svm_top1_conf, svm_model.classes_, svm_prob)

        return rf_results, svm_results

    except Exception as e:
        st.error(f"Error during baseline prediction: {e}")
        return None, None
# --- End of Baseline Model Functions ---


# --- Streamlit UI: Page 1 (Hybrid Prediction) ---
def page_hybrid_prediction():
    st.header("Hybrid Model Prediction")
    st.write("Upload an image to identify the scanner model (using the main Hybrid Keras Model).")
    
    if model is None:
        st.error("Hybrid Keras model failed to load. Cannot perform predictions.")
        return

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "tif", "tiff"], key="hybrid_uploader")

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image.', use_column_width=True)
        except Exception as e:
            st.error(f"Error displaying image: {e}"); return
        image_bytes = uploaded_file.getvalue()

        if st.button("Predict with Hybrid Model"):
            with st.spinner('Processing and predicting (Hybrid)...'):
                label, confidence, top_k = predict_scanner_hybrid(image_bytes)
            if label is not None:
                st.subheader("Hybrid Prediction Result")
                st.success(f"Predicted Scanner Model: **{label}**")
                st.info(f"Confidence: **{confidence:.2f}%**")
                st.subheader("Top Prediction Probabilities")
                prob_df = pd.DataFrame(top_k, columns=["Scanner Model", "Probability (%)"])
                prob_df["Probability (%)"] = prob_df["Probability (%)"].map('{:.2f}%'.format)
                st.dataframe(prob_df.head(EXPECTED_CLASSES))
            else:
                st.error("Prediction failed. Check error messages above.")

# --- Streamlit UI: Page 2 (NEW: Baseline Prediction) ---
def page_baseline_prediction():
    st.header("Baseline Model Prediction")
    st.write("Upload an image to identify the scanner model (using SVM and Random Forest).")
    st.warning("Note: These models predict based on *metadata* (file size, aspect ratio, etc.), not image texture.")

    if baseline_scaler is None or svm_model is None or rf_model is None:
        st.error("Baseline models failed to load. Cannot perform predictions.")
        return

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "tif", "tiff"], key="baseline_uploader")

    if uploaded_file is not None:
        try:
            image_pil = Image.open(uploaded_file)
            st.image(image_pil, caption='Uploaded Image.', use_column_width=True)
        except Exception as e:
            st.error(f"Error displaying image: {e}"); return
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
            tmp.write(uploaded_file.getvalue())
            temp_file_path = tmp.name

        if st.button("Predict with Baseline Models"):
            with st.spinner('Processing and predicting (Baseline)...'):
                rf_results, svm_results = predict_scanner_baseline(temp_file_path)

            if rf_results and svm_results:
                col1, col2 = st.columns(2)
                
                # --- Random Forest Results ---
                with col1:
                    st.subheader("Random Forest Result")
                    if os.path.exists(RF_IMAGE_PATH):
                        st.image(RF_IMAGE_PATH, caption="Random Forest Confusion Matrix")
                    else:
                        st.warning(f"Image not found: {RF_IMAGE_PATH}")
                    rf_label, rf_conf, rf_classes, rf_probs = rf_results
                    st.success(f"Predicted Model: **{rf_label}**")
                    st.info(f"Confidence: **{rf_conf:.2f}%**")
                    
                    if rf_probs is not None:
                        st.subheader("Top Probabilities (RF)")
                        rf_top_k = sorted(zip(rf_classes, rf_probs * 100), key=lambda x: x[1], reverse=True)
                        prob_df_rf = pd.DataFrame(rf_top_k, columns=["Scanner Model", "Probability (%)"])
                        prob_df_rf["Probability (%)"] = prob_df_rf["Probability (%)"].map('{:.2f}%'.format)
                        st.dataframe(prob_df_rf.head())

                # --- SVM Results ---
                with col2:
                    st.subheader("SVM Result")
                    if os.path.exists(SVM_IMAGE_PATH):
                        st.image(SVM_IMAGE_PATH, caption="SVM Confusion Matrix")
                    else:
                        st.warning(f"Image not found: {SVM_IMAGE_PATH}")
                    svm_label, svm_conf, svm_classes, svm_probs = svm_results
                    st.success(f"Predicted Model: **{svm_label}**")
                    if svm_probs is not None:
                         st.info(f"Confidence: **{svm_conf:.2f}%**")
                         st.subheader("Top Probabilities (SVM)")
                         svm_top_k = sorted(zip(svm_classes, svm_probs * 100), key=lambda x: x[1], reverse=True)
                         prob_df_svm = pd.DataFrame(svm_top_k, columns=["Scanner Model", "Probability (%)"])
                         prob_df_svm["Probability (%)"] = prob_df_svm["Probability (%)"].map('{:.2f}%'.format)
                         st.dataframe(prob_df_svm.head())
                    else:
                        st.info("Confidence scores not available for this SVM model.")
            else:
                st.error("Baseline prediction failed. Check error messages above.")
        
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

# --- Streamlit UI: Page 3 (EDA) ---
def page_exploratory_data_analysis():
    st.header("Exploratory Data Analysis (EDA)")
    st.write(f"Visualizations based on `{os.path.basename(EDA_CSV_PATH)}`.")
    if os.path.exists(EDA_CSV_PATH):
        try:
            df_meta = pd.read_csv(EDA_CSV_PATH)
            st.subheader("Dataset Overview")
            st.dataframe(df_meta.head())
            st.subheader("Class Distribution")
            label_column = 'resolution' 
            if label_column in df_meta.columns:
                 fig, ax = plt.subplots()
                 class_counts = df_meta[label_column].value_counts()
                 sns.barplot(x=class_counts.index, y=class_counts.values, ax=ax)
                 plt.xticks(rotation=45, ha='right')
                 ax.set_title("Number of Images per Scanner Class")
                 ax.set_ylabel("Count")
                 st.pyplot(fig)
            else:
                 st.error(f"Could not find column named '{label_column}' in `{os.path.basename(EDA_CSV_PATH)}`.")
        except Exception as e:
            st.error(f"Could not load or plot EDA data: {e}")
    else:
        st.warning(f"File '{os.path.basename(EDA_CSV_PATH)}' not found. Cannot display EDA.")

# --- Streamlit UI: Page 4 (Batch Testing) ---
def page_model_testing():
    st.header("Model Testing (Batch Prediction)")
    st.write("Upload multiple images for batch prediction (using the main Hybrid Keras Model).")
    
    if model is None:
        st.error("Hybrid Keras model failed to load. Cannot perform predictions.")
        return

    uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png", "tif", "tiff"], accept_multiple_files=True, key="batch_uploader")
    results_list = []

    if uploaded_files:
        st.write(f"Processing {len(uploaded_files)} files...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        error_occurred = False

        for i, uploaded_file in enumerate(uploaded_files):
            label, confidence = "Error", "N/A"
            try:
                status_text.text(f"Processing: {uploaded_file.name}")
                image_bytes = uploaded_file.getvalue()
                pred_label, pred_conf, _ = predict_scanner_hybrid(image_bytes)
                if pred_label is not None:
                    label, confidence = pred_label, f"{pred_conf:.2f}"
                else:
                    error_occurred = True; label = "Prediction Failed"
            except Exception as e:
                 st.error(f"Unexpected error processing {uploaded_file.name}: {e}")
                 error_occurred = True; label = "Processing Error"

            results_list.append({"Filename": uploaded_file.name, "Predicted Scanner": label, "Confidence (%)": confidence})
            progress_bar.progress((i + 1) / len(uploaded_files))

        status_text.text("Batch prediction complete!")
        if results_list:
            df_results = pd.DataFrame(results_list)
            st.dataframe(df_results)
            csv = df_results.to_csv(index=False).encode('utf-8')
            st.download_button(label="Download Results as CSV", data=csv, file_name='batch_predictions.csv', mime='text/csv')
        if error_occurred:
             st.warning("One or more files failed during prediction. See error messages above.")

# --- Streamlit UI: Page 5 (Baseline Performance) ---
@st.cache_data 
def get_baseline_scores():
    """Loads baseline data, scaler, and models, and calculates scores."""
    required_baseline_files = [BASELINE_CSV_PATH, BASELINE_SCALER_PATH, BASELINE_SVM_PATH, BASELINE_RF_PATH]
    missing_files = [f for f in required_baseline_files if not os.path.exists(f)]
    if missing_files:
        st.error(f"Error: Missing required BASELINE files: {', '.join(missing_files)}. Cannot show this page.")
        return None
    try:
        df = pd.read_csv(BASELINE_CSV_PATH)
        X = df.drop(columns=["file_name", "main_class", "resolution", "class_label"])
        y = df["class_label"]
    except Exception as e:
        st.error(f"Error loading baseline data from '{os.path.basename(BASELINE_CSV_PATH)}': {e}"); return None
    try:
        scaler_perf = joblib.load(BASELINE_SCALER_PATH)
        X_scaled = scaler_perf.transform(X)
    except Exception as e:
        st.error(f"Error loading or using baseline scaler from '{os.path.basename(BASELINE_SCALER_PATH)}': {e}"); return None
    model_paths = {"SVM": BASELINE_SVM_PATH, "Random Forest": BASELINE_RF_PATH}
    results = []
    for model_name, path in model_paths.items():
        if not os.path.exists(path):
            st.warning(f"File not found: {os.path.basename(path)}. Skipping model: {model_name}"); continue
        try:
            model_perf = joblib.load(path)
            y_pred = model_perf.predict(X_scaled)
            accuracy = accuracy_score(y, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average='macro', zero_division=0)
            results.append({"Model": model_name, "Accuracy": accuracy, "Precision (macro)": precision, "Recall (macro)": recall, "F1-score (macro)": f1})
        except Exception as e:
            st.error(f"Error testing model {model_name} from {os.path.basename(path)}: {e}")
    if not results:
        st.error("No baseline models were successfully tested."); return None
    return pd.DataFrame(results)

def page_baseline_performance():
    st.header("Baseline Model Performance")
    st.write("Comparison of traditional ML models (SVM, RF) on the *Metadata Features*.")
    st.write("Scores are calculated live by running the saved models on the entire baseline dataset.")
    results_df = get_baseline_scores()
    if results_df is not None:
        try:
            results_df['Accuracy'] = results_df['Accuracy'].map('{:.2%}'.format)
            results_df['Precision (macro)'] = results_df['Precision (macro)'].map('{:.2%}'.format)
            results_df['Recall (macro)'] = results_df['Recall (macro)'].map('{:.2%}'.format)
            results_df['F1-score (macro)'] = results_df['F1-score (macro)'].map('{:.2%}'.format)
        except KeyError: pass 
        st.subheader("Baseline Model Metrics"); st.dataframe(results_df)

# --- Streamlit UI: Page 6 (Hybrid Performance) ---
def page_hybrid_model_performance():
    st.header("Hybrid Model Performance")
    st.write("Visualization of the Hybrid CNN + Handcrafted model's performance.")
    if os.path.exists(HYBRID_HISTORY_PKL_PATH):
        try:
            with open(HYBRID_HISTORY_PKL_PATH, "rb") as f: history = pickle.load(f)
            if isinstance(history, dict) and all(k in history for k in ['accuracy', 'val_accuracy', 'loss', 'val_loss']):
                 st.subheader("Training History (Step E)")
                 fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                 ax1.plot(history['accuracy'], label='Train Accuracy'); ax1.plot(history['val_accuracy'], label='Val Accuracy')
                 ax1.set_title('Model Accuracy'); ax1.set_ylabel('Accuracy'); ax1.set_xlabel('Epoch'); ax1.legend()
                 ax2.plot(history['loss'], label='Train Loss'); ax2.plot(history['val_loss'], label='Val Loss')
                 ax2.set_title('Model Loss'); ax2.set_ylabel('Loss'); ax2.set_xlabel('Epoch'); ax2.legend()
                 st.pyplot(fig)
            else: st.warning(f"File '{os.path.basename(HYBRID_HISTORY_PKL_PATH)}' is not a valid Keras history dictionary.")
        except Exception as e: st.error(f"Could not load/plot history: {e}")
    else:
        st.info(f"Optional file '{os.path.basename(HYBRID_HISTORY_PKL_PATH)}' not found.")
    if os.path.exists(HYBRID_EVAL_CSV_PATH):
        try:
            eval_df = pd.read_csv(HYBRID_EVAL_CSV_PATH)
            st.subheader(f"Test Set Evaluation (Step G)")
            st.dataframe(eval_df.head())
            true_label_col = 'true_label'; pred_label_col = 'pred_label'
            if true_label_col in eval_df.columns and pred_label_col in eval_df.columns:
                 y_true_eval = eval_df[true_label_col]; y_pred_eval = eval_df[pred_label_col]
                 labels_in_encoder = list(le.classes_) 
                 accuracy_eval = accuracy_score(y_true_eval, y_pred_eval)
                 st.metric("Overall Test Accuracy (from CSV)", f"{accuracy_eval:.2%}")
                 st.subheader("Classification Report (Test Set - from CSV)")
                 report_dict = classification_report(y_true_eval, y_pred_eval, labels=labels_in_encoder, target_names=labels_in_encoder, zero_division=0, output_dict=True)
                 st.dataframe(pd.DataFrame(report_dict).transpose())
                 st.subheader("Confusion Matrix (Test Set - from CSV)")
                 cm = confusion_matrix(y_true_eval, y_pred_eval, labels=labels_in_encoder)
                 fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
                 sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                             xticklabels=labels_in_encoder, yticklabels=labels_in_encoder, ax=ax_cm)
                 plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0)
                 ax_cm.set_xlabel('Predicted Label'); ax_cm.set_ylabel('True Label'); ax_cm.set_title('Confusion Matrix')
                 st.pyplot(fig_cm)
            else: st.warning(f"Required columns '{true_label_col}' or '{pred_label_col}' not found in `{os.path.basename(HYBRID_EVAL_CSV_PATH)}`.")
        except Exception as e: st.error(f"Could not load/process evaluation data: {e}")
    else:
        st.info(f"Optional evaluation file ('{os.path.basename(HYBRID_EVAL_CSV_PATH)}') not found.")


# --- Main App Logic ---
st.set_page_config(page_title="AI TraceFinder", layout="wide")
st.title("🔍 AI TraceFinder: Digital Scanner Identification")
st.sidebar.title("Navigation")

page = st.sidebar.radio("Go to", [
    "Hybrid Prediction",
    "Baseline Prediction",
    "Model Testing (Batch)",
    "Exploratory Data Analysis",
    "Baseline Performance",
    "Hybrid Model Performance"
])

# Main routing logic
if page == "Hybrid Prediction":
    if model is None:
        st.error("Hybrid Keras model failed to load. Check file paths and errors at the top.")
    else:
        page_hybrid_prediction()
elif page == "Baseline Prediction":
    if baseline_scaler is None or svm_model is None or rf_model is None:
        st.error("Baseline models failed to load. Check file paths (e.g., 'models/scaler.pkl') and errors at the top.")
    else:
        page_baseline_prediction()
elif page == "Model Testing (Batch)":
    if model is None:
        st.error("Hybrid Keras model failed to load. Check file paths and errors at the top.")
    else:
        page_model_testing()
elif page == "Exploratory Data Analysis":
    page_exploratory_data_analysis()
elif page == "Baseline Performance":
    page_baseline_performance()
elif page == "Hybrid Model Performance":
    if le is None: # Check if hybrid label encoder loaded
         st.error("Hybrid Keras model failed to load. Cannot show performance.")
    else:
        page_hybrid_model_performance()