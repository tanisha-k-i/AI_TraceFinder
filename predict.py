import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from skimage.feature import local_binary_pattern as sk_lbp
from sklearn.preprocessing import StandardScaler
import pywt
import cv2

# --- 1. SET UP FILE PATHS (UPDATED TO MATCH YOUR FILES) ---
MODEL_PATH = "AI Tracefinder Hybrid 14 Final.keras"
SCALER_PATH = "AI Tracefinder Hybrid Feature Scaler.pkl"
ENCODER_PATH = "Hybrid Label Encoder.pkl"
FPS_PATH = "Fingerprints 14.pkl"
KEYS_PATH = "AI Tracefinder Keys.npy"

# --- 2. LOAD MODEL AND ARTIFACTS ---
print("⏳ Loading model and artifacts...")
try:
    model = keras.models.load_model(MODEL_PATH, compile=False)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(ENCODER_PATH, "rb") as f:
        le = pickle.load(f)
    with open(FPS_PATH, "rb") as f:
        scanner_fps = pickle.load(f)
    fp_keys = np.load(KEYS_PATH, allow_pickle=True).tolist()
    print("✅ All files loaded successfully.")
except FileNotFoundError as fnf_error:
    print(f"❌ Error: File not found - {fnf_error}.")
    print("Please check the filenames at the top of the predict.py script.")
    exit()
except Exception as e:
    print(f"❌ Error loading files: {e}")
    exit()

# --- 3. HELPER FUNCTIONS (Copied from Notebook) ---
IMG_SIZE = (256, 256)

def corr2d(a, b):
    a = a.astype(np.float32).ravel(); b = b.astype(np.float32).ravel()
    a -= a.mean(); b -= b.mean()
    d = np.linalg.norm(a)*np.linalg.norm(b)
    return float((a @ b) / d) if d != 0 else 0.0

def fft_radial_energy(img, K=6):
    f = np.fft.fftshift(np.fft.fft2(img)); mag = np.abs(f)
    h, w = mag.shape; cy, cx = h//2, w//2
    yy, xx = np.ogrid[:h, :w]; r = np.sqrt((yy - cy)**2 + (xx - cx)**2)
    rmax = r.max() + 1e-6; bins = np.linspace(0, rmax, K + 1)
    feats = []
    for i in range(K):
        m = (r >= bins[i]) & (r < bins[i + 1]);
        feats.append(float(mag[m].mean() if m.any() else 0.0))
    return feats

def lbp_hist_safe(img, P=8, R=1.0):
    rng = float(np.ptp(img))
    g = np.zeros_like(img, dtype=np.float32) if rng < 1e-12 else (img - float(np.min(img))) / (rng + 1e-8)
    g8 = (g * 255.0).astype(np.uint8)
    codes = sk_lbp(g8, P=P, R=R, method="uniform")
    n_bins = P + 2
    hist, _ = np.histogram(codes, bins=np.arange(n_bins + 1), density=True)
    return hist.astype(np.float32).tolist()

def preprocess_residual_pywt(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Cannot read image file: {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    cA, (cH, cV, cD) = pywt.dwt2(img, 'haar')
    cH.fill(0); cV.fill(0); cD.fill(0)
    den = pywt.idwt2((cA, (cH, cV, cD)), 'haar')
    den = cv2.resize(den, IMG_SIZE, interpolation=cv2.INTER_NEAREST)
    residual = (img - den).astype(np.float32)
    if residual.shape != IMG_SIZE:
        residual = cv2.resize(residual, IMG_SIZE, interpolation=cv2.INTER_NEAREST)
    return residual

def make_features_from_residual(res):
    v_corr = [corr2d(res, scanner_fps[k]) for k in fp_keys]
    v_fft = fft_radial_energy(res, K=6)
    v_lbp = lbp_hist_safe(res, P=8, R=1.0)
    features = v_corr + v_fft + v_lbp
    features_np = np.array(features, dtype=np.float32).reshape(1, -1)
    features_scaled = scaler.transform(features_np)
    return features_scaled

def topk_indices_1d(p, k=3):
    p = np.asarray(p).ravel()
    k = min(k, p.size)
    idx = np.argpartition(p, -k)[-k:]
    idx = idx[np.argsort(p[idx])[::-1]]
    return idx

def predict_scanner_hybrid(image_path, top_k=3):
    try:
        residual = preprocess_residual_pywt(image_path)
        res_img_input = np.expand_dims(residual, axis=(0, -1))
        handcrafted_features_input = make_features_from_residual(residual)
        pred_prob = model.predict([res_img_input, handcrafted_features_input], verbose=0).ravel()
        top_indices = topk_indices_1d(pred_prob, k=top_k)
        top_labels = le.inverse_transform(top_indices)
        top_confidences = pred_prob[top_indices] * 100
        return list(zip(top_labels, top_confidences))
    except FileNotFoundError:
        print(f"❌ Error: Image file not found at {image_path}")
        return None
    except ValueError as ve:
        print(f"❌ Error processing image {image_path}: {ve}")
        return None
    except Exception as e:
        print(f"❌ An unexpected error occurred for {image_path}: {e}")
        return None

# --- 7. EXAMPLE USAGE ---
if __name__ == "__main__":
    # --- !! REPLACE WITH YOUR IMAGE PATH !! ---
    test_image = "'/Users/tanishakumari/Downloads/Test Image s11 2 (1).tif'" # Or .tif, .png etc.
    # --- !! ----------------------------- !! ---

    print(f"\n--- Predicting for: {test_image} ---")
    if not os.path.exists(test_image):
        print(f"⚠️ Test image '{test_image}' not found. Please update the path in the script.")
    else:
        results = predict_scanner_hybrid(test_image, top_k=3)
        if results:
            print("\n--- Prediction Results ---")
            for i, (label, confidence) in enumerate(results):
                print(f"Top {i+1}: {label} (Confidence: {confidence:.2f}%)")
        else:
            print("Prediction failed.")