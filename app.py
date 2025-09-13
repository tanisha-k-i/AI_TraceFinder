import streamlit as st
import pandas as pd
import numpy as np
import os
import cv2
from PIL import Image
from scipy.stats import skew, kurtosis, entropy


st.set_page_config(page_title="Forgery Dataset Feature Extractor", layout="wide")
st.title("✍️ Forged Handwritten Document Database - Auto Class Detection & Feature Extraction")



def extract_features(image_path, main_class, resolution):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return {
                "file_name": os.path.basename(image_path),
                "main_class": main_class,
                "resolution": resolution,
                "class_label": f"{main_class}_{resolution}",
                "error": "Unreadable file"
            }

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        
        height, width = gray.shape
        file_size = os.path.getsize(image_path) / 1024  
        aspect_ratio = round(width / height, 3)

        
        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)
        skewness = skew(gray.flatten())
        kurt = kurtosis(gray.flatten())

        # Entropy
        hist = np.histogram(gray, bins=256, range=(0, 255))[0]
        shannon_entropy = entropy(hist + 1e-9)

        # Edge features
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.mean(edges > 0)

        return {
            "file_name": os.path.basename(image_path),
            "main_class": main_class,
            "resolution": resolution,
            "class_label": f"{main_class}_{resolution}",
            "width": width,
            "height": height,
            "aspect_ratio": aspect_ratio,
            "file_size_kb": round(file_size, 2),
            "mean_intensity": round(mean_intensity, 3),
            "std_intensity": round(std_intensity, 3),
            "skewness": round(skewness, 3),
            "kurtosis": round(kurt, 3),
            "entropy": round(shannon_entropy, 3),
            "edge_density": round(edge_density, 3)
        }
    except Exception as e:
        return {
            "file_name": image_path,
            "main_class": main_class,
            "resolution": resolution,
            "class_label": f"{main_class}_{resolution}",
            "error": str(e)
        }



dataset_root = st.text_input(" Enter dataset root path:", "")

if dataset_root and os.path.isdir(dataset_root):
    st.info("🔎 Scanning dataset...")
    records = []

    # Top-level classes
    main_classes = [f for f in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, f))]
    st.success(f" Detected {len(main_classes)} main classes: {main_classes}")

    for main_class in main_classes:
        main_class_path = os.path.join(dataset_root, main_class)

        # Subfolders (150, 300)
        subfolders = [sf for sf in os.listdir(main_class_path) if os.path.isdir(os.path.join(main_class_path, sf))]

        for sub in subfolders:
            sub_path = os.path.join(main_class_path, sub)

            # Collect images (including .tif)
            files = [f for f in os.listdir(sub_path) if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff"))]

            st.write(f" Class '{main_class}/{sub}' → {len(files)} images")
            for fname in files:
                path = os.path.join(sub_path, fname)
                rec = extract_features(path, main_class, sub)
                records.append(rec)

    # Convert to DataFrame
    df = pd.DataFrame(records)
    st.subheader(" Features Extracted (Preview)")
    st.dataframe(df.head(20))

    # Save features
    save_path = os.path.join(dataset_root, "metadata_features.csv")
    df.to_csv(save_path, index=False)
    st.success(f" Features saved to {save_path}")

    # Class distribution
    if "class_label" in df.columns:
        st.subheader(" Class Distribution")
        st.bar_chart(df["class_label"].value_counts())

    # Sample Images
    st.subheader(" Sample Images")
    cols = st.columns(5)

    shown_classes = set()
    for idx, row in df.iterrows():
        cls_label = row["class_label"]
        if cls_label not in shown_classes:
            sample_path = os.path.join(dataset_root, row["main_class"], row["resolution"], row["file_name"])
            if os.path.exists(sample_path):
                try:
                    img = Image.open(sample_path)
                    cols[len(shown_classes) % 5].image(img, caption=cls_label, width="stretch")
                    shown_classes.add(cls_label)
                except:
                    st.warning(f"⚠️ Could not display sample image: {sample_path}")

else:
    if dataset_root:
        st.error("❌ Invalid dataset path. Please enter a valid folder.")