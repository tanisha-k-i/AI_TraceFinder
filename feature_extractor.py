import streamlit as st
import pandas as pd
import numpy as np
import os
import cv2  


# --- 2. Configure the Web Application Page ---
st.set_page_config(
    page_title="Forensic Feature Extractor",
    page_icon="🔎",
    layout="wide"
)

st.title("🔎 Forensic Scanner Feature Extractor")
st.write("A tool to analyze image properties from different scanner devices.")
st.markdown("---") # Adds a visual separator

# --- 3. Image Analysis Function ---
def extract_image_metadata(image_filepath, device_class):
    """
    Analyzes a single image file and extracts a focused set of features relevant
    to identifying scanner hardware.
    
    Args:
        image_filepath (str): The full path to the image file.
        device_class (str): The name of the scanner/device (from the folder name).

    Returns:
        dict: A dictionary containing the extracted features for one image.
    """
    try:
        # Load the image in color and grayscale
        image_color = cv2.imread(image_filepath)
        if image_color is None:
            return {"filename": os.path.basename(image_filepath), "device_class": device_class, "error": "Cannot read file"}
        
        image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)

        # --- Feature Extraction ---
        # A. Basic File Properties
        height, width = image_gray.shape
        filesize_kb = round(os.path.getsize(image_filepath) / 1024, 2)

        # B. Pixel Intensity Statistics (Brightness & Contrast)
        mean_pixel_intensity = round(np.mean(image_gray), 2)
        std_dev_intensity = round(np.std(image_gray), 2) # A good proxy for contrast

        # C. Texture/Sharpness Metric (Laplacian Variance)
        # A low value suggests blur, a high value suggests high variance/sharpness.
        sharpness_score = round(cv2.Laplacian(image_gray, cv2.CV_64F).var(), 2)
        
        # D. Compile features into a record
        image_record = {
            "filename": os.path.basename(image_filepath),
            "device_class": device_class,
            "width": width,
            "height": height,
            "filesize_kb": filesize_kb,
            "brightness": mean_pixel_intensity,
            "contrast": std_dev_intensity,
            "sharpness": sharpness_score
        }
        return image_record

    except Exception as e:
        return {"filename": os.path.basename(image_filepath), "device_class": device_class, "error": str(e)}

# --- 4. Main Application Interface ---
st.header("Step 1: Locate Your Image Dataset")
root_folder_path = st.text_input(
    "Enter the full path to the folder containing your scanner sub-folders:",
    placeholder="/Users/yourname/Desktop/AI_TraceFinder_Dataset/data-tifs-2016-maps"
)

# --- 5. Processing Logic ---
if root_folder_path and os.path.isdir(root_folder_path):
    st.info(f"Accessing directory: {root_folder_path}")

    # Find all subdirectories, which represent the scanner classes
    device_folders = [d for d in os.listdir(root_folder_path) if os.path.isdir(os.path.join(root_folder_path, d))]

    if not device_folders:
        st.error("No device sub-folders found in the provided path. Please check the folder structure.")
    else:
        st.success(f"Found {len(device_folders)} device classes: {', '.join(device_folders)}")
        
        if st.button("▶️ Start Feature Extraction"):
            st.markdown("---")
            st.header("Step 2: Analysis in Progress...")
            
            all_image_records = []
            progress_bar = st.progress(0, text="Initializing...")

            # Loop through each device folder
            for i, device_name in enumerate(device_folders):
                folder_path = os.path.join(root_folder_path, device_name)
                image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
                
                status_text = f"Processing class '{device_name}' ({len(image_files)} images)..."
                progress_bar.progress((i + 1) / len(device_folders), text=status_text)

                # Loop through each image in the folder
                for image_name in image_files:
                    full_path = os.path.join(folder_path, image_name)
                    metadata = extract_image_metadata(full_path, device_name)
                    all_image_records.append(metadata)

            # --- 6. Display Results and Save ---
            st.header("Step 3: Analysis Complete")
            
            # Create a DataFrame from the collected data
            results_df = pd.DataFrame(all_image_records)
            st.dataframe(results_df)

            # Save the DataFrame to a CSV file
            output_filename = "forensic_features_dataset.csv"
            output_path = os.path.join(root_folder_path, output_filename)
            results_df.to_csv(output_path, index=False)
            
            st.success(f"✅ Success! Dataset saved to: {output_path}")

elif root_folder_path:
    st.warning("The path you entered does not seem to be a valid directory. Please check it.")

