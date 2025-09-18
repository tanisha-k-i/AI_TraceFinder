import pandas as pd
import joblib
import os  # --- FIX 1: Import the os module ---
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

CSV_PATH = "Baseline Models Metadata Features.csv"

def train_models():
    df = pd.read_csv(CSV_PATH)
    X = df.drop(columns=["file_name", "main_class", "resolution", "class_label"])
    y = df["class_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    # NOTE: You will need the scaler later for evaluation, so let's save it.
    
    X_test = scaler.transform(X_test)

    # --- FIX 2: Create the 'models' directory before saving files into it ---
    os.makedirs("models", exist_ok=True)

    # Save the scaler
    joblib.dump(scaler, "models/scaler.pkl")
    print("Scaler saved.")

    # Train and save Random Forest
    rf = RandomForestClassifier(n_estimators=300, random_state=42)
    rf.fit(X_train, y_train)
    joblib.dump(rf, "models/random_forest.pkl")
    print("Random Forest model saved.")

    # Train and save SVM
    svm = SVC(kernel="rbf", C=10, gamma="scale", probability=True)
    svm.fit(X_train, y_train)
    joblib.dump(svm, "models/svm.pkl")
    print("SVM model saved.")


# This part makes the script run the training process when you execute the file
if __name__ == "__main__":
    print("Starting model training...")
    train_models()
    print("Training complete. Models and scaler saved in 'models' directory.")
