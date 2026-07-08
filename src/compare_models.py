import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# ---- Baseline model (Logistic Regression on MFCCs) ----
df = pd.read_csv("data/features.csv")
X_mfcc = df.drop("label", axis=1)
y_mfcc = df["label"]

X_train_mfcc, X_test_mfcc, y_train_mfcc, y_test_mfcc = train_test_split(
    X_mfcc, y_mfcc, test_size=0.2, random_state=42
)

baseline_model = LogisticRegression(max_iter=1000)
baseline_model.fit(X_train_mfcc, y_train_mfcc)
baseline_preds = baseline_model.predict(X_test_mfcc)

# ---- CNN model (on spectrograms) ----
X_spec = np.load("data/spectrograms_X.npy")
y_spec = np.load("data/spectrograms_y.npy")

encoder = LabelEncoder()
y_spec_encoded = encoder.fit_transform(y_spec)
X_spec = X_spec[..., np.newaxis]

X_train_spec, X_test_spec, y_train_spec, y_test_spec = train_test_split(
    X_spec, y_spec_encoded, test_size=0.2, random_state=42
)

cnn_model = keras.models.load_model("models/cnn_model.keras")
cnn_preds_prob = cnn_model.predict(X_test_spec)
cnn_preds = (cnn_preds_prob > 0.5).astype(int).flatten()

# ---- Confusion matrices ----
cm_baseline = confusion_matrix(y_test_mfcc, baseline_preds, labels=["real", "fake"])
cm_cnn = confusion_matrix(y_test_spec, cnn_preds)

print("Baseline (Logistic Regression) confusion matrix:")
print("             Predicted Real  Predicted Fake")
print(f"Actual Real       {cm_baseline[0][0]:<15} {cm_baseline[0][1]}")
print(f"Actual Fake       {cm_baseline[1][0]:<15} {cm_baseline[1][1]}")

print(f"\nLabel mapping for CNN: {list(encoder.classes_)}")
print("CNN confusion matrix:")
print(f"             Predicted {encoder.classes_[0]}  Predicted {encoder.classes_[1]}")
print(f"Actual {encoder.classes_[0]}       {cm_cnn[0][0]:<15} {cm_cnn[0][1]}")
print(f"Actual {encoder.classes_[1]}       {cm_cnn[1][0]:<15} {cm_cnn[1][1]}")
