import os
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc
)

# =========================
# CONFIGURATION
# =========================
IMG_SIZE = (64, 64)
BATCH_SIZE = 32

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_PATH = os.path.join(BASE_DIR, "classifier_model.h5")

CLASS_NAMES = ["star", "galaxy"]

# =========================
# LOAD MODEL
# =========================
model = load_model(MODEL_PATH)
model.summary()
# =========================
# DATA GENERATOR (TEST)
# =========================
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

test_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    shuffle=False
)

# =========================
# PREDICTION
# =========================
y_true = test_generator.classes
y_prob = model.predict(test_generator).ravel()
y_pred = (y_prob >= 0.5).astype(int)

# =========================
# METRICS
# =========================
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("\n===== MODEL PERFORMANCE =====")
print(f"Accuracy  : {acc:.4f}")
print(f"Precision : {prec:.4f}")
print(f"Recall    : {rec:.4f}")
print(f"F1-score  : {f1:.4f}")

# =========================
# CONFUSION MATRIX
# =========================
cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(
    y_true,
    y_pred,
    target_names=CLASS_NAMES
))

# =========================
# ROC CURVE & AUC
# =========================
fpr, tpr, _ = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Star vs Galaxy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("roc_curve.png")
plt.show()

print(f"\nROC AUC Score: {roc_auc:.4f}")
