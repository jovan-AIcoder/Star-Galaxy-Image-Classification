import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D,
    Flatten, Dense, Dropout
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# =========================
# ==== CONFIGURATION ======
# =========================

IMG_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 1e-3


DENSE_UNITS = [128, 64]
DROPOUT_RATE = 0.3

# =========================
# ===== PATH DATA =========
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_PATH = os.path.join(BASE_DIR, "classifier_model.h5")

# =========================
# === DATA GENERATOR ======
# =========================

datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training",
    shuffle=True
)

val_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    shuffle=False
)

print("Class indices:", train_generator.class_indices)

# =========================
# ====== MODEL ============
# =========================

model = Sequential()

model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 1)))
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(128, (3, 3), activation="relu"))
model.add(MaxPooling2D(2, 2))

model.add(Flatten())

for units in DENSE_UNITS:
    model.add(Dense(units, activation="relu"))
    model.add(Dropout(DROPOUT_RATE))

model.add(Dense(1, activation="sigmoid"))

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =========================
# ====== TRAINING =========
# =========================

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator
)

# =========================
# ===== SAVE MODEL ========
# =========================

model.save(MODEL_PATH)