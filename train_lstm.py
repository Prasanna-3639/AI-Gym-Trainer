import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# ============================================================
# AI GYM TRAINER - LSTM MODEL TRAINING
# ============================================================

print("=" * 60)
print("AI GYM TRAINER - LSTM MODEL TRAINING")
print("=" * 60)

# ============================================================
# 1. LOAD SEQUENCES
# ============================================================

X = np.load("dataset/sequence_X.npy")
y = np.load(
    "dataset/sequence_y.npy",
    allow_pickle=True
)

print("\nDataset loaded successfully!")

print("X shape:", X.shape)
print("y shape:", y.shape)

# ============================================================
# 2. LABEL ENCODING
# ============================================================

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

print("\nClasses:")
print(label_encoder.classes_)

# ============================================================
# 3. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

print("\nTraining sequences:", len(X_train))
print("Testing sequences :", len(X_test))

# ============================================================
# 4. FEATURE SCALING
# ============================================================

# Shape:
# samples × timesteps × features

n_train = X_train.shape[0]
n_test = X_test.shape[0]

timesteps = X_train.shape[1]
features = X_train.shape[2]

scaler = StandardScaler()

X_train_2d = X_train.reshape(
    n_train * timesteps,
    features
)

X_test_2d = X_test.reshape(
    n_test * timesteps,
    features
)

X_train_scaled = scaler.fit_transform(
    X_train_2d
)

X_test_scaled = scaler.transform(
    X_test_2d
)

X_train = X_train_scaled.reshape(
    n_train,
    timesteps,
    features
)

X_test = X_test_scaled.reshape(
    n_test,
    timesteps,
    features
)

print("\nFeature scaling completed!")

# ============================================================
# 5. CLASS WEIGHTS
# ============================================================

classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = dict(
    zip(classes, weights)
)

print("\nClass weights:")
print(class_weights)

# ============================================================
# 6. BUILD LSTM MODEL
# ============================================================

print("\nBuilding LSTM model...")

model = Sequential([

    LSTM(
        64,
        input_shape=(
            timesteps,
            features
        ),
        return_sequences=True
    ),

    Dropout(0.3),

    LSTM(
        32
    ),

    Dropout(0.3),

    Dense(
        32,
        activation="relu"
    ),

    Dropout(0.2),

    Dense(
        len(label_encoder.classes_),
        activation="softmax"
    )
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ============================================================
# 7. EARLY STOPPING
# ============================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

# ============================================================
# 8. TRAIN
# ============================================================

print("\nTraining LSTM...")

history = model.fit(

    X_train,
    y_train,

    validation_split=0.20,

    epochs=100,

    batch_size=8,

    class_weight=class_weights,

    callbacks=[
        early_stopping
    ],

    verbose=1
)

# ============================================================
# 9. TEST
# ============================================================

print("\nEvaluating LSTM...")

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print(
    f"\nTest Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

# ============================================================
# 10. PREDICTIONS
# ============================================================

probabilities = model.predict(
    X_test,
    verbose=0
)

predictions = np.argmax(
    probabilities,
    axis=1
)

# ============================================================
# 11. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        target_names=label_encoder.classes_,
        zero_division=0
    )
)

# ============================================================
# 12. CONFUSION MATRIX
# ============================================================

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        predictions
    )
)

# ============================================================
# 13. SAVE MODEL
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
)

MODEL_PATH = (
    "models/lstm_exercise_model.keras"
)

SCALER_PATH = (
    "models/lstm_scaler.pkl"
)

ENCODER_PATH = (
    "models/lstm_label_encoder.pkl"
)

model.save(
    MODEL_PATH
)

joblib.dump(
    scaler,
    SCALER_PATH
)

joblib.dump(
    label_encoder,
    ENCODER_PATH
)

print("\nModel saved:")
print(MODEL_PATH)

print("\nScaler saved:")
print(SCALER_PATH)

print("\nLabel encoder saved:")
print(ENCODER_PATH)

print("\n" + "=" * 60)
print("LSTM TRAINING COMPLETED")
print("=" * 60)