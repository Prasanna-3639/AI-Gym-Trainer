import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from keras.callbacks import EarlyStopping
# ============================================================
# AI GYM TRAINER - ANN MODEL TRAINING
# ============================================================

DATASET_PATH = "dataset/combined_features.csv"

MODEL_PATH = "models/ann_exercise_model.keras"
SCALER_PATH = "models/ann_scaler.pkl"
ENCODER_PATH = "models/ann_label_encoder.pkl"


print("=" * 60)
print("AI GYM TRAINER - ANN MODEL TRAINING")
print("=" * 60)


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv(DATASET_PATH)

print("\nDataset loaded successfully!")

print("Dataset shape:", df.shape)


# ============================================================
# 2. FEATURES AND TARGET
# ============================================================

X = df.drop("exercise", axis=1)

y = df["exercise"]

print("\nFeatures:", X.shape)

print("Target:", y.shape)

print("\nExercise distribution:")

print(y.value_counts())


# ============================================================
# 3. LABEL ENCODING
# ============================================================

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

print("\nClasses:")

print(label_encoder.classes_)


# ============================================================
# 4. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

print("\nTraining samples:", len(X_train))

print("Testing samples :", len(X_test))


# ============================================================
# 5. FEATURE SCALING
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# ============================================================
# 6. BUILD ANN
# ============================================================

print("\nBuilding ANN model...")

model = Sequential([

    Input(shape=(X_train_scaled.shape[1],)),

    Dense(
        128,
        activation="relu"
    ),

    Dropout(0.30),

    Dense(
        64,
        activation="relu"
    ),

    Dropout(0.20),

    Dense(
        32,
        activation="relu"
    ),

    Dense(
        len(label_encoder.classes_),
        activation="softmax"
    )
])


# ============================================================
# 7. COMPILE
# ============================================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


model.summary()


# ============================================================
# 8. EARLY STOPPING
# ============================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)


# ============================================================
# 9. TRAIN
# ============================================================

print("\nTraining ANN...")

history = model.fit(
    X_train_scaled,
    y_train,

    validation_split=0.20,

    epochs=100,

    batch_size=32,

    callbacks=[early_stopping],

    verbose=1
)


# ============================================================
# 10. EVALUATION
# ============================================================

print("\n" + "=" * 60)

print("ANN MODEL EVALUATION")

print("=" * 60)


test_loss, test_accuracy = model.evaluate(
    X_test_scaled,
    y_test,
    verbose=0
)

print(
    f"\nTest Accuracy: {test_accuracy * 100:.2f}%"
)


# ============================================================
# 11. PREDICTIONS
# ============================================================

y_probability = model.predict(
    X_test_scaled,
    verbose=0
)

y_pred = np.argmax(
    y_probability,
    axis=1
)


# ============================================================
# 12. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_
    )
)


# ============================================================
# 13. CONFUSION MATRIX
# ============================================================

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ============================================================
# 14. SAVE MODEL
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
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


# ============================================================
# 15. FINAL RESULT
# ============================================================

print("\n" + "=" * 60)

print("ANN TRAINING COMPLETED")

print("=" * 60)

print("\nModel saved:")
print(MODEL_PATH)

print("\nScaler saved:")
print(SCALER_PATH)

print("\nLabel encoder saved:")
print(ENCODER_PATH)

print("=" * 60)