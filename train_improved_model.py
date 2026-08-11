import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ============================================================
# AI GYM TRAINER - IMPROVED MODEL TRAINING
# ============================================================

DATASET_PATH = "dataset/gym_features.csv"
MODEL_PATH = "models/improved_exercise_model.pkl"


print("=" * 60)
print("AI GYM TRAINER - IMPROVED MODEL TRAINING")
print("=" * 60)


# ------------------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------------------

df = pd.read_csv(DATASET_PATH)

print("\nDataset loaded successfully!")

print("Dataset shape:", df.shape)


# ------------------------------------------------------------
# 2. SEPARATE FEATURES AND TARGET
# ------------------------------------------------------------

X = df.drop("exercise", axis=1)

y = df["exercise"]

print("\nFeatures shape:", X.shape)

print("Target shape:", y.shape)

print("\nExercise distribution:")

print(y.value_counts())


# ------------------------------------------------------------
# 3. TRAIN / TEST SPLIT
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))

print("Testing samples :", len(X_test))


# ------------------------------------------------------------
# 4. RANDOM FOREST
# ------------------------------------------------------------

print("\nTraining improved Random Forest...")

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Training completed!")


# ------------------------------------------------------------
# 5. PREDICTION
# ------------------------------------------------------------

y_pred = model.predict(X_test)


# ------------------------------------------------------------
# 6. ACCURACY
# ------------------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n" + "=" * 60)

print("MODEL EVALUATION")

print("=" * 60)

print(
    f"\nAccuracy: {accuracy * 100:.2f}%"
)


# ------------------------------------------------------------
# 7. CLASSIFICATION REPORT
# ------------------------------------------------------------

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)


# ------------------------------------------------------------
# 8. CONFUSION MATRIX
# ------------------------------------------------------------

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ------------------------------------------------------------
# 9. SAVE MODEL
# ------------------------------------------------------------

os.makedirs(
    "models",
    exist_ok=True
)

joblib.dump(
    model,
    MODEL_PATH
)

print("\n" + "=" * 60)

print("IMPROVED MODEL SAVED SUCCESSFULLY")

print("=" * 60)

print("\nLocation:")

print(MODEL_PATH)