import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib


# ============================================
# AI GYM TRAINER - MODEL TRAINING
# ============================================

print("=" * 60)
print("AI GYM TRAINER - MODEL TRAINING")
print("=" * 60)


# --------------------------------------------
# 1. Load landmark dataset
# --------------------------------------------

DATASET_PATH = "dataset/gym_landmarks.csv"

df = pd.read_csv(DATASET_PATH)

print("\nDataset loaded successfully!")
print("Dataset shape:", df.shape)


# --------------------------------------------
# 2. Separate features and target
# --------------------------------------------

X = df.drop("exercise", axis=1)
y = df["exercise"]

print("\nFeatures:", X.shape)
print("Target:", y.shape)

print("\nExercise classes:")
print(y.value_counts())


# --------------------------------------------
# 3. Train-test split
# --------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))


# --------------------------------------------
# 4. Create Random Forest model
# --------------------------------------------

print("\nTraining Random Forest model...")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Training completed!")


# --------------------------------------------
# 5. Prediction
# --------------------------------------------

y_pred = model.predict(X_test)


# --------------------------------------------
# 6. Evaluation
# --------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

print(f"\nAccuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# --------------------------------------------
# 7. Save model
# --------------------------------------------

MODEL_FOLDER = "models"

os.makedirs(MODEL_FOLDER, exist_ok=True)

MODEL_PATH = os.path.join(
    MODEL_FOLDER,
    "exercise_model.pkl"
)

joblib.dump(model, MODEL_PATH)

print("\n" + "=" * 60)
print("MODEL SAVED SUCCESSFULLY")
print("=" * 60)

print("\nLocation:", MODEL_PATH)