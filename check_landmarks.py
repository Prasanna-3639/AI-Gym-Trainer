import pandas as pd

# Load landmark dataset
df = pd.read_csv("dataset/squat_landmarks.csv")

print("=" * 60)
print("AI GYM TRAINER - LANDMARK DATASET")
print("=" * 60)

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum().sum())

print("\nExercise Classes:")
print(df["exercise"].value_counts())

print("\n" + "=" * 60)