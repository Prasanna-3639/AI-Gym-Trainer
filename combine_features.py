import pandas as pd

# ============================================================
# AI GYM TRAINER - COMBINE DATASETS
# ============================================================

ORIGINAL_FILE = "dataset/gym_features.csv"
MY_FILE = "dataset/my_squat_features.csv"
OUTPUT_FILE = "dataset/combined_features.csv"

print("=" * 60)
print("AI GYM TRAINER - COMBINING DATASETS")
print("=" * 60)

# Load datasets

original_df = pd.read_csv(ORIGINAL_FILE)
my_df = pd.read_csv(MY_FILE)

print("\nOriginal dataset:")
print(original_df.shape)

print("\nMy squat dataset:")
print(my_df.shape)

# Check columns

if list(original_df.columns) != list(my_df.columns):

    print("\nERROR: Column mismatch!")
    print("Datasets cannot be combined.")

    exit()

# Combine

combined_df = pd.concat(
    [original_df, my_df],
    ignore_index=True
)

# Remove duplicate rows if any

combined_df = combined_df.drop_duplicates()

# Save

combined_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 60)
print("COMBINATION COMPLETED")
print("=" * 60)

print("\nCombined dataset shape:")
print(combined_df.shape)

print("\nExercise distribution:")
print(
    combined_df["exercise"].value_counts()
)

print("\nSaved to:")
print(OUTPUT_FILE)

print("=" * 60)