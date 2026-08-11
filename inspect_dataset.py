import os

# Dataset location
DATASET_PATH = "dataset/raw"

# Exercise classes
EXERCISES = [
    "barbell biceps curl",
    "push-up",
    "shoulder press",
    "squat"
]

# Image extensions
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")

print("=" * 60)
print("          AI GYM TRAINER - DATASET INSPECTION")
print("=" * 60)

total_original_images = 0
total_label_images = 0

for exercise in EXERCISES:

    exercise_path = os.path.join(DATASET_PATH, exercise)

    print("\n" + "=" * 60)
    print(f"EXERCISE: {exercise.upper()}")
    print("=" * 60)

    if not os.path.exists(exercise_path):
        print("Folder not found!")
        continue

    sequence_folders = [
        folder for folder in os.listdir(exercise_path)
        if os.path.isdir(os.path.join(exercise_path, folder))
    ]

    print(f"Sequence Folders : {len(sequence_folders)}")

    original_count = 0
    label_count = 0

    for folder in sequence_folders:

        folder_path = os.path.join(exercise_path, folder)

        files = os.listdir(folder_path)

        for file in files:

            if file.lower().endswith(IMAGE_EXTENSIONS):

                # Segmentation/label image
                if ".cseg." in file.lower():
                    label_count += 1

                # Original image
                else:
                    original_count += 1

    print(f"Original Images  : {original_count}")
    print(f"Label Images     : {label_count}")

    total_original_images += original_count
    total_label_images += label_count


print("\n" + "=" * 60)
print("                 DATASET SUMMARY")
print("=" * 60)

print(f"Total Original Images : {total_original_images}")
print(f"Total Label Images    : {total_label_images}")

print("=" * 60)