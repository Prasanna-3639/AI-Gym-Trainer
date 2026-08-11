import cv2
import mediapipe as mp
import os
import csv

# ==========================================
# AI GYM TRAINER - MULTI-EXERCISE
# LANDMARK EXTRACTION
# ==========================================

INPUT_ROOT = "dataset/frames"
OUTPUT_FILE = "dataset/gym_landmarks.csv"

EXERCISES = [
    "squat",
    "push-up",
    "shoulder press",
    "barbell biceps curl"
]

# ==========================================
# MEDIAPIPE POSE
# ==========================================

mp_pose = mp.solutions.pose

pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=1,
    min_detection_confidence=0.5
)

# ==========================================
# CSV COLUMNS
# ==========================================

columns = ["exercise"]

for i in range(33):

    columns.extend([
        f"x_{i}",
        f"y_{i}",
        f"z_{i}",
        f"visibility_{i}"
    ])

rows = []

total_frames = 0
successful_detections = 0
failed_detections = 0

print("=" * 60)
print("AI GYM TRAINER - MULTI-EXERCISE LANDMARK EXTRACTION")
print("=" * 60)

# ==========================================
# PROCESS EACH EXERCISE
# ==========================================

for exercise in EXERCISES:

    exercise_folder = os.path.join(
        INPUT_ROOT,
        exercise
    )

    print("\n" + "=" * 60)
    print("EXERCISE:", exercise.upper())
    print("=" * 60)

    if not os.path.exists(exercise_folder):

        print("WARNING: Folder not found:")
        print(exercise_folder)

        continue

    sequence_folders = sorted([
        f for f in os.listdir(exercise_folder)
        if os.path.isdir(
            os.path.join(exercise_folder, f)
        )
    ])

    print(
        "Video folders:",
        len(sequence_folders)
    )

    exercise_success = 0
    exercise_failed = 0

    # ==========================================
    # PROCESS EACH VIDEO
    # ==========================================

    for sequence in sequence_folders:

        sequence_path = os.path.join(
            exercise_folder,
            sequence
        )

        image_files = sorted([
            f for f in os.listdir(sequence_path)
            if f.lower().endswith(
                (".jpg", ".jpeg", ".png")
            )
        ])

        for image_file in image_files:

            image_path = os.path.join(
                sequence_path,
                image_file
            )

            image = cv2.imread(image_path)

            if image is None:
                continue

            total_frames += 1

            # BGR → RGB
            image_rgb = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )

            # Pose detection
            results = pose.process(
                image_rgb
            )

            if results.pose_landmarks:

                row = [exercise]

                for landmark in (
                    results.pose_landmarks.landmark
                ):

                    row.extend([
                        landmark.x,
                        landmark.y,
                        landmark.z,
                        landmark.visibility
                    ])

                rows.append(row)

                successful_detections += 1
                exercise_success += 1

            else:

                failed_detections += 1
                exercise_failed += 1

    print(
        "Successful detections:",
        exercise_success
    )

    print(
        "Failed detections:",
        exercise_failed
    )

# ==========================================
# SAVE CSV
# ==========================================

print("\n" + "=" * 60)
print("LANDMARK EXTRACTION COMPLETED")
print("=" * 60)

print(
    "Total frames processed:",
    total_frames
)

print(
    "Successful detections:",
    successful_detections
)

print(
    "Failed detections:",
    failed_detections
)

# Save CSV

with open(
    OUTPUT_FILE,
    "w",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow(columns)

    writer.writerows(rows)

print("\nCSV saved successfully!")

print(
    "Location:",
    OUTPUT_FILE
)

print(
    "Rows in CSV:",
    len(rows)
)

print("=" * 60)

pose.close()