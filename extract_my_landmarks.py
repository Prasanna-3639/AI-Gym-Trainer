import cv2
import mediapipe as mp
import pandas as pd
import os

# ============================================================
# AI GYM TRAINER - MY RECORDINGS LANDMARK EXTRACTION
# ============================================================

INPUT_FOLDER = "dataset/frames/my_recordings/squat"
OUTPUT_FILE = "dataset/my_squat_landmarks.csv"

# ============================================================
# INITIALIZE MEDIAPIPE
# ============================================================

mp_pose = mp.solutions.pose

pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=1,
    min_detection_confidence=0.5
)

# ============================================================
# STORAGE
# ============================================================

rows = []

total_frames = 0
successful = 0
failed = 0

print("=" * 60)
print("AI GYM TRAINER - MY RECORDINGS LANDMARK EXTRACTION")
print("=" * 60)

# ============================================================
# PROCESS VIDEO FOLDERS
# ============================================================

video_folders = sorted(os.listdir(INPUT_FOLDER))

print(f"\nVideo folders: {len(video_folders)}")

for video_folder in video_folders:

    folder_path = os.path.join(INPUT_FOLDER, video_folder)

    if not os.path.isdir(folder_path):
        continue

    print(f"\nProcessing: {video_folder}")

    frame_files = sorted([
        f for f in os.listdir(folder_path)
        if f.lower().endswith(".jpg")
    ])

    for frame_file in frame_files:

        total_frames += 1

        image_path = os.path.join(folder_path, frame_file)

        image = cv2.imread(image_path)

        if image is None:
            failed += 1
            continue

        # BGR → RGB
        rgb_image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        results = pose.process(rgb_image)

        if results.pose_landmarks:

            landmarks = []

            for landmark in results.pose_landmarks.landmark:

                landmarks.extend([
                    landmark.x,
                    landmark.y,
                    landmark.z,
                    landmark.visibility
                ])

            row = ["squat"] + landmarks

            rows.append(row)

            successful += 1

        else:
            failed += 1

# ============================================================
# CREATE COLUMN NAMES
# ============================================================

columns = ["exercise"]

for i in range(33):

    columns.extend([
        f"x_{i}",
        f"y_{i}",
        f"z_{i}",
        f"visibility_{i}"
    ])

# ============================================================
# CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(
    rows,
    columns=columns
)

# ============================================================
# SAVE CSV
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)

pose.close()

# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 60)
print("MY LANDMARK EXTRACTION COMPLETED")
print("=" * 60)

print(f"Total frames processed     : {total_frames}")
print(f"Successful detections      : {successful}")
print(f"Failed detections          : {failed}")

print(f"\nDataset shape: {df.shape}")

print(f"\nCSV saved successfully!")
print(f"Location: {OUTPUT_FILE}")

print("=" * 60)