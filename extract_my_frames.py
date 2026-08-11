import cv2
import os

# ============================================================
# AI GYM TRAINER - MY RECORDINGS FRAME EXTRACTION
# ============================================================

INPUT_FOLDER = "dataset/raw/my_recordings/squat"
OUTPUT_FOLDER = "dataset/frames/my_recordings/squat"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

video_files = [
    f for f in os.listdir(INPUT_FOLDER)
    if f.lower().endswith((".mp4", ".avi", ".mov"))
]

print("=" * 60)
print("AI GYM TRAINER - MY RECORDINGS")
print("=" * 60)

print(f"\nVideos found: {len(video_files)}")

total_saved = 0

for index, video_name in enumerate(video_files, 1):

    print(f"\n[{index}/{len(video_files)}] Processing:")
    print(video_name)

    video_path = os.path.join(INPUT_FOLDER, video_name)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("ERROR: Could not open video")
        continue

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Save approximately 10% of the frames
    step = max(1, total_frames // 30)

    video_id = os.path.splitext(video_name)[0]

    video_output_folder = os.path.join(
        OUTPUT_FOLDER,
        video_id
    )

    os.makedirs(video_output_folder, exist_ok=True)

    frame_number = 0
    saved = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        if frame_number % step == 0:

            output_path = os.path.join(
                video_output_folder,
                f"frame_{saved:05d}.jpg"
            )

            cv2.imwrite(output_path, frame)

            saved += 1
            total_saved += 1

        frame_number += 1

    cap.release()

    print(f"Original frames: {total_frames}")
    print(f"Saved frames   : {saved}")


print("\n" + "=" * 60)
print("MY RECORDINGS FRAME EXTRACTION COMPLETED")
print("=" * 60)

print(f"Total frames saved: {total_saved}")
print("=" * 60)