import cv2
import os

# ==========================================
# AI GYM TRAINER - FRAME EXTRACTION
# ==========================================

INPUT_ROOT = "dataset/raw/videos"
OUTPUT_ROOT = "dataset/frames"

# Four exercise classes
EXERCISES = [
    "squat",
    "push-up",
    "shoulder press",
    "barbell biceps curl"
]

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")

os.makedirs(OUTPUT_ROOT, exist_ok=True)

grand_total_frames = 0

print("=" * 60)
print("AI GYM TRAINER - FRAME EXTRACTION")
print("=" * 60)

for exercise in EXERCISES:

    input_folder = os.path.join(INPUT_ROOT, exercise)
    output_folder = os.path.join(OUTPUT_ROOT, exercise)

    os.makedirs(output_folder, exist_ok=True)

    if not os.path.exists(input_folder):
        print(f"\nWARNING: Folder not found: {input_folder}")
        continue

    videos = [
        f for f in os.listdir(input_folder)
        if f.lower().endswith(VIDEO_EXTENSIONS)
    ]

    print("\n" + "=" * 60)
    print(f"EXERCISE: {exercise.upper()}")
    print(f"Videos found: {len(videos)}")
    print("=" * 60)

    exercise_total = 0

    for video_index, video_name in enumerate(videos, start=1):

        video_path = os.path.join(input_folder, video_name)

        video_id = os.path.splitext(video_name)[0]

        video_output = os.path.join(
            output_folder,
            video_id
        )

        os.makedirs(video_output, exist_ok=True)

        print(
            f"[{video_index}/{len(videos)}] "
            f"{video_name}"
        )

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print("  ERROR: Could not open video")
            continue

        frame_number = 0
        saved_frames = 0

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            # Save every 10th frame
            if frame_number % 10 == 0:

                output_path = os.path.join(
                    video_output,
                    f"frame_{saved_frames:05d}.jpg"
                )

                cv2.imwrite(
                    output_path,
                    frame
                )

                saved_frames += 1
                exercise_total += 1
                grand_total_frames += 1

            frame_number += 1

        cap.release()

        print(
            f"  Original frames: {frame_number}"
        )

        print(
            f"  Saved frames   : {saved_frames}"
        )

    print(
        f"\nTotal {exercise} frames saved: "
        f"{exercise_total}"
    )


print("\n" + "=" * 60)
print("FRAME EXTRACTION COMPLETED")
print("=" * 60)

print(
    "Total frames saved:",
    grand_total_frames
)

print("=" * 60)