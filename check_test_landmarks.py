import cv2
import mediapipe as mp
import pandas as pd

VIDEO_PATH = r"C:\Users\hp\Downloads\archive (8)\my_test_video_1\squat\squat_4.mp4"

mp_pose = mp.solutions.pose

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(VIDEO_PATH)

rows = []

frame_count = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = pose.process(rgb)

    if results.pose_landmarks:

        features = []

        for lm in results.pose_landmarks.landmark:

            features.extend([
                lm.x,
                lm.y,
                lm.z,
                lm.visibility
            ])

        rows.append(features)

    # Only first 50 frames
    if frame_count >= 50:
        break


cap.release()
pose.close()


df = pd.DataFrame(rows)

print("=" * 60)
print("TEST VIDEO LANDMARK INSPECTION")
print("=" * 60)

print("Frames checked       :", frame_count)
print("Successful landmarks:", len(rows))
print("Features per frame   :", df.shape[1])

print("\nFirst frame landmarks:")
print(df.iloc[0].values)

print("\nMean landmark values:")
print(df.mean().head(20))

print("\nMin values:")
print(df.min().head(20))

print("\nMax values:")
print(df.max().head(20))

print("=" * 60)