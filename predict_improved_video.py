import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import joblib
from collections import Counter


# ============================================================
# AI GYM TRAINER - IMPROVED MODEL VIDEO TEST
# ============================================================

MODEL_PATH = "models/improved_exercise_model.pkl"

VIDEO_PATH = r"C:\Users\hp\Downloads\archive (8)\my_test_video_1\squat\squat_4.mp4"


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("AI GYM TRAINER - IMPROVED MODEL TEST")
print("=" * 60)

model = joblib.load(MODEL_PATH)

print("\nImproved model loaded successfully!")

print("Classes:")
print(model.classes_)


# ============================================================
# MEDIAPIPE
# ============================================================

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# ============================================================
# ANGLE FUNCTION
# ============================================================

def calculate_angle(a, b, c):

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (
        np.linalg.norm(ba) *
        np.linalg.norm(bc) + 1e-8
    )

    cosine_angle = np.clip(
        cosine_angle,
        -1.0,
        1.0
    )

    return np.degrees(
        np.arccos(cosine_angle)
    )


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def create_features(landmarks):

    def point(index):

        lm = landmarks[index]

        return np.array([
            lm.x,
            lm.y,
            lm.z
        ])

    nose = point(0)

    left_shoulder = point(11)
    right_shoulder = point(12)

    left_elbow = point(13)
    right_elbow = point(14)

    left_wrist = point(15)
    right_wrist = point(16)

    left_hip = point(23)
    right_hip = point(24)

    left_knee = point(25)
    right_knee = point(26)

    left_ankle = point(27)
    right_ankle = point(28)

    hip_center = (
        left_hip + right_hip
    ) / 2

    shoulder_center = (
        left_shoulder + right_shoulder
    ) / 2

    body_scale = np.linalg.norm(
        shoulder_center - hip_center
    )

    if body_scale < 1e-6:
        body_scale = 1.0

    points = [
        nose,
        left_shoulder,
        right_shoulder,
        left_elbow,
        right_elbow,
        left_wrist,
        right_wrist,
        left_hip,
        right_hip,
        left_knee,
        right_knee,
        left_ankle,
        right_ankle
    ]

    normalized_points = []

    for p in points:

        normalized = (
            p - hip_center
        ) / body_scale

        normalized_points.extend(
            normalized.tolist()
        )

    # Angles

    left_elbow_angle = calculate_angle(
        left_shoulder,
        left_elbow,
        left_wrist
    )

    right_elbow_angle = calculate_angle(
        right_shoulder,
        right_elbow,
        right_wrist
    )

    left_knee_angle = calculate_angle(
        left_hip,
        left_knee,
        left_ankle
    )

    right_knee_angle = calculate_angle(
        right_hip,
        right_knee,
        right_ankle
    )

    left_hip_angle = calculate_angle(
        left_shoulder,
        left_hip,
        left_knee
    )

    right_hip_angle = calculate_angle(
        right_shoulder,
        right_hip,
        right_knee
    )

    left_shoulder_angle = calculate_angle(
        left_elbow,
        left_shoulder,
        left_hip
    )

    right_shoulder_angle = calculate_angle(
        right_elbow,
        right_shoulder,
        right_hip
    )

    # Distances

    shoulder_width = (
        np.linalg.norm(
            left_shoulder - right_shoulder
        ) / body_scale
    )

    hip_width = (
        np.linalg.norm(
            left_hip - right_hip
        ) / body_scale
    )

    knee_width = (
        np.linalg.norm(
            left_knee - right_knee
        ) / body_scale
    )

    features = normalized_points + [

        left_elbow_angle,
        right_elbow_angle,

        left_knee_angle,
        right_knee_angle,

        left_hip_angle,
        right_hip_angle,

        left_shoulder_angle,
        right_shoulder_angle,

        shoulder_width,
        hip_width,
        knee_width
    ]

    return features


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():

    print("\nERROR: Could not open video!")
    print(VIDEO_PATH)
    exit()

print("\nVideo opened successfully!")
print("Press Q to quit.")


# ============================================================
# PREDICTIONS
# ============================================================

prediction_counts = Counter()

total_frames = 0
successful_frames = 0


# ============================================================
# PROCESS VIDEO
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    total_frames += 1

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = pose.process(rgb)

    prediction = "No pose detected"

    if results.pose_landmarks:

        successful_frames += 1

        features = create_features(
            results.pose_landmarks.landmark
        )

        feature_df = pd.DataFrame(
            [features],
            columns=model.feature_names_in_
        )

        prediction = model.predict(
            feature_df
        )[0]

        prediction_counts[
            prediction
        ] += 1

        # Draw pose

        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

    # Display

    cv2.putText(
        frame,
        f"Exercise: {prediction}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow(
        "AI Gym Trainer - Improved Model",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == 27:
        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()
cv2.destroyAllWindows()
pose.close()


# ============================================================
# FINAL RESULT
# ============================================================

print("\n" + "=" * 60)
print("VIDEO ANALYSIS COMPLETED")
print("=" * 60)

print("Total frames processed    :", total_frames)
print("Frames with pose detected :", successful_frames)

print("\nPrediction counts:")

for exercise, count in prediction_counts.most_common():

    percentage = (
        count / successful_frames
    ) * 100

    print(
        f"{exercise:<25} : "
        f"{count:4d} "
        f"({percentage:.1f}%)"
    )


if prediction_counts:

    final_prediction = (
        prediction_counts
        .most_common(1)[0][0]
    )

    print("\n" + "=" * 60)

    print(
        "FINAL EXERCISE:",
        final_prediction.upper()
    )

    print("=" * 60)

else:

    print("\nNo pose detected.")


print("\nDone!")