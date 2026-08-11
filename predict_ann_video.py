import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import joblib
from tensorflow import keras

# ============================================================
# AI GYM TRAINER - ANN TEST VIDEO
# ============================================================

MODEL_PATH = "models/ann_exercise_model.keras"
SCALER_PATH = "models/ann_scaler.pkl"
ENCODER_PATH = "models/ann_label_encoder.pkl"

# CHANGE THIS TO YOUR ACTUAL TEST VIDEO
VIDEO_PATH = r"C:\Users\hp\Downloads\archive (8)\my_test_video_1\squat\squat_4.mp4"



print("=" * 60)
print("AI GYM TRAINER - ANN TEST VIDEO")
print("=" * 60)


# ============================================================
# 1. LOAD MODEL
# ============================================================

print("\nLoading ANN model...")

model = keras.models.load_model(MODEL_PATH)

scaler = joblib.load(SCALER_PATH)

label_encoder = joblib.load(ENCODER_PATH)

print("ANN model loaded successfully!")

print("Classes:")
print(label_encoder.classes_)


# ============================================================
# 2. INITIALIZE MEDIAPIPE
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
# 3. OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():

    print("\nERROR: Could not open video!")

    print("Check VIDEO_PATH:")
    print(VIDEO_PATH)

    exit()


print("\nVideo opened successfully!")
print("Press Q to quit.")


# ============================================================
# LANDMARK FUNCTION
# ============================================================

def get_point(landmarks, landmark_id):

    landmark = landmarks[landmark_id]

    return np.array([
        landmark.x,
        landmark.y,
        landmark.z
    ])


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

    nose = get_point(landmarks, 0)

    left_shoulder = get_point(landmarks, 11)
    right_shoulder = get_point(landmarks, 12)

    left_elbow = get_point(landmarks, 13)
    right_elbow = get_point(landmarks, 14)

    left_wrist = get_point(landmarks, 15)
    right_wrist = get_point(landmarks, 16)

    left_hip = get_point(landmarks, 23)
    right_hip = get_point(landmarks, 24)

    left_knee = get_point(landmarks, 25)
    right_knee = get_point(landmarks, 26)

    left_ankle = get_point(landmarks, 27)
    right_ankle = get_point(landmarks, 28)


    # Body centers

    hip_center = (
        left_hip + right_hip
    ) / 2

    shoulder_center = (
        left_shoulder + right_shoulder
    ) / 2


    # Body scale

    body_scale = np.linalg.norm(
        shoulder_center - hip_center
    )

    if body_scale < 1e-6:
        body_scale = 1.0


    # Normalize landmarks

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

    for point in points:

        normalized = (
            point - hip_center
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


    # Final 50 features

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
# PROCESS VIDEO
# ============================================================

total_frames = 0
pose_detected = 0

prediction_counts = {}

while True:

    ret, frame = cap.read()

    if not ret:
        break

    total_frames += 1

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = pose.process(rgb_frame)

    prediction = "No pose"

    if results.pose_landmarks:

        pose_detected += 1

        landmarks = results.pose_landmarks.landmark

        features = create_features(
            landmarks
        )

        # Convert to numpy

        features = np.array(
            features
        ).reshape(1, -1)

        # Scale

        features_scaled = scaler.transform(
            features
        )

        # ANN prediction

        probabilities = model.predict(
            features_scaled,
            verbose=0
        )[0]

        predicted_index = np.argmax(
            probabilities
        )

        prediction = label_encoder.inverse_transform(
            [predicted_index]
        )[0]

        # Count prediction

        prediction_counts[prediction] = (
            prediction_counts.get(
                prediction,
                0
            ) + 1
        )

        # Draw landmarks

        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )


    # Display prediction

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
        "AI Gym Trainer - ANN",
        frame
    )


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()
cv2.destroyAllWindows()
pose.close()


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 60)
print("ANN VIDEO PREDICTION RESULTS")
print("=" * 60)

print(
    f"Total frames processed    : {total_frames}"
)

print(
    f"Frames with pose detected : {pose_detected}"
)

print("\nPrediction counts:")

for exercise, count in sorted(
    prediction_counts.items(),
    key=lambda x: x[1],
    reverse=True
):

    percentage = (
        count / pose_detected * 100
        if pose_detected > 0
        else 0
    )

    print(
        f"{exercise:<25}: "
        f"{count:4d} "
        f"({percentage:.1f}%)"
    )

print("=" * 60)