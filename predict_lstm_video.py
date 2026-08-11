import cv2
import mediapipe as mp
import numpy as np
import joblib
import tensorflow as tf
from collections import Counter, deque

# ============================================================
# AI GYM TRAINER - LSTM VIDEO PREDICTION
# ============================================================

MODEL_PATH = "models/lstm_exercise_model.keras"
SCALER_PATH = "models/lstm_scaler.pkl"
ENCODER_PATH = "models/lstm_label_encoder.pkl"

# CORRECT ORIGINAL VIDEO PATH
VIDEO_PATH = "dataset/raw/my_recordings/squat/squat_1.mp4"

SEQUENCE_LENGTH = 10

print("=" * 60)
print("AI GYM TRAINER - LSTM VIDEO PREDICTION")
print("=" * 60)

# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading LSTM model...")

model = tf.keras.models.load_model(MODEL_PATH)

scaler = joblib.load(SCALER_PATH)

label_encoder = joblib.load(ENCODER_PATH)

print("LSTM model loaded successfully!")

print("\nClasses:")
print(label_encoder.classes_)

# ============================================================
# MEDIAPIPE
# ============================================================

mp_pose = mp.solutions.pose

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ============================================================
# LANDMARK IDs
# ============================================================

NOSE = 0

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12

LEFT_ELBOW = 13
RIGHT_ELBOW = 14

LEFT_WRIST = 15
RIGHT_WRIST = 16

LEFT_HIP = 23
RIGHT_HIP = 24

LEFT_KNEE = 25
RIGHT_KNEE = 26

LEFT_ANKLE = 27
RIGHT_ANKLE = 28


# ============================================================
# GET POINT
# ============================================================

def get_point(landmarks, landmark_id):

    p = landmarks[landmark_id]

    return np.array([
        p.x,
        p.y,
        p.z
    ])


# ============================================================
# CALCULATE ANGLE
# ============================================================

def calculate_angle(a, b, c):

    ba = a - b
    bc = c - b

    denominator = (
        np.linalg.norm(ba) *
        np.linalg.norm(bc)
    )

    if denominator < 1e-8:
        return 0.0

    cosine_angle = (
        np.dot(ba, bc) /
        denominator
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
# EXTRACT 50 FEATURES
# ============================================================

def extract_features(landmarks):

    nose = get_point(
        landmarks,
        NOSE
    )

    left_shoulder = get_point(
        landmarks,
        LEFT_SHOULDER
    )

    right_shoulder = get_point(
        landmarks,
        RIGHT_SHOULDER
    )

    left_elbow = get_point(
        landmarks,
        LEFT_ELBOW
    )

    right_elbow = get_point(
        landmarks,
        RIGHT_ELBOW
    )

    left_wrist = get_point(
        landmarks,
        LEFT_WRIST
    )

    right_wrist = get_point(
        landmarks,
        RIGHT_WRIST
    )

    left_hip = get_point(
        landmarks,
        LEFT_HIP
    )

    right_hip = get_point(
        landmarks,
        RIGHT_HIP
    )

    left_knee = get_point(
        landmarks,
        LEFT_KNEE
    )

    right_knee = get_point(
        landmarks,
        RIGHT_KNEE
    )

    left_ankle = get_point(
        landmarks,
        LEFT_ANKLE
    )

    right_ankle = get_point(
        landmarks,
        RIGHT_ANKLE
    )

    # --------------------------------------------------------
    # BODY CENTER
    # --------------------------------------------------------

    hip_center = (
        left_hip +
        right_hip
    ) / 2

    shoulder_center = (
        left_shoulder +
        right_shoulder
    ) / 2

    # --------------------------------------------------------
    # BODY SCALE
    # --------------------------------------------------------

    body_scale = np.linalg.norm(
        shoulder_center -
        hip_center
    )

    if body_scale < 1e-6:
        body_scale = 1.0

    # --------------------------------------------------------
    # NORMALIZE LANDMARKS
    # --------------------------------------------------------

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
            point -
            hip_center
        ) / body_scale

        normalized_points.extend(
            normalized.tolist()
        )

    # --------------------------------------------------------
    # BODY ANGLES
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # DISTANCES
    # --------------------------------------------------------

    shoulder_width = (
        np.linalg.norm(
            left_shoulder -
            right_shoulder
        ) / body_scale
    )

    hip_width = (
        np.linalg.norm(
            left_hip -
            right_hip
        ) / body_scale
    )

    knee_width = (
        np.linalg.norm(
            left_knee -
            right_knee
        ) / body_scale
    )

    # --------------------------------------------------------
    # FINAL 50 FEATURES
    # --------------------------------------------------------

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

print("\nOpening video:")
print(VIDEO_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():

    print("\nERROR: Could not open video.")
    print("Check the VIDEO_PATH.")

    pose.close()
    exit()

# ============================================================
# PROCESS VIDEO
# ============================================================

sequence = deque(
    maxlen=SEQUENCE_LENGTH
)

prediction_history = []

total_frames = 0
successful_detections = 0

print("\nProcessing video...")

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

    if not results.pose_landmarks:
        continue

    successful_detections += 1

    features = extract_features(
        results.pose_landmarks.landmark
    )

    sequence.append(features)

    # --------------------------------------------------------
    # PREDICT EVERY 10 FRAMES
    # --------------------------------------------------------

    if len(sequence) == SEQUENCE_LENGTH:

        seq_array = np.array(
            sequence,
            dtype=np.float32
        )

        # Apply the SAME scaler used during training
        seq_scaled = scaler.transform(
            seq_array
        )

        # Add batch dimension
        seq_scaled = np.expand_dims(
            seq_scaled,
            axis=0
        )

        probabilities = model.predict(
            seq_scaled,
            verbose=0
        )[0]

        predicted_index = np.argmax(
            probabilities
        )

        predicted_class = (
            label_encoder.inverse_transform(
                [predicted_index]
            )[0]
        )

        prediction_history.append(
            predicted_class
        )


# ============================================================
# CLOSE
# ============================================================

cap.release()
pose.close()

# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 60)
print("LSTM VIDEO PREDICTION RESULTS")
print("=" * 60)

print(
    "\nTotal frames processed    :",
    total_frames
)

print(
    "Frames with pose detected :",
    successful_detections
)

print(
    "Predictions generated     :",
    len(prediction_history)
)

# ============================================================
# PREDICTION COUNTS
# ============================================================

if len(prediction_history) > 0:

    counts = Counter(
        prediction_history
    )

    print("\nPrediction counts:")

    for exercise, count in counts.most_common():

        percentage = (
            count /
            len(prediction_history)
        ) * 100

        print(
            f"{exercise:25s}: "
            f"{count:4d} "
            f"({percentage:.1f}%)"
        )

    # --------------------------------------------------------
    # FINAL PREDICTION
    # --------------------------------------------------------

    final_prediction = (
        counts.most_common(1)[0][0]
    )

    print(
        "\nFinal predicted exercise:",
        final_prediction
    )

else:

    print(
        "\nNo predictions generated."
    )

print("\n" + "=" * 60)
print("PREDICTION COMPLETED")
print("=" * 60)