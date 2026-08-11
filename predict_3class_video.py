import cv2
import numpy as np
import tensorflow as tf
import joblib
import mediapipe as mp
import os


# ============================================================
# AI GYM TRAINER - 3 CLASS LSTM VIDEO PREDICTION
# ACTIVE MOVEMENT VERSION
# ============================================================
VIDEO_PATH = rVIDEO_PATH = r"C:\Users\hp\Desktop\AI_Gym_Tainer\dataset\raw\my_recordings\shoulder press\shoulder press_1.mp4"

MODEL_PATH = r"models\lstm_3class_exercise_model.keras"
SCALER_PATH = r"models\lstm_3class_scaler.pkl"
ENCODER_PATH = r"models\lstm_3class_label_encoder.pkl"

SEQUENCE_LENGTH = 10

# Ignore initial setup/standing portion
IGNORE_INITIAL_PERCENT = 0.15


# ============================================================
# CHECK FILES
# ============================================================

print("=" * 60)
print("AI GYM TRAINER - 3 CLASS VIDEO PREDICTION")
print("=" * 60)

required_files = [
    VIDEO_PATH,
    MODEL_PATH,
    SCALER_PATH,
    ENCODER_PATH
]

for path in required_files:

    if not os.path.exists(path):

        print("\nERROR: File not found:")
        print(path)

        raise SystemExit


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

scaler = joblib.load(
    SCALER_PATH
)

label_encoder = joblib.load(
    ENCODER_PATH
)

print("Model loaded successfully.")

print("\nClasses:")

for i, class_name in enumerate(
    label_encoder.classes_
):

    print(
        f"{i} -> {class_name}"
    )


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
# LANDMARK IDS
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

    return np.array(
        [
            p.x,
            p.y,
            p.z
        ],
        dtype=np.float32
    )


# ============================================================
# CALCULATE ANGLE
# ============================================================

def calculate_angle(a, b, c):

    ba = a - b
    bc = c - b

    denominator = (
        np.linalg.norm(ba)
        *
        np.linalg.norm(bc)
    )

    if denominator < 1e-8:
        return 0.0

    cosine_angle = (
        np.dot(ba, bc)
        /
        denominator
    )

    cosine_angle = np.clip(
        cosine_angle,
        -1.0,
        1.0
    )

    return np.degrees(
        np.arccos(
            cosine_angle
        )
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
    # BODY CENTERS
    # --------------------------------------------------------

    hip_center = (
        left_hip +
        right_hip
    ) / 2.0

    shoulder_center = (
        left_shoulder +
        right_shoulder
    ) / 2.0


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
    # NORMALIZED LANDMARKS
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
    # ANGLES
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
        )
        /
        body_scale
    )

    hip_width = (
        np.linalg.norm(
            left_hip -
            right_hip
        )
        /
        body_scale
    )

    knee_width = (
        np.linalg.norm(
            left_knee -
            right_knee
        )
        /
        body_scale
    )


    # --------------------------------------------------------
    # 50 FEATURES
    # --------------------------------------------------------

    features = (
        normalized_points
        +
        [
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
    )


    if len(features) != 50:

        raise ValueError(
            f"Expected 50 features, "
            f"got {len(features)}"
        )

    return features


# ============================================================
# OPEN VIDEO
# ============================================================

print("\nOpening video:")
print(VIDEO_PATH)

cap = cv2.VideoCapture(
    VIDEO_PATH
)

if not cap.isOpened():

    print(
        "\nERROR: Could not open video."
    )

    pose.close()

    raise SystemExit


# ============================================================
# PROCESS VIDEO
# ============================================================

frame_features = []

total_frames = 0
pose_detected = 0


while True:

    ret, frame = cap.read()

    if not ret:
        break

    total_frames += 1

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = pose.process(
        rgb
    )

    if results.pose_landmarks:

        pose_detected += 1

        features = extract_features(
            results.pose_landmarks.landmark
        )

        frame_features.append(
            features
        )


cap.release()


# ============================================================
# VIDEO INFORMATION
# ============================================================

print(
    "\nTotal frames processed:",
    total_frames
)

print(
    "Frames with pose detected:",
    pose_detected
)


if len(frame_features) < SEQUENCE_LENGTH:

    print(
        "\nERROR: Not enough pose frames."
    )

    pose.close()

    raise SystemExit


# ============================================================
# CREATE SEQUENCES
# ============================================================

sequences = []

for start in range(
    0,
    len(frame_features) - SEQUENCE_LENGTH + 1
):

    sequence = frame_features[
        start:start + SEQUENCE_LENGTH
    ]

    sequences.append(
        sequence
    )


X = np.array(
    sequences,
    dtype=np.float32
)


print(
    "Sequences created:",
    len(X)
)

print(
    "Sequence shape:",
    X.shape
)


# ============================================================
# SCALE
# ============================================================

X_2d = X.reshape(
    -1,
    X.shape[2]
)

X_scaled = scaler.transform(
    X_2d
)

X_scaled = X_scaled.reshape(
    X.shape
)


# ============================================================
# PREDICT
# ============================================================

print(
    "\nGenerating predictions..."
)

probabilities = model.predict(
    X_scaled,
    verbose=0
)

predicted_indices = np.argmax(
    probabilities,
    axis=1
)

predicted_labels = (
    label_encoder.inverse_transform(
        predicted_indices
    )
)


# ============================================================
# RAW PREDICTION COUNTS
# ============================================================

print(
    "\nPrediction counts:"
)

for class_index, class_name in enumerate(
    label_encoder.classes_
):

    count = np.sum(
        predicted_indices ==
        class_index
    )

    percentage = (
        count /
        len(predicted_indices)
        *
        100
    )

    print(
        f"{class_name:22s}: "
        f"{count:4d} "
        f"({percentage:.1f}%)"
    )


# ============================================================
# ACTIVE MOVEMENT SELECTION
# ============================================================

total_sequences = len(
    predicted_indices
)

start_index = int(
    total_sequences *
    IGNORE_INITIAL_PERCENT
)

active_indices = predicted_indices[
    start_index:
]

active_probabilities = probabilities[
    start_index:
]


print(
    "\n" + "=" * 60
)

print(
    "ACTIVE MOVEMENT ANALYSIS"
)

print(
    "=" * 60
)

print(
    "Initial sequences ignored:",
    start_index
)

print(
    "Active sequences analyzed:",
    len(active_indices)
)


# ============================================================
# ACTIVE MAJORITY VOTE
# ============================================================

active_counts = np.bincount(
    active_indices,
    minlength=len(
        label_encoder.classes_
    )
)


print(
    "\nActive movement prediction:"
)


for class_index, class_name in enumerate(
    label_encoder.classes_
):

    count = active_counts[
        class_index
    ]

    percentage = (
        count /
        len(active_indices)
        *
        100
    )

    print(
        f"{class_name:22s}: "
        f"{count:4d} "
        f"({percentage:.1f}%)"
    )


# ============================================================
# ACTIVE AVERAGE PROBABILITIES
# ============================================================

active_average_probabilities = (
    active_probabilities.mean(
        axis=0
    )
)


print(
    "\nActive movement probabilities:"
)


for class_name, probability in zip(

    label_encoder.classes_,

    active_average_probabilities

):

    print(
        f"{class_name:22s}: "
        f"{probability * 100:.2f}%"
    )


# ============================================================
# FINAL PREDICTION
# ============================================================

final_index = np.argmax(
    active_average_probabilities
)

final_prediction = (
    label_encoder.inverse_transform(
        [final_index]
    )[0]
)


print(
    "\nFinal predicted exercise:",
    final_prediction
)


# ============================================================
# FIRST 10 PREDICTIONS
# ============================================================

print(
    "\nFirst 10 sequence predictions:"
)

limit = min(
    10,
    len(predicted_labels)
)

for i in range(limit):

    predicted_class = (
        predicted_labels[i]
    )

    confidence = (
        probabilities[
            i,
            predicted_indices[i]
        ]
        *
        100
    )

    print(
        f"Sequence {i + 1}: "
        f"{predicted_class} "
        f"({confidence:.2f}%)"
    )


# ============================================================
# FINISH
# ============================================================

pose.close()

print(
    "\n" + "=" * 60
)

print(
    "3 CLASS VIDEO PREDICTION COMPLETED"
)

print(
    "=" * 60
)