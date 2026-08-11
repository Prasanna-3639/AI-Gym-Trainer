import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import joblib
from collections import Counter


# ============================================================
# AI GYM TRAINER - LSTM V2 VIDEO PREDICTION
# ============================================================

VIDEO_PATH = r"dataset\raw\my_recordings\squat\squat_3.mp4"

MODEL_PATH = r"models\lstm_v2_exercise_model.keras"
SCALER_PATH = r"models\lstm_v2_scaler.pkl"
ENCODER_PATH = r"models\lstm_v2_label_encoder.pkl"

SEQUENCE_LENGTH = 10
STEP = 3


print("=" * 60)
print("AI GYM TRAINER - LSTM V2 VIDEO PREDICTION")
print("=" * 60)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading LSTM V2 model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("LSTM V2 model loaded successfully!")


# ============================================================
# LOAD SCALER
# ============================================================

scaler = joblib.load(
    SCALER_PATH
)

print("Scaler loaded successfully!")


# ============================================================
# LOAD LABEL ENCODER
# ============================================================

label_encoder = joblib.load(
    ENCODER_PATH
)

print("Label encoder loaded successfully!")

print("\nClasses:")

print(
    label_encoder.classes_
)


# ============================================================
# MEDIAPIPE
# ============================================================

mp_pose = mp.solutions.pose

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
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

print(
    VIDEO_PATH
)


cap = cv2.VideoCapture(
    VIDEO_PATH
)


if not cap.isOpened():

    raise RuntimeError(
        "Could not open video. "
        "Check the VIDEO_PATH."
    )


# ============================================================
# PROCESS VIDEO
# ============================================================

frame_features = []

total_frames = 0
pose_frames = 0


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

        pose_frames += 1


        features = extract_features(
            results.pose_landmarks.landmark
        )


        frame_features.append(
            features
        )


cap.release()

pose.close()


# ============================================================
# CHECK FRAMES
# ============================================================

print(
    "\nTotal frames processed:",
    total_frames
)

print(
    "Frames with pose detected:",
    pose_frames
)


if len(frame_features) < SEQUENCE_LENGTH:

    raise RuntimeError(
        "Not enough pose frames to create "
        "a sequence."
    )


# ============================================================
# CREATE OVERLAPPING SEQUENCES
# ============================================================

sequences = []


for start in range(

    0,

    len(frame_features)
    - SEQUENCE_LENGTH + 1,

    STEP

):

    sequence = frame_features[
        start:
        start + SEQUENCE_LENGTH
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
# SCALE FEATURES
# ============================================================

number_of_sequences = X.shape[0]

number_of_frames = X.shape[1]

number_of_features = X.shape[2]


X_2d = X.reshape(
    -1,
    number_of_features
)


X_scaled = scaler.transform(
    X_2d
)


X_scaled = X_scaled.reshape(
    number_of_sequences,
    number_of_frames,
    number_of_features
)


# ============================================================
# PREDICTION
# ============================================================

print("\nGenerating predictions...")


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
# PREDICTION COUNTS
# ============================================================

prediction_counts = Counter(
    predicted_labels
)


print(
    "\nPrediction counts:"
)


for exercise in label_encoder.classes_:

    count = prediction_counts.get(
        exercise,
        0
    )

    percentage = (
        count /
        len(predicted_labels)
    ) * 100


    print(
        f"{exercise:25s}: "
        f"{count:4d} "
        f"({percentage:.1f}%)"
    )


# ============================================================
# FINAL PREDICTION
# ============================================================

final_prediction = (
    prediction_counts
    .most_common(1)[0][0]
)


print(
    "\nFinal predicted exercise:",
    final_prediction
)


# ============================================================
# AVERAGE CLASS PROBABILITY
# ============================================================

average_probabilities = (
    probabilities.mean(
        axis=0
    )
)


print(
    "\nAverage class probabilities:"
)


for class_name, probability in zip(

    label_encoder.classes_,

    average_probabilities

):

    print(
        f"{class_name:25s}: "
        f"{probability * 100:.2f}%"
    )


# ============================================================
# INDIVIDUAL SEQUENCE PREDICTIONS
# ============================================================

print(
    "\nFirst 10 sequence predictions:"
)


for i in range(
    min(10, len(predicted_labels))
):

    predicted_class = (
        predicted_labels[i]
    )

    confidence = (
        probabilities[i]
        .max()
        * 100
    )


    print(
        f"Sequence {i + 1}: "
        f"{predicted_class} "
        f"({confidence:.2f}%)"
    )


print("\n" + "=" * 60)
print("LSTM V2 VIDEO PREDICTION COMPLETED")
print("=" * 60)