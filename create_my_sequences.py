import cv2
import mediapipe as mp
import numpy as np
import os

# ============================================================
# AI GYM TRAINER - MY RECORDINGS SEQUENCE CREATION
# ============================================================

BASE_PATH = "dataset/frames/my_recordings/squat"

TRAIN_OUTPUT_X = "dataset/my_train_sequences.npy"
TRAIN_OUTPUT_Y = "dataset/my_train_labels.npy"

TEST_OUTPUT_X = "dataset/my_test_sequences.npy"
TEST_OUTPUT_Y = "dataset/my_test_labels.npy"

SEQUENCE_LENGTH = 10

print("=" * 60)
print("AI GYM TRAINER - MY RECORDINGS SEQUENCES")
print("=" * 60)

# ============================================================
# MEDIAPIPE
# ============================================================

mp_pose = mp.solutions.pose

pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=1,
    min_detection_confidence=0.5
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
# ANGLE
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

    nose = get_point(landmarks, NOSE)

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
        left_hip + right_hip
    ) / 2

    shoulder_center = (
        left_shoulder + right_shoulder
    ) / 2

    # --------------------------------------------------------
    # BODY SCALE
    # --------------------------------------------------------

    body_scale = np.linalg.norm(
        shoulder_center - hip_center
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
            point - hip_center
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
# PROCESS ONE VIDEO FOLDER
# ============================================================

def process_video(video_name):

    video_path = os.path.join(
        BASE_PATH,
        video_name
    )

    image_files = [
        f for f in os.listdir(video_path)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]

    # Sort frame_00000, frame_00001, ...
    image_files.sort(
        key=lambda x: int(
            os.path.splitext(x)[0].split("_")[-1]
        )
    )

    frame_features = []

    for image_file in image_files:

        image_path = os.path.join(
            video_path,
            image_file
        )

        image = cv2.imread(
            image_path
        )

        if image is None:
            continue

        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        results = pose.process(rgb)

        if results.pose_landmarks:

            features = extract_features(
                results.pose_landmarks.landmark
            )

            frame_features.append(
                features
            )

    sequences = []

    if len(frame_features) >= SEQUENCE_LENGTH:

        for start in range(
            0,
            len(frame_features)
            - SEQUENCE_LENGTH + 1,
            SEQUENCE_LENGTH
        ):

            sequence = frame_features[
                start:start + SEQUENCE_LENGTH
            ]

            sequences.append(sequence)

    return sequences


# ============================================================
# TRAINING VIDEOS
# ============================================================

train_videos = [
    "squat_1",
    "squat_2"
]

test_videos = [
    "squat_3"
]

# ============================================================
# CREATE TRAINING SEQUENCES
# ============================================================

train_sequences = []
train_labels = []

print("\nTraining videos:")

for video in train_videos:

    sequences = process_video(video)

    print(
        f"{video}: "
        f"{len(sequences)} sequences"
    )

    for sequence in sequences:

        train_sequences.append(
            sequence
        )

        train_labels.append(
            "squat"
        )


# ============================================================
# CREATE TEST SEQUENCES
# ============================================================

test_sequences = []
test_labels = []

print("\nTest video:")

for video in test_videos:

    sequences = process_video(video)

    print(
        f"{video}: "
        f"{len(sequences)} sequences"
    )

    for sequence in sequences:

        test_sequences.append(
            sequence
        )

        test_labels.append(
            "squat"
        )


# ============================================================
# CONVERT TO NUMPY
# ============================================================

X_train = np.array(
    train_sequences,
    dtype=np.float32
)

y_train = np.array(
    train_labels
)

X_test = np.array(
    test_sequences,
    dtype=np.float32
)

y_test = np.array(
    test_labels
)

# ============================================================
# SAVE
# ============================================================

np.save(
    TRAIN_OUTPUT_X,
    X_train
)

np.save(
    TRAIN_OUTPUT_Y,
    y_train
)

np.save(
    TEST_OUTPUT_X,
    X_test
)

np.save(
    TEST_OUTPUT_Y,
    y_test
)

# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 60)
print("MY RECORDINGS SEQUENCE CREATION COMPLETED")
print("=" * 60)

print(
    "\nTraining X shape:",
    X_train.shape
)

print(
    "Training y shape:",
    y_train.shape
)

print(
    "\nTest X shape:",
    X_test.shape
)

print(
    "Test y shape:",
    y_test.shape
)

print("\nSaved files:")

print(
    TRAIN_OUTPUT_X
)

print(
    TRAIN_OUTPUT_Y
)

print(
    TEST_OUTPUT_X
)

print(
    TEST_OUTPUT_Y
)

print("=" * 60)

pose.close()