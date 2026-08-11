import pandas as pd
import numpy as np
import os

# ============================================================
# AI GYM TRAINER - FEATURE ENGINEERING
# ============================================================

INPUT_FILE = "dataset/gym_landmarks.csv"
OUTPUT_FILE = "dataset/gym_features.csv"

print("=" * 60)
print("AI GYM TRAINER - FEATURE ENGINEERING")
print("=" * 60)

# ------------------------------------------------------------
# Load landmark dataset
# ------------------------------------------------------------

df = pd.read_csv(INPUT_FILE)

print("\nOriginal dataset shape:")
print(df.shape)

# ------------------------------------------------------------
# Landmark helper
# ------------------------------------------------------------

def get_point(row, landmark_id):

    return np.array([
        row[f"x_{landmark_id}"],
        row[f"y_{landmark_id}"],
        row[f"z_{landmark_id}"]
    ])


# MediaPipe landmark IDs

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


# ------------------------------------------------------------
# Angle function
# ------------------------------------------------------------

def calculate_angle(a, b, c):

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (
        np.linalg.norm(ba) *
        np.linalg.norm(bc) + 1e-8
    )

    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    angle = np.degrees(
        np.arccos(cosine_angle)
    )

    return angle


# ------------------------------------------------------------
# Create features
# ------------------------------------------------------------

feature_rows = []

for _, row in df.iterrows():

    # Get landmarks

    nose = get_point(row, NOSE)

    left_shoulder = get_point(row, LEFT_SHOULDER)
    right_shoulder = get_point(row, RIGHT_SHOULDER)

    left_elbow = get_point(row, LEFT_ELBOW)
    right_elbow = get_point(row, RIGHT_ELBOW)

    left_wrist = get_point(row, LEFT_WRIST)
    right_wrist = get_point(row, RIGHT_WRIST)

    left_hip = get_point(row, LEFT_HIP)
    right_hip = get_point(row, RIGHT_HIP)

    left_knee = get_point(row, LEFT_KNEE)
    right_knee = get_point(row, RIGHT_KNEE)

    left_ankle = get_point(row, LEFT_ANKLE)
    right_ankle = get_point(row, RIGHT_ANKLE)

    # --------------------------------------------------------
    # Body center
    # --------------------------------------------------------

    hip_center = (left_hip + right_hip) / 2

    shoulder_center = (
        left_shoulder + right_shoulder
    ) / 2

    # --------------------------------------------------------
    # Body scale
    # --------------------------------------------------------

    body_scale = np.linalg.norm(
        shoulder_center - hip_center
    )

    if body_scale < 1e-6:
        body_scale = 1.0

    # --------------------------------------------------------
    # Normalize important landmarks
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
    # Important body angles
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
    # Distances
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
    # Final feature vector
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

    features.append(row["exercise"])

    feature_rows.append(features)


# ============================================================
# Column names
# ============================================================

landmark_names = [
    "nose",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle"
]

columns = []

for name in landmark_names:

    columns.extend([
        f"{name}_x",
        f"{name}_y",
        f"{name}_z"
    ])


columns.extend([
    "left_elbow_angle",
    "right_elbow_angle",

    "left_knee_angle",
    "right_knee_angle",

    "left_hip_angle",
    "right_hip_angle",

    "left_shoulder_angle",
    "right_shoulder_angle",

    "shoulder_width",
    "hip_width",
    "knee_width",

    "exercise"
])


# ============================================================
# Save
# ============================================================

features_df = pd.DataFrame(
    feature_rows,
    columns=columns
)

features_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nFeature engineering completed!")

print("\nNew dataset shape:")
print(features_df.shape)

print("\nExercise classes:")
print(
    features_df["exercise"].value_counts()
)

print("\nSaved to:")
print(OUTPUT_FILE)

print("=" * 60)