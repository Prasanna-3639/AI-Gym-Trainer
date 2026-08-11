import cv2
import mediapipe as mp
import numpy as np
import os
import pandas as pd

# ============================================================
# AI GYM TRAINER - 3 CLASS TEMPORAL SEQUENCE CREATION
# ============================================================

BASE_PATH = "dataset/frames"

OUTPUT_X = "dataset/sequence_X.npy"
OUTPUT_Y = "dataset/sequence_y.npy"

SEQUENCE_LENGTH = 10
STEP = 3

# ============================================================
# ONLY 3 EXERCISES
# ============================================================

EXERCISES = [
    "push-up",
    "shoulder press",
    "squat"
]

print("=" * 60)
print("AI GYM TRAINER - 3 CLASS SEQUENCE CREATION")
print("=" * 60)

print("\nExercises used for training:")

for exercise in EXERCISES:
    print("-", exercise)

print("\nExcluded:")
print("- barbell biceps curl")


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
# GET LANDMARK POINT
# ============================================================

def get_point(landmarks, landmark_id):

    p = landmarks[landmark_id]

    return np.array([
        p.x,
        p.y,
        p.z
    ], dtype=np.float32)


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
            f"but got {len(features)}"
        )

    return features


# ============================================================
# PROCESS ONE VIDEO FOLDER
# ============================================================

def process_video(video_path):

    image_files = []

    for file_name in os.listdir(video_path):

        if file_name.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):

            image_files.append(
                file_name
            )


    # --------------------------------------------------------
    # SORT FRAMES
    # --------------------------------------------------------

    def frame_number(file_name):

        try:

            return int(
                os.path.splitext(
                    file_name
                )[0].split("_")[-1]
            )

        except:

            return 0


    image_files.sort(
        key=frame_number
    )


    frame_features = []


    # ========================================================
    # EXTRACT FEATURES
    # ========================================================

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


        results = pose.process(
            rgb
        )


        if results.pose_landmarks:

            features = extract_features(
                results.pose_landmarks.landmark
            )

            frame_features.append(
                features
            )


    # ========================================================
    # CREATE OVERLAPPING SEQUENCES
    # ========================================================

    sequences = []


    if len(frame_features) >= SEQUENCE_LENGTH:

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


    return frame_features, sequences


# ============================================================
# MAIN PROCESSING
# ============================================================

all_sequences = []
all_labels = []

total_videos = 0
total_pose_frames = 0


for exercise in EXERCISES:

    exercise_path = os.path.join(
        BASE_PATH,
        exercise
    )


    if not os.path.exists(
        exercise_path
    ):

        print(
            f"\nWARNING: Folder not found: "
            f"{exercise_path}"
        )

        continue


    video_folders = []


    for folder in os.listdir(
        exercise_path
    ):

        folder_path = os.path.join(
            exercise_path,
            folder
        )


        if os.path.isdir(
            folder_path
        ):

            video_folders.append(
                folder
            )


    video_folders.sort()


    print(
        f"\nExercise: {exercise} | "
        f"Videos: {len(video_folders)}"
    )


    for video_folder in video_folders:

        video_path = os.path.join(
            exercise_path,
            video_folder
        )


        try:

            frame_features, sequences = (
                process_video(video_path)
            )


        except Exception as e:

            print(
                f"{video_folder}: ERROR - {e}"
            )

            continue


        total_videos += 1

        total_pose_frames += len(
            frame_features
        )


        for sequence in sequences:

            all_sequences.append(
                sequence
            )

            all_labels.append(
                exercise
            )


        print(
            f"{video_folder}: "
            f"{len(frame_features)} frames "
            f"→ {len(sequences)} sequences"
        )


# ============================================================
# CONVERT TO NUMPY
# ============================================================

X = np.array(
    all_sequences,
    dtype=np.float32
)

y = np.array(
    all_labels
)


# ============================================================
# SAVE
# ============================================================

np.save(
    OUTPUT_X,
    X
)

np.save(
    OUTPUT_Y,
    y
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 60)
print("3 CLASS SEQUENCE CREATION COMPLETED")
print("=" * 60)

print(
    "\nTotal videos processed:",
    total_videos
)

print(
    "Total pose frames:",
    total_pose_frames
)

print(
    "Total sequences:",
    len(X)
)

print(
    "\nX shape:",
    X.shape
)

print(
    "y shape:",
    y.shape
)

print(
    "\nFrames per sequence:",
    SEQUENCE_LENGTH
)

print(
    "Step size:",
    STEP
)

if len(X) > 0:

    print(
        "Features per frame:",
        X.shape[2]
    )

    print(
        "\nExercise distribution:"
    )

    distribution = pd.Series(
        y
    ).value_counts()

    print(
        distribution
    )

else:

    print(
        "\nWARNING: No sequences were created!"
    )


print("\nSaved files:")

print(
    OUTPUT_X
)

print(
    OUTPUT_Y
)

print("=" * 60)


# ============================================================
# CLOSE MEDIAPIPE
# ============================================================

pose.close()