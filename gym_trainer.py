import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import joblib
from collections import Counter


# ============================================================
# AI GYM TRAINER
# FINAL INTEGRATED VERSION
# ============================================================

VIDEO_PATH = r"dataset\raw\my_recordings\shoulder press\shoulder press_1.mp4"

MODEL_PATH = r"models\lstm_3class_exercise_model.keras"
SCALER_PATH = r"models\lstm_3class_scaler.pkl"
ENCODER_PATH = r"models\lstm_3class_label_encoder.pkl"


# ============================================================
# LSTM SETTINGS
# ============================================================

SEQUENCE_LENGTH = 10

CONFIDENCE_THRESHOLD = 0.60

WARMUP_SEQUENCES = 80

MIN_DOMINANCE = 0.55
MIN_MARGIN = 0.15


# ============================================================
# REP COUNTING THRESHOLDS
# ============================================================

# Squat
SQUAT_DOWN_ANGLE = 100
SQUAT_UP_ANGLE = 160

# Push-up
PUSHUP_DOWN_ANGLE = 90
PUSHUP_UP_ANGLE = 160

# Shoulder press
PRESS_DOWN_ANGLE = 70
PRESS_UP_ANGLE = 150


# ============================================================
# MEDIAPIPE
# ============================================================

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


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
# SAME AS create_sequences.py
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
# ANGLE FUNCTION
# ============================================================

def calculate_angle(a, b, c):

    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    denominator = (
        np.linalg.norm(ba)
        *
        np.linalg.norm(bc)
    )

    if denominator == 0:

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

    angle = np.degrees(
        np.arccos(cosine_angle)
    )

    return angle


# ============================================================
# EXACT 50-FEATURE EXTRACTION
# SAME AS create_sequences.py
# ============================================================

def extract_features(landmarks):

    # --------------------------------------------------------
    # LANDMARKS
    # --------------------------------------------------------

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
    # 13 × 3 = 39
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
    # 8 ANGLES
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
    # 3 DISTANCES
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


    return np.array(
        features,
        dtype=np.float32
    )


# ============================================================
# HEADER
# ============================================================

print("=" * 60)
print("AI GYM TRAINER")
print("FINAL INTEGRATED VERSION")
print("=" * 60)


# ============================================================
# LOAD LSTM MODEL
# ============================================================

print("\nLoading LSTM model...")

try:

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print(
        "Model loaded successfully."
    )

except Exception as e:

    print(
        "\nERROR loading model:"
    )

    print(e)

    raise SystemExit


# ============================================================
# LOAD SCALER
# ============================================================

print("\nLoading scaler...")

try:

    scaler = joblib.load(
        SCALER_PATH
    )

    print(
        "Scaler loaded successfully."
    )

except Exception as e:

    print(
        "\nERROR loading scaler:"
    )

    print(e)

    raise SystemExit


# ============================================================
# LOAD LABEL ENCODER
# ============================================================

print("\nLoading label encoder...")

try:

    label_encoder = joblib.load(
        ENCODER_PATH
    )

    print(
        "Label encoder loaded successfully."
    )

except Exception as e:

    print(
        "\nERROR loading label encoder:"
    )

    print(e)

    raise SystemExit


# ============================================================
# SHOW CLASSES
# ============================================================

print("\nClasses:")

for i, name in enumerate(
    label_encoder.classes_
):

    print(
        f"{i} -> {name}"
    )


# ============================================================
# MEDIAPIPE POSE
# ============================================================

pose = mp_pose.Pose(

    static_image_mode=False,

    model_complexity=1,

    min_detection_confidence=0.5,

    min_tracking_confidence=0.5
)


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
# LSTM VARIABLES
# ============================================================

sequence_buffer = []

warmup_predictions = []

locked_exercise = None

current_exercise = "Detecting..."

current_confidence = 0.0

sequence_count = 0


# ============================================================
# REP COUNTERS
#
# These count from FRAME 1.
# They do not wait for LSTM locking.
# ============================================================

squat_reps = 0

pushup_reps = 0

shoulder_press_reps = 0


# ============================================================
# REP STATES
# ============================================================

squat_stage = "UP"

pushup_stage = "UP"

press_stage = "DOWN"


# ============================================================
# FRAME COUNTERS
# ============================================================

frame_count = 0

detected_frames = 0


# ============================================================
# VIDEO LOOP
# ============================================================

while True:

    ret, frame = cap.read()


    if not ret:

        break


    frame_count += 1


    # ========================================================
    # MEDIAPIPE
    # ========================================================

    image = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    results = pose.process(
        image
    )


    if results.pose_landmarks:

        detected_frames += 1

        landmarks = (
            results.pose_landmarks.landmark
        )


        # ====================================================
        # DRAW POSE
        # ====================================================

        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )


        # ====================================================
        # EXTRACT 50 FEATURES
        # ====================================================

        features = extract_features(
            landmarks
        )


        # ====================================================
        # LSTM SEQUENCE
        # ====================================================

        sequence_buffer.append(
            features
        )


        if len(sequence_buffer) > SEQUENCE_LENGTH:

            sequence_buffer.pop(0)


        # ====================================================
        # LSTM PREDICTION
        # ====================================================

        if len(sequence_buffer) == SEQUENCE_LENGTH:

            sequence = np.array(
                sequence_buffer,
                dtype=np.float32
            )


            sequence_2d = sequence.reshape(
                -1,
                50
            )


            sequence_scaled = scaler.transform(
                sequence_2d
            )


            sequence_scaled = (
                sequence_scaled.reshape(
                    1,
                    SEQUENCE_LENGTH,
                    50
                )
            )


            probabilities = model.predict(
                sequence_scaled,
                verbose=0
            )[0]


            predicted_index = int(
                np.argmax(probabilities)
            )


            confidence = float(
                probabilities[
                    predicted_index
                ]
            )


            predicted_label = (
                label_encoder.inverse_transform(
                    [predicted_index]
                )[0]
            )


            sequence_count += 1

            current_confidence = confidence


            # =================================================
            # EXERCISE WARM-UP
            # =================================================

            if locked_exercise is None:

                if confidence >= CONFIDENCE_THRESHOLD:

                    warmup_predictions.append(
                        predicted_label
                    )


                if (
                    len(warmup_predictions)
                    >= WARMUP_SEQUENCES
                ):

                    counts = Counter(
                        warmup_predictions
                    )


                    sorted_classes = (
                        counts.most_common()
                    )


                    top_class = (
                        sorted_classes[0][0]
                    )

                    top_count = (
                        sorted_classes[0][1]
                    )


                    top_percentage = (
                        top_count /
                        len(
                            warmup_predictions
                        )
                    )


                    if len(sorted_classes) > 1:

                        second_count = (
                            sorted_classes[1][1]
                        )

                    else:

                        second_count = 0


                    second_percentage = (
                        second_count /
                        len(
                            warmup_predictions
                        )
                    )


                    margin = (
                        top_percentage -
                        second_percentage
                    )


                    print("\n")
                    print("=" * 60)
                    print(
                        "EXERCISE WARM-UP ANALYSIS"
                    )
                    print("=" * 60)


                    print(
                        f"Sequences analyzed : "
                        f"{len(warmup_predictions)}"
                    )


                    print(
                        "\nClass distribution:"
                    )


                    for class_name, count in (
                        counts.most_common()
                    ):

                        percentage = (
                            count /
                            len(
                                warmup_predictions
                            )
                            *
                            100
                        )


                        print(
                            f"{class_name:<22}"
                            f": {count:4d} "
                            f"({percentage:.1f}%)"
                        )


                    print(
                        "\nDominant class:"
                    )


                    print(
                        f"{top_class} "
                        f"({top_percentage * 100:.1f}%)"
                    )


                    print(
                        f"Margin over second: "
                        f"{margin * 100:.1f}%"
                    )


                    # =========================================
                    # LOCK EXERCISE
                    # =========================================

                    if (

                        top_percentage >=
                        MIN_DOMINANCE

                        and

                        margin >=
                        MIN_MARGIN

                    ):

                        locked_exercise = (
                            top_class
                        )


                        current_exercise = (
                            locked_exercise
                        )


                        print("\n")
                        print("=" * 60)
                        print(
                            "EXERCISE LOCKED"
                        )
                        print(
                            f"Exercise: "
                            f"{locked_exercise}"
                        )
                        print("=" * 60)
                        print("\n")


        # ====================================================
        # 2D LANDMARKS FOR REP COUNTING
        # ====================================================

        left_shoulder = [

            landmarks[
                mp_pose.PoseLandmark.LEFT_SHOULDER
            ].x,

            landmarks[
                mp_pose.PoseLandmark.LEFT_SHOULDER
            ].y

        ]


        right_shoulder = [

            landmarks[
                mp_pose.PoseLandmark.RIGHT_SHOULDER
            ].x,

            landmarks[
                mp_pose.PoseLandmark.RIGHT_SHOULDER
            ].y

        ]


        left_elbow = [

            landmarks[
                mp_pose.PoseLandmark.LEFT_ELBOW
            ].x,

            landmarks[
                mp_pose.PoseLandmark.LEFT_ELBOW
            ].y

        ]


        right_elbow = [

            landmarks[
                mp_pose.PoseLandmark.RIGHT_ELBOW
            ].x,

            landmarks[
                mp_pose.PoseLandmark.RIGHT_ELBOW
            ].y

        ]


        left_wrist = [

            landmarks[
                mp_pose.PoseLandmark.LEFT_WRIST
            ].x,

            landmarks[
                mp_pose.PoseLandmark.LEFT_WRIST
            ].y

        ]


        right_wrist = [

            landmarks[
                mp_pose.PoseLandmark.RIGHT_WRIST
            ].x,

            landmarks[
                mp_pose.PoseLandmark.RIGHT_WRIST
            ].y

        ]


        left_hip = [

            landmarks[
                mp_pose.PoseLandmark.LEFT_HIP
            ].x,

            landmarks[
                mp_pose.PoseLandmark.LEFT_HIP
            ].y

        ]


        right_hip = [

            landmarks[
                mp_pose.PoseLandmark.RIGHT_HIP
            ].x,

            landmarks[
                mp_pose.PoseLandmark.RIGHT_HIP
            ].y

        ]


        left_knee = [

            landmarks[
                mp_pose.PoseLandmark.LEFT_KNEE
            ].x,

            landmarks[
                mp_pose.PoseLandmark.LEFT_KNEE
            ].y

        ]


        right_knee = [

            landmarks[
                mp_pose.PoseLandmark.RIGHT_KNEE
            ].x,

            landmarks[
                mp_pose.PoseLandmark.RIGHT_KNEE
            ].y

        ]


        left_ankle = [

            landmarks[
                mp_pose.PoseLandmark.LEFT_ANKLE
            ].x,

            landmarks[
                mp_pose.PoseLandmark.LEFT_ANKLE
            ].y

        ]


        right_ankle = [

            landmarks[
                mp_pose.PoseLandmark.RIGHT_ANKLE
            ].x,

            landmarks[
                mp_pose.PoseLandmark.RIGHT_ANKLE
            ].y

        ]


        # ====================================================
        # KNEE ANGLES
        # ====================================================

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


        knee_angle = (
            left_knee_angle +
            right_knee_angle
        ) / 2.0


        # ====================================================
        # ELBOW ANGLES
        # ====================================================

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


        elbow_angle = (
            left_elbow_angle +
            right_elbow_angle
        ) / 2.0


        # ====================================================
        # SQUAT REP COUNTER
        # COUNTS FROM FRAME 1
        # ====================================================

        if knee_angle < SQUAT_DOWN_ANGLE:

            squat_stage = "DOWN"


        if (
            knee_angle > SQUAT_UP_ANGLE
            and
            squat_stage == "DOWN"
        ):

            squat_reps += 1

            squat_stage = "UP"

            print(
                f"Squat Rep "
                f"{squat_reps} completed"
            )


        # ====================================================
        # PUSH-UP REP COUNTER
        # COUNTS FROM FRAME 1
        # ====================================================

        if elbow_angle < PUSHUP_DOWN_ANGLE:

            pushup_stage = "DOWN"


        if (
            elbow_angle > PUSHUP_UP_ANGLE
            and
            pushup_stage == "DOWN"
        ):

            pushup_reps += 1

            pushup_stage = "UP"

            print(
                f"Push-up Rep "
                f"{pushup_reps} completed"
            )


        # ====================================================
        # SHOULDER PRESS REP COUNTER
        # COUNTS FROM FRAME 1
        # ====================================================

        if elbow_angle < PRESS_DOWN_ANGLE:

            press_stage = "DOWN"


        if (
            elbow_angle > PRESS_UP_ANGLE
            and
            press_stage == "DOWN"
        ):

            shoulder_press_reps += 1

            press_stage = "UP"

            print(
                f"Shoulder Press Rep "
                f"{shoulder_press_reps} completed"
            )


    # ========================================================
    # DISPLAY
    #
    # Before exercise lock:
    # show all counters temporarily.
    #
    # After exercise lock:
    # show ONLY the locked exercise count.
    # ========================================================

    if locked_exercise == "squat":

        display_squat = squat_reps
        display_pushup = 0
        display_press = 0


    elif locked_exercise == "push-up":

        display_squat = 0
        display_pushup = pushup_reps
        display_press = 0


    elif locked_exercise == "shoulder press":

        display_squat = 0
        display_pushup = 0
        display_press = shoulder_press_reps


    else:

        display_squat = squat_reps
        display_pushup = pushup_reps
        display_press = shoulder_press_reps


    # ========================================================
    # DISPLAY EXERCISE
    # ========================================================

    cv2.putText(
        frame,
        f"Exercise: {current_exercise}",
        (25, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )


    # ========================================================
    # DISPLAY CONFIDENCE
    # ========================================================

    cv2.putText(
        frame,
        f"Confidence: "
        f"{current_confidence * 100:.1f}%",
        (25, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    # ========================================================
    # DISPLAY SQUAT
    # ========================================================

    cv2.putText(
        frame,
        f"Squat: {display_squat}",
        (25, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    # ========================================================
    # DISPLAY PUSH-UP
    # ========================================================

    cv2.putText(
        frame,
        f"Push-up: {display_pushup}",
        (200, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    # ========================================================
    # DISPLAY SHOULDER PRESS
    # ========================================================

    cv2.putText(
        frame,
        f"Shoulder Press: {display_press}",
        (380, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    # ========================================================
    # DISPLAY LSTM SEQUENCES
    # ========================================================

    cv2.putText(
        frame,
        f"LSTM sequences: {sequence_count}",
        (25, 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )


    # ========================================================
    # SHOW VIDEO
    # ========================================================

    cv2.imshow(
        "AI Gym Trainer",
        frame
    )


    # ========================================================
    # PRESS Q TO STOP
    # ========================================================

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

pose.close()


# ============================================================
# FINAL RESULTS
# ONLY LOCKED EXERCISE IS REPORTED
# ============================================================

if locked_exercise == "squat":

    final_reps = squat_reps

    final_squat_reps = squat_reps
    final_pushup_reps = 0
    final_press_reps = 0


elif locked_exercise == "push-up":

    final_reps = pushup_reps

    final_squat_reps = 0
    final_pushup_reps = pushup_reps
    final_press_reps = 0


elif locked_exercise == "shoulder press":

    final_reps = shoulder_press_reps

    final_squat_reps = 0
    final_pushup_reps = 0
    final_press_reps = shoulder_press_reps


else:

    final_reps = 0

    final_squat_reps = 0
    final_pushup_reps = 0
    final_press_reps = 0


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n")

print("=" * 60)

print("AI GYM TRAINER - FINAL RESULTS")

print("=" * 60)

print(
    f"Total frames processed : "
    f"{frame_count}"
)

print(
    f"Frames with pose       : "
    f"{detected_frames}"
)

print(
    f"Total LSTM sequences   : "
    f"{sequence_count}"
)

print(
    f"Squat reps             : "
    f"{final_squat_reps}"
)

print(
    f"Push-up reps           : "
    f"{final_pushup_reps}"
)

print(
    f"Shoulder press reps    : "
    f"{final_press_reps}"
)

print(
    f"Last detected exercise : "
    f"{current_exercise}"
)

print(
    f"Locked exercise        : "
    f"{locked_exercise}"
)

print(
    f"FINAL REP COUNT        : "
    f"{final_reps}"
)

print("=" * 60)