import os
import threading
from collections import Counter

import av
import cv2
import joblib
import mediapipe as mp
import numpy as np
import streamlit as st
import tensorflow as tf

from streamlit_webrtc import webrtc_streamer


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Gym Trainer",
    page_icon="🏋️",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🏋️ AI Gym Trainer")

st.subheader(
    "AI-based Exercise Detection & Rep Counting"
)

st.write(
    "Detect Squats, Push-ups and Shoulder Press using "
    "MediaPipe Pose Estimation + LSTM."
)


# ============================================================
# PATH FINDER
# ============================================================

def find_file(possible_names):

    for name in possible_names:

        if os.path.exists(name):
            return name

    return None


# ============================================================
# MODEL PATHS
# ============================================================

MODEL_PATH = find_file([
    r"models\lstm_3class_exercise_model.keras",
    r"models\lstm_3class_model.keras",
    r"models\lstm_model.keras",
    r"lstm_3class_exercise_model.keras",
    r"lstm_3class_model.keras",
    r"lstm_model.keras"
])


SCALER_PATH = find_file([
    r"models\lstm_3class_scaler.pkl",
    r"models\scaler.pkl",
    r"lstm_3class_scaler.pkl",
    r"scaler.pkl"
])


ENCODER_PATH = find_file([
    r"models\lstm_3class_label_encoder.pkl",
    r"models\label_encoder.pkl",
    r"lstm_3class_label_encoder.pkl",
    r"label_encoder.pkl"
])


# ============================================================
# SETTINGS
# ============================================================

SEQUENCE_LENGTH = 10

FEATURE_COUNT = 50

CONFIDENCE_THRESHOLD = 0.60

WARMUP_SEQUENCES = 80

MIN_DOMINANCE = 0.55

MIN_MARGIN = 0.15


# ============================================================
# REP THRESHOLDS
# ============================================================

SQUAT_DOWN_ANGLE = 100
SQUAT_UP_ANGLE = 160

PUSHUP_DOWN_ANGLE = 90
PUSHUP_UP_ANGLE = 160

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
# ANGLE
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

    return np.degrees(
        np.arccos(cosine_angle)
    )


# ============================================================
# EXACT 50-FEATURE EXTRACTION
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
    # 13 points x 3 = 39
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
    # ANGLES = 8
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
    # DISTANCES = 3
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
    # FINAL 50
    # 39 + 8 + 3 = 50
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


    if len(features) != FEATURE_COUNT:

        raise ValueError(
            f"Expected 50 features, "
            f"got {len(features)}"
        )


    return np.array(
        features,
        dtype=np.float32
    )


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_models():

    if MODEL_PATH is None:

        raise FileNotFoundError(
            "LSTM model file not found."
        )


    if SCALER_PATH is None:

        raise FileNotFoundError(
            "Scaler file not found."
        )


    if ENCODER_PATH is None:

        raise FileNotFoundError(
            "Label encoder file not found."
        )


    model = tf.keras.models.load_model(
        MODEL_PATH
    )


    scaler = joblib.load(
        SCALER_PATH
    )


    label_encoder = joblib.load(
        ENCODER_PATH
    )


    return (
        model,
        scaler,
        label_encoder
    )


# ============================================================
# GET EXERCISE ANGLES
# ============================================================

def get_angles(landmarks):

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


    knee_angle = (
        left_knee_angle +
        right_knee_angle
    ) / 2.0


    elbow_angle = (
        left_elbow_angle +
        right_elbow_angle
    ) / 2.0


    return knee_angle, elbow_angle


# ============================================================
# REP COUNTER CLASS
# ============================================================

class RepCounter:

    def __init__(
        self,
        model,
        scaler,
        label_encoder
    ):

        self.model = model
        self.scaler = scaler
        self.label_encoder = label_encoder

        self.pose = mp_pose.Pose(

            static_image_mode=False,

            model_complexity=1,

            min_detection_confidence=0.5,

            min_tracking_confidence=0.5
        )


        self.sequence_buffer = []

        self.warmup_predictions = []

        self.locked_exercise = None

        self.current_exercise = "Detecting..."

        self.confidence = 0.0


        self.squat_reps = 0

        self.pushup_reps = 0

        self.press_reps = 0


        self.squat_stage = "UP"

        self.pushup_stage = "UP"

        self.press_stage = "DOWN"


        self.frame_count = 0

        self.pose_frames = 0

        self.sequence_count = 0


        self.lock = threading.Lock()


    # ========================================================
    # RESET
    # ========================================================

    def reset(self):

        with self.lock:

            self.sequence_buffer.clear()

            self.warmup_predictions.clear()

            self.locked_exercise = None

            self.current_exercise = "Detecting..."

            self.confidence = 0.0

            self.squat_reps = 0

            self.pushup_reps = 0

            self.press_reps = 0

            self.squat_stage = "UP"

            self.pushup_stage = "UP"

            self.press_stage = "DOWN"

            self.frame_count = 0

            self.pose_frames = 0

            self.sequence_count = 0


    # ========================================================
    # PROCESS FRAME
    # ========================================================

    def process(self, frame):

        image = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        results = self.pose.process(
            image
        )


        self.frame_count += 1


        if not results.pose_landmarks:

            return frame


        self.pose_frames += 1


        landmarks = (
            results.pose_landmarks.landmark
        )


        # ====================================================
        # FEATURES
        # ====================================================

        try:

            features = extract_features(
                landmarks
            )

        except Exception:

            return frame


        self.sequence_buffer.append(
            features
        )


        if len(
            self.sequence_buffer
        ) > SEQUENCE_LENGTH:

            self.sequence_buffer.pop(0)


        # ====================================================
        # LSTM PREDICTION
        # ====================================================

        if len(
            self.sequence_buffer
        ) == SEQUENCE_LENGTH:


            sequence = np.array(
                self.sequence_buffer,
                dtype=np.float32
            )


            sequence_2d = sequence.reshape(
                -1,
                FEATURE_COUNT
            )


            sequence_scaled = (
                self.scaler.transform(
                    sequence_2d
                )
            )


            sequence_scaled = (
                sequence_scaled.reshape(
                    1,
                    SEQUENCE_LENGTH,
                    FEATURE_COUNT
                )
            )


            probabilities = (
                self.model.predict(
                    sequence_scaled,
                    verbose=0
                )[0]
            )


            predicted_index = int(
                np.argmax(
                    probabilities
                )
            )


            confidence = float(
                probabilities[
                    predicted_index
                ]
            )


            predicted_label = (
                self.label_encoder
                .inverse_transform(
                    [predicted_index]
                )[0]
            )


            self.confidence = confidence

            self.sequence_count += 1


            # =================================================
            # WARM-UP
            # =================================================

            if self.locked_exercise is None:

                if (
                    confidence
                    >=
                    CONFIDENCE_THRESHOLD
                ):

                    self.warmup_predictions.append(
                        predicted_label
                    )


                if len(
                    self.warmup_predictions
                ) >= WARMUP_SEQUENCES:

                    counts = Counter(
                        self.warmup_predictions
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
                        top_count
                        /
                        len(
                            self.warmup_predictions
                        )
                    )


                    if len(
                        sorted_classes
                    ) > 1:

                        second_count = (
                            sorted_classes[1][1]
                        )

                    else:

                        second_count = 0


                    second_percentage = (
                        second_count
                        /
                        len(
                            self.warmup_predictions
                        )
                    )


                    margin = (
                        top_percentage
                        -
                        second_percentage
                    )


                    if (

                        top_percentage
                        >=
                        MIN_DOMINANCE

                        and

                        margin
                        >=
                        MIN_MARGIN

                    ):

                        self.locked_exercise = (
                            top_class
                        )


                        self.current_exercise = (
                            top_class
                        )


            # =================================================
            # USE ONLY LOCKED EXERCISE
            # =================================================

            if self.locked_exercise is not None:

                self.current_exercise = (
                    self.locked_exercise
                )


        # ====================================================
        # ANGLES
        # ====================================================

        knee_angle, elbow_angle = (
            get_angles(
                landmarks
            )
        )


        # ====================================================
        # SQUAT
        # ====================================================

        if self.locked_exercise == "squat":

            if knee_angle < SQUAT_DOWN_ANGLE:

                self.squat_stage = "DOWN"


            if (

                knee_angle > SQUAT_UP_ANGLE

                and

                self.squat_stage == "DOWN"

            ):

                self.squat_reps += 1

                self.squat_stage = "UP"


        # ====================================================
        # PUSH-UP
        # ====================================================

        elif self.locked_exercise == "push-up":

            if elbow_angle < PUSHUP_DOWN_ANGLE:

                self.pushup_stage = "DOWN"


            if (

                elbow_angle > PUSHUP_UP_ANGLE

                and

                self.pushup_stage == "DOWN"

            ):

                self.pushup_reps += 1

                self.pushup_stage = "UP"


        # ====================================================
        # SHOULDER PRESS
        # ====================================================

        elif self.locked_exercise == "shoulder press":

            if elbow_angle < PRESS_DOWN_ANGLE:

                self.press_stage = "DOWN"


            if (

                elbow_angle > PRESS_UP_ANGLE

                and

                self.press_stage == "DOWN"

            ):

                self.press_reps += 1

                self.press_stage = "UP"


        # ====================================================
        # DRAW MEDIAPIPE
        # ====================================================

        mp_drawing.draw_landmarks(

            frame,

            results.pose_landmarks,

            mp_pose.POSE_CONNECTIONS
        )


        # ====================================================
        # DISPLAY
        # ====================================================

        exercise_text = (
            self.locked_exercise
            if self.locked_exercise
            else "Detecting..."
        )


        if exercise_text == "squat":

            reps = self.squat_reps

        elif exercise_text == "push-up":

            reps = self.pushup_reps

        elif exercise_text == "shoulder press":

            reps = self.press_reps

        else:

            reps = 0


        # ----------------------------------------------------
        # Background box
        # ----------------------------------------------------

        cv2.rectangle(

            frame,

            (15, 15),

            (450, 180),

            (0, 0, 0),

            -1
        )


        cv2.putText(

            frame,

            "AI GYM TRAINER",

            (30, 45),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.9,

            (0, 255, 255),

            2
        )


        cv2.putText(

            frame,

            f"Exercise: {exercise_text}",

            (30, 80),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (255, 255, 255),

            2
        )


        cv2.putText(

            frame,

            f"Reps: {reps}",

            (30, 120),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.8,

            (0, 255, 0),

            2
        )


        cv2.putText(

            frame,

            f"Confidence: {self.confidence * 100:.1f}%",

            (30, 155),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.6,

            (255, 255, 255),

            2
        )


        return frame


    # ========================================================
    # RESULTS
    # ========================================================

    def get_results(self):

        with self.lock:

            exercise = (
                self.locked_exercise
            )


            if exercise == "squat":

                reps = self.squat_reps

            elif exercise == "push-up":

                reps = self.pushup_reps

            elif exercise == "shoulder press":

                reps = self.press_reps

            else:

                reps = 0


            return {

                "exercise": exercise,

                "reps": reps,

                "confidence": self.confidence,

                "frames": self.frame_count,

                "pose_frames": self.pose_frames,

                "sequences": self.sequence_count,

                "squat": self.squat_reps,

                "pushup": self.pushup_reps,

                "press": self.press_reps
            }


# ============================================================
# UPLOAD VIDEO PROCESSOR
# ============================================================

def process_uploaded_video(
    video_path,
    model,
    scaler,
    label_encoder,
    progress_bar,
    status
):

    counter = RepCounter(
        model,
        scaler,
        label_encoder
    )


    cap = cv2.VideoCapture(
        video_path
    )


    if not cap.isOpened():

        raise ValueError(
            "Could not open uploaded video."
        )


    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )


    while True:

        ret, frame = cap.read()


        if not ret:

            break


        counter.process(
            frame
        )


        if total_frames > 0:

            progress = (
                counter.frame_count
                /
                total_frames
            )


            progress_bar.progress(
                min(
                    float(progress),
                    1.0
                )
            )


        status.write(
            f"Processing frame "
            f"{counter.frame_count}/"
            f"{total_frames}"
        )


    cap.release()


    return counter.get_results()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Settings")

    st.write("Supported exercises:")

    st.write("✅ Squat")

    st.write("✅ Push-up")

    st.write("✅ Shoulder Press")

    st.divider()

    st.write("Model: LSTM")

    st.write("Features: 50")

    st.write("Sequence Length: 10")


# ============================================================
# MODEL STATUS
# ============================================================

if (
    MODEL_PATH is None
    or
    SCALER_PATH is None
    or
    ENCODER_PATH is None
):

    st.warning(
        "⚠️ Model files were not found automatically. "
        "Check your model/scaler/encoder paths."
    )


# ============================================================
# MODE
# ============================================================

st.subheader(
    "🎯 Choose Workout Mode"
)


mode = st.radio(

    "Select mode:",

    [
        "📹 Upload Video",
        "🎥 Live Webcam"
    ],

    horizontal=True
)


# ============================================================
# UPLOAD MODE
# ============================================================

if mode == "📹 Upload Video":

    st.subheader(
        "📹 Upload Workout Video"
    )


    uploaded_file = st.file_uploader(

        "Choose a workout video",

        type=[
            "mp4",
            "avi",
            "mov",
            "mkv"
        ]
    )


    if uploaded_file is not None:

        temp_path = (
            "temp_uploaded_video.mp4"
        )


        with open(
            temp_path,
            "wb"
        ) as f:

            f.write(
                uploaded_file.getbuffer()
            )


        st.video(
            temp_path
        )


        if st.button(
            "🚀 Start AI Analysis",
            use_container_width=True
        ):

            try:

                with st.spinner(
                    "Loading LSTM model..."
                ):

                    model, scaler, label_encoder = (
                        load_models()
                    )


                st.success(
                    "Model loaded successfully."
                )


                progress_bar = st.progress(
                    0
                )


                status = st.empty()


                result = process_uploaded_video(

                    temp_path,

                    model,

                    scaler,

                    label_encoder,

                    progress_bar,

                    status
                )


                progress_bar.progress(
                    1.0
                )


                status.success(
                    "Analysis completed!"
                )


                # =================================================
                # RESULTS
                # =================================================

                st.divider()

                st.subheader(
                    "🏆 AI Gym Trainer Results"
                )


                c1, c2, c3 = st.columns(3)


                with c1:

                    st.metric(
                        "Exercise",
                        result["exercise"]
                        or
                        "Unknown"
                    )


                with c2:

                    st.metric(
                        "Final Reps",
                        result["reps"]
                    )


                with c3:

                    st.metric(
                        "Confidence",
                        f"{result['confidence'] * 100:.1f}%"
                    )


                st.subheader(
                    "💪 Rep Count"
                )


                r1, r2, r3 = st.columns(3)


                with r1:

                    st.metric(
                        "Squat",
                        result["squat"]
                    )


                with r2:

                    st.metric(
                        "Push-up",
                        result["pushup"]
                    )


                with r3:

                    st.metric(
                        "Shoulder Press",
                        result["press"]
                    )


                st.subheader(
                    "📊 Processing Details"
                )


                d1, d2, d3 = st.columns(3)


                with d1:

                    st.metric(
                        "Total Frames",
                        result["frames"]
                    )


                with d2:

                    st.metric(
                        "Pose Frames",
                        result["pose_frames"]
                    )


                with d3:

                    st.metric(
                        "LSTM Sequences",
                        result["sequences"]
                    )


                st.success(

                    f"🎯 Detected "
                    f"{result['exercise'] or 'unknown'} "
                    f"with "
                    f"{result['reps']} "
                    f"completed reps."
                )


            except Exception as e:

                st.error(
                    "Error during analysis."
                )

                st.exception(e)


        # --------------------------------------------------------
        # CLEANUP
        # --------------------------------------------------------

        # Don't delete immediately because st.video
        # may still need the file during the rerun.


# ============================================================
# LIVE WEBCAM
# ============================================================

else:

    st.subheader(
        "🎥 Live Workout Recording"
    )


    st.info(
        "Click START below and allow camera access. "
        "The AI will detect your exercise and count reps "
        "in real time."
    )


    # ========================================================
    # LOAD MODEL
    # ========================================================

    try:

        model, scaler, label_encoder = (
            load_models()
        )

    except Exception as e:

        st.error(
            "Could not load the AI model."
        )

        st.exception(e)

        st.stop()


    # ========================================================
    # CREATE SESSION COUNTER
    # ========================================================

    if (
        "live_counter" not in st.session_state
    ):

        st.session_state.live_counter = (
            RepCounter(
                model,
                scaler,
                label_encoder
            )
        )


    counter = (
        st.session_state.live_counter
    )


    # ========================================================
    # LIVE CALLBACK
    # ========================================================

    def video_frame_callback(frame):

        img = frame.to_ndarray(
            format="bgr24"
        )


        processed = counter.process(
            img
        )


        return av.VideoFrame.from_ndarray(

            processed,

            format="bgr24"
        )


    # ========================================================
    # WEBCAM
    # ========================================================

    ctx = webrtc_streamer(

        key="ai-gym-live-camera",

        video_frame_callback=(
            video_frame_callback
        ),

        media_stream_constraints={

            "video": True,

            "audio": False
        },

        async_processing=True
    )


    # ========================================================
    # LIVE INFORMATION
    # ========================================================

    st.divider()

    st.subheader(
        "📊 Live Statistics"
    )


    result = counter.get_results()


    c1, c2, c3 = st.columns(3)


    with c1:

        st.metric(
            "Exercise",
            result["exercise"]
            or
            "Detecting..."
        )


    with c2:

        st.metric(
            "Reps",
            result["reps"]
        )


    with c3:

        st.metric(
            "Confidence",
            f"{result['confidence'] * 100:.1f}%"
        )


    st.subheader(
        "💪 Current Rep Count"
    )


    r1, r2, r3 = st.columns(3)


    with r1:

        st.metric(
            "Squat",
            result["squat"]
        )


    with r2:

        st.metric(
            "Push-up",
            result["pushup"]
        )


    with r3:

        st.metric(
            "Shoulder Press",
            result["press"]
        )


    st.divider()


    st.write(
        "📌 The exercise is locked after the "
        "initial warm-up analysis."
    )


    st.write(
        "📌 Only the locked exercise is counted."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI Gym Trainer | "
    "LSTM + MediaPipe Pose Estimation | "
    "Real-time Exercise Recognition"
)
