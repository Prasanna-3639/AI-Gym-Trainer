import cv2
import mediapipe as mp
import numpy as np


# ============================================================
# AI GYM TRAINER - SQUAT REP COUNTER
# ============================================================

VIDEO_PATH = r"dataset\raw\my_recordings\squat\squat_2.mp4"

# Knee-angle thresholds
DOWN_ANGLE = 100
UP_ANGLE = 160

# Count only after a complete DOWN -> UP movement
MIN_CONFIDENCE = 0.5


# ============================================================
# MEDIAPIPE
# ============================================================

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=MIN_CONFIDENCE,
    min_tracking_confidence=MIN_CONFIDENCE
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
        np.linalg.norm(ba) *
        np.linalg.norm(bc)
    )

    if denominator == 0:
        return 0

    cosine_angle = (
        np.dot(ba, bc) /
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
# OPEN VIDEO
# ============================================================

print("=" * 60)
print("AI GYM TRAINER - SQUAT REP COUNTER")
print("=" * 60)

print("\nOpening video:")
print(VIDEO_PATH)

cap = cv2.VideoCapture(
    VIDEO_PATH
)

if not cap.isOpened():

    print("\nERROR: Could not open video.")

    pose.close()

    raise SystemExit


# ============================================================
# VARIABLES
# ============================================================

rep_count = 0

stage = "UP"

left_knee_angles = []
right_knee_angles = []

frame_count = 0
detected_frames = 0


# ============================================================
# PROCESS VIDEO
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

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


        # ----------------------------------------------------
        # LEFT LEG
        # ----------------------------------------------------

        left_hip = [
            landmarks[
                mp_pose.PoseLandmark.LEFT_HIP
            ].x,

            landmarks[
                mp_pose.PoseLandmark.LEFT_HIP
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

        left_ankle = [
            landmarks[
                mp_pose.PoseLandmark.LEFT_ANKLE
            ].x,

            landmarks[
                mp_pose.PoseLandmark.LEFT_ANKLE
            ].y
        ]


        # ----------------------------------------------------
        # RIGHT LEG
        # ----------------------------------------------------

        right_hip = [
            landmarks[
                mp_pose.PoseLandmark.RIGHT_HIP
            ].x,

            landmarks[
                mp_pose.PoseLandmark.RIGHT_HIP
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

        right_ankle = [
            landmarks[
                mp_pose.PoseLandmark.RIGHT_ANKLE
            ].x,

            landmarks[
                mp_pose.PoseLandmark.RIGHT_ANKLE
            ].y
        ]


        # ----------------------------------------------------
        # CALCULATE KNEE ANGLES
        # ----------------------------------------------------

        left_angle = calculate_angle(
            left_hip,
            left_knee,
            left_ankle
        )

        right_angle = calculate_angle(
            right_hip,
            right_knee,
            right_ankle
        )


        left_knee_angles.append(
            left_angle
        )

        right_knee_angles.append(
            right_angle
        )


        # ----------------------------------------------------
        # USE AVERAGE KNEE ANGLE
        # ----------------------------------------------------

        knee_angle = (
            left_angle +
            right_angle
        ) / 2


        # ----------------------------------------------------
        # SQUAT STATE MACHINE
        # ----------------------------------------------------

        if knee_angle < DOWN_ANGLE:

            stage = "DOWN"


        if (
            knee_angle > UP_ANGLE
            and
            stage == "DOWN"
        ):

            rep_count += 1

            stage = "UP"

            print(
                f"Squat Rep {rep_count} completed"
            )


        # ----------------------------------------------------
        # DRAW LANDMARKS
        # ----------------------------------------------------

        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )


        # ----------------------------------------------------
        # DISPLAY
        # ----------------------------------------------------

        cv2.putText(
            frame,
            f"Squats: {rep_count}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3
        )

        cv2.putText(
            frame,
            f"Knee Angle: {knee_angle:.1f}",
            (30, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Stage: {stage}",
            (30, 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )


    # --------------------------------------------------------
    # SHOW VIDEO
    # --------------------------------------------------------

    cv2.imshow(
        "AI Gym Trainer - Squat Counter",
        frame
    )


    # Press Q to stop
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
# ============================================================

print("\n" + "=" * 60)

print("SQUAT REP COUNTING COMPLETED")

print("=" * 60)

print(
    f"Total frames processed : {frame_count}"
)

print(
    f"Frames with pose       : {detected_frames}"
)

print(
    f"Total squat reps       : {rep_count}"
)

print("=" * 60)