import cv2
import mediapipe as mp
import numpy as np


# ============================================================
# AI GYM TRAINER - PUSH-UP REP COUNTER
# ============================================================

VIDEO_PATH = r"dataset\raw\my_recordings\push-up\push-up_1.mp4"

# Push-up angle thresholds
DOWN_ANGLE = 90
UP_ANGLE = 160

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
print("AI GYM TRAINER - PUSH-UP REP COUNTER")
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
        # LEFT ARM
        # ----------------------------------------------------

        left_shoulder = [
            landmarks[
                mp_pose.PoseLandmark.LEFT_SHOULDER
            ].x,

            landmarks[
                mp_pose.PoseLandmark.LEFT_SHOULDER
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

        left_wrist = [
            landmarks[
                mp_pose.PoseLandmark.LEFT_WRIST
            ].x,

            landmarks[
                mp_pose.PoseLandmark.LEFT_WRIST
            ].y
        ]


        # ----------------------------------------------------
        # RIGHT ARM
        # ----------------------------------------------------

        right_shoulder = [
            landmarks[
                mp_pose.PoseLandmark.RIGHT_SHOULDER
            ].x,

            landmarks[
                mp_pose.PoseLandmark.RIGHT_SHOULDER
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

        right_wrist = [
            landmarks[
                mp_pose.PoseLandmark.RIGHT_WRIST
            ].x,

            landmarks[
                mp_pose.PoseLandmark.RIGHT_WRIST
            ].y
        ]


        # ----------------------------------------------------
        # ELBOW ANGLES
        # ----------------------------------------------------

        left_angle = calculate_angle(
            left_shoulder,
            left_elbow,
            left_wrist
        )

        right_angle = calculate_angle(
            right_shoulder,
            right_elbow,
            right_wrist
        )


        # Average both arms
        elbow_angle = (
            left_angle +
            right_angle
        ) / 2


        # ----------------------------------------------------
        # PUSH-UP STATE MACHINE
        # ----------------------------------------------------

        if elbow_angle < DOWN_ANGLE:

            stage = "DOWN"


        if (
            elbow_angle > UP_ANGLE
            and
            stage == "DOWN"
        ):

            rep_count += 1

            stage = "UP"

            print(
                f"Push-up Rep {rep_count} completed"
            )


        # ----------------------------------------------------
        # DRAW POSE
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
            f"Push-ups: {rep_count}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3
        )

        cv2.putText(
            frame,
            f"Elbow Angle: {elbow_angle:.1f}",
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
        "AI Gym Trainer - Push-up Counter",
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

print("PUSH-UP REP COUNTING COMPLETED")

print("=" * 60)

print(
    f"Total frames processed : {frame_count}"
)

print(
    f"Frames with pose       : {detected_frames}"
)

print(
    f"Total push-up reps     : {rep_count}"
)

print("=" * 60)