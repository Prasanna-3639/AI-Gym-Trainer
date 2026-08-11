import cv2
import mediapipe as mp
import pandas as pd
import joblib
from collections import Counter

# ============================================================
# AI GYM TRAINER - TEST VIDEO PREDICTION
# ============================================================

MODEL_PATH = "models/exercise_model.pkl"

VIDEO_PATH = r"C:\Users\hp\Downloads\archive (8)\my_test_video_1\squat\squat_4.mp4"


# ============================================================
# 1. LOAD MODEL
# ============================================================

print("=" * 60)
print("AI GYM TRAINER - TEST VIDEO")
print("=" * 60)

print("\nLoading trained model...")

model = joblib.load(MODEL_PATH)

print("Model loaded successfully!")

print("\nModel classes:")
print(model.classes_)


# ============================================================
# 2. INITIALIZE MEDIAPIPE
# ============================================================

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# ============================================================
# 3. OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("\nERROR: Could not open video!")
    print("Check VIDEO_PATH:")
    print(VIDEO_PATH)
    exit()

print("\nVideo opened successfully!")
print("Press Q to quit.")


# ============================================================
# 4. STORE PREDICTIONS
# ============================================================

prediction_counts = Counter()

total_frames = 0
successful_frames = 0


# ============================================================
# 5. PROCESS VIDEO
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    total_frames += 1

    # --------------------------------------------------------
    # Convert BGR -> RGB
    # --------------------------------------------------------

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # --------------------------------------------------------
    # Pose detection
    # --------------------------------------------------------

    results = pose.process(rgb_frame)

    prediction = "No pose detected"
    confidence_text = ""

    if results.pose_landmarks:

        successful_frames += 1

        # ----------------------------------------------------
        # Extract 33 landmarks
        # ----------------------------------------------------

        features = []

        for landmark in results.pose_landmarks.landmark:

            features.extend([
                landmark.x,
                landmark.y,
                landmark.z,
                landmark.visibility
            ])

        # ----------------------------------------------------
        # Convert to DataFrame
        # ----------------------------------------------------

        feature_df = pd.DataFrame(
            [features],
            columns=model.feature_names_in_
        )

        # ----------------------------------------------------
        # Predict
        # ----------------------------------------------------

        prediction = model.predict(feature_df)[0]

        # ----------------------------------------------------
        # Prediction probability
        # ----------------------------------------------------

        probabilities = model.predict_proba(feature_df)[0]

        best_probability = max(probabilities)

        confidence_text = f"Confidence: {best_probability * 100:.1f}%"

        # Count prediction
        prediction_counts[prediction] += 1

        # ----------------------------------------------------
        # Draw pose
        # ----------------------------------------------------

        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.putText(
        frame,
        f"Exercise: {prediction}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    if confidence_text:

        cv2.putText(
            frame,
            confidence_text,
            (30, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

    cv2.imshow("AI Gym Trainer", frame)

    # Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ============================================================
# 6. CLEANUP
# ============================================================

cap.release()
cv2.destroyAllWindows()
pose.close()


# ============================================================
# 7. FINAL RESULT
# ============================================================

print("\n")
print("=" * 60)
print("VIDEO ANALYSIS COMPLETED")
print("=" * 60)

print(f"Total frames processed     : {total_frames}")
print(f"Frames with pose detected  : {successful_frames}")

print("\nPrediction counts:")
print("-" * 40)

for exercise, count in prediction_counts.most_common():
    percentage = (count / successful_frames) * 100

    print(
        f"{exercise:<25} : {count:4d} "
        f"({percentage:.1f}%)"
    )


# ============================================================
# 8. MAJORITY VOTE
# ============================================================

if prediction_counts:

    final_prediction = prediction_counts.most_common(1)[0][0]

    print("\n" + "=" * 60)
    print(f"FINAL EXERCISE: {final_prediction.upper()}")
    print("=" * 60)

else:

    print("\nNo pose was detected in the video.")


print("\nVideo processing completed!")