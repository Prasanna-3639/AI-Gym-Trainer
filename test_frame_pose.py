import cv2
import mediapipe as mp
import os

# ==========================================
# TEST MEDIAPIPE ON EXTRACTED SQUAT FRAME
# ==========================================

frames_folder = "dataset/frames/squat"

# Find video folders
video_folders = [
    f for f in os.listdir(frames_folder)
    if os.path.isdir(os.path.join(frames_folder, f))
]

if not video_folders:
    print("ERROR: No frame folders found!")
    exit()

first_folder = video_folders[0]

folder_path = os.path.join(
    frames_folder,
    first_folder
)

# Find first extracted frame
frames = [
    f for f in os.listdir(folder_path)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

if not frames:
    print("ERROR: No frames found!")
    exit()

image_path = os.path.join(folder_path, frames[0])

print("Testing image:")
print(image_path)

# Load image
image = cv2.imread(image_path)

if image is None:
    print("ERROR: Could not load image!")
    exit()

print("Image loaded successfully!")
print("Image shape:", image.shape)

# ==========================================
# MEDIAPIPE POSE
# ==========================================

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=1,
    min_detection_confidence=0.5
)

rgb_image = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)

results = pose.process(rgb_image)

# ==========================================
# CHECK RESULT
# ==========================================

if results.pose_landmarks:

    print()
    print("Pose detected successfully!")
    print(
        "Number of landmarks:",
        len(results.pose_landmarks.landmark)
    )

    # Draw landmarks
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS
    )

else:

    print()
    print("No pose detected.")

# ==========================================
# DISPLAY
# ==========================================

cv2.imshow("Squat Pose Test", image)

print()
print("Press Q to close the window.")

while True:

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()

pose.close()

print("Test completed.")