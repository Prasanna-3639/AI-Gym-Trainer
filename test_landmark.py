import cv2
import mediapipe as mp
import os

# -------------------------------------------------
# 1. Folder containing one sequence
# -------------------------------------------------

folder_path = os.path.join(
    "dataset",
    "raw",
    "squat",
    "000000_img_labels"
)

print("Checking folder:")
print(folder_path)

if not os.path.exists(folder_path):
    print("\nERROR: Folder not found!")
    exit()

# -------------------------------------------------
# 2. Find the first ORIGINAL image
# -------------------------------------------------

image_file = None

for file in os.listdir(folder_path):

    # Ignore segmentation/label images
    if ".cseg." in file.lower() or ".iseg." in file.lower():
        continue

    if file.lower().endswith((".png", ".jpg", ".jpeg")):
        image_file = file
        break

if image_file is None:
    print("\nERROR: No original image found!")
    exit()

image_path = os.path.join(folder_path, image_file)

print("\nImage found:")
print(image_file)

# -------------------------------------------------
# 3. Load image
# -------------------------------------------------

image = cv2.imread(image_path)

if image is None:
    print("\nERROR: Could not read image!")
    exit()

print("\nImage loaded successfully!")
print("Image shape:", image.shape)

# -------------------------------------------------
# 4. MediaPipe Pose
# -------------------------------------------------

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=1,
    min_detection_confidence=0.5
)

# -------------------------------------------------
# 5. Convert BGR → RGB
# -------------------------------------------------

rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# -------------------------------------------------
# 6. Detect pose
# -------------------------------------------------

results = pose.process(rgb_image)

# -------------------------------------------------
# 7. Draw landmarks
# -------------------------------------------------

if results.pose_landmarks:

    print("Pose detected successfully!")
    print("Number of landmarks:", len(results.pose_landmarks.landmark))

    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS
    )

else:

    print("No pose detected.")

# -------------------------------------------------
# 8. Display
# -------------------------------------------------

cv2.imshow("MediaPipe Pose Test", image)

print("\nPress Q to close the window.")

while True:

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()

pose.close()

print("Test completed.")