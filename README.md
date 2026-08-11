# 🏋️ AI Gym Trainer

An AI-powered fitness application that detects exercises and counts repetitions using **LSTM, MediaPipe Pose Estimation, OpenCV, and Streamlit**.

## 🚀 Features

- 🏋️ Squat detection and rep counting
- 💪 Push-up detection and rep counting
- 🏋️‍♂️ Shoulder press detection and rep counting
- 🎥 Live webcam workout tracking
- 📹 Upload-video analysis
- 🧠 LSTM-based exercise recognition
- 🦾 MediaPipe pose estimation
- 📊 50 pose-based features
- 🔒 Exercise locking to prevent cross-exercise counting
- 📈 Live confidence and repetition display

## 🧠 Exercises Supported

- Squat
- Push-up
- Shoulder Press

## 🛠️ Technologies

- Python
- TensorFlow
- Keras
- LSTM
- MediaPipe
- OpenCV
- NumPy
- Scikit-learn
- Streamlit
- Streamlit-WebRTC
- Joblib

## 🔄 System Pipeline

```text
Live Webcam / Video
        ↓
MediaPipe Pose Estimation
        ↓
50 Pose Features
        ↓
Feature Scaling
        ↓
Sequence Creation
        ↓
LSTM Model
        ↓
Exercise Detection
        ↓
Exercise Locking
        ↓
Rep Counting
        ↓
Streamlit Interface
