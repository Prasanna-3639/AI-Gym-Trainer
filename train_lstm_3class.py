import numpy as np
import tensorflow as tf
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# ============================================================
# AI GYM TRAINER - 3 CLASS LSTM TRAINING
# ============================================================

DATA_X = "dataset/sequence_X.npy"
DATA_Y = "dataset/sequence_y.npy"

MODEL_PATH = "models/lstm_3class_exercise_model.keras"
SCALER_PATH = "models/lstm_3class_scaler.pkl"
ENCODER_PATH = "models/lstm_3class_label_encoder.pkl"


print("=" * 60)
print("AI GYM TRAINER - 3 CLASS LSTM TRAINING")
print("=" * 60)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading dataset...")

X = np.load(DATA_X)

y = np.load(
    DATA_Y,
    allow_pickle=True
)

print("X shape:", X.shape)
print("y shape:", y.shape)


# ============================================================
# DATA INFORMATION
# ============================================================

print("\nFrames per sequence:", X.shape[1])
print("Features per frame:", X.shape[2])

print("\nExercise distribution:")

unique, counts = np.unique(
    y,
    return_counts=True
)

for label, count in zip(
    unique,
    counts
):

    print(
        f"{label:20s}: {count}"
    )


# ============================================================
# LABEL ENCODING
# ============================================================

print("\nEncoding labels...")

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(
    y
)

print("\nClasses:")

for index, class_name in enumerate(
    label_encoder.classes_
):

    print(
        index,
        "->",
        class_name
    )


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y_encoded,

    test_size=0.20,

    random_state=42,

    stratify=y_encoded
)


print(
    "\nTraining samples:",
    X_train.shape[0]
)

print(
    "Testing samples:",
    X_test.shape[0]
)


# ============================================================
# SCALE FEATURES
# ============================================================

print("\nScaling features...")

sequence_length = X_train.shape[1]

n_features = X_train.shape[2]


# Flatten only temporarily
X_train_2d = X_train.reshape(
    -1,
    n_features
)

X_test_2d = X_test.reshape(
    -1,
    n_features
)


scaler = StandardScaler()


X_train_scaled = scaler.fit_transform(
    X_train_2d
)

X_test_scaled = scaler.transform(
    X_test_2d
)


# Restore sequence shape
X_train_scaled = X_train_scaled.reshape(
    X_train.shape[0],
    sequence_length,
    n_features
)

X_test_scaled = X_test_scaled.reshape(
    X_test.shape[0],
    sequence_length,
    n_features
)


# ============================================================
# BUILD 3-CLASS LSTM
# ============================================================

print("\nBuilding LSTM model...")


model = Sequential([

    tf.keras.Input(
        shape=(
            sequence_length,
            n_features
        )
    ),

    LSTM(
        128,
        return_sequences=True
    ),

    Dropout(0.30),

    LSTM(
        64
    ),

    Dropout(0.30),

    Dense(
        32,
        activation="relu"
    ),

    Dropout(0.20),

    Dense(
        3,
        activation="softmax"
    )
])


# ============================================================
# COMPILE
# ============================================================

model.compile(

    optimizer="adam",

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]
)


print("\nModel summary:")

model.summary()


# ============================================================
# EARLY STOPPING
# ============================================================

early_stopping = EarlyStopping(

    monitor="val_loss",

    patience=12,

    restore_best_weights=True
)


# ============================================================
# TRAIN
# ============================================================

print("\nTraining 3-class LSTM...")

history = model.fit(

    X_train_scaled,

    y_train,

    validation_split=0.20,

    epochs=100,

    batch_size=16,

    callbacks=[
        early_stopping
    ],

    shuffle=True,

    verbose=1
)


# ============================================================
# TEST MODEL
# ============================================================

print("\n" + "=" * 60)
print("3 CLASS LSTM TEST RESULTS")
print("=" * 60)


test_loss, test_accuracy = model.evaluate(

    X_test_scaled,

    y_test,

    verbose=0
)


print(
    f"\nTest Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)


# ============================================================
# PREDICTIONS
# ============================================================

probabilities = model.predict(

    X_test_scaled,

    verbose=0
)


predicted_indices = np.argmax(

    probabilities,

    axis=1
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")


print(
    classification_report(

        y_test,

        predicted_indices,

        labels=np.arange(
            len(
                label_encoder.classes_
            )
        ),

        target_names=(
            label_encoder.classes_
        ),

        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\nConfusion Matrix:")


cm = confusion_matrix(

    y_test,

    predicted_indices,

    labels=np.arange(
        len(
            label_encoder.classes_
        )
    )
)


print(cm)


# ============================================================
# SAVE MODEL
# ============================================================

print("\nSaving model...")


model.save(
    MODEL_PATH
)


joblib.dump(
    scaler,
    SCALER_PATH
)


joblib.dump(
    label_encoder,
    ENCODER_PATH
)


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n" + "=" * 60)
print("3 CLASS LSTM TRAINING COMPLETED")
print("=" * 60)


print("\nModel saved:")
print(
    MODEL_PATH
)


print("\nScaler saved:")
print(
    SCALER_PATH
)


print("\nLabel encoder saved:")
print(
    ENCODER_PATH
)


print("\nFinal Test Accuracy:")

print(
    f"{test_accuracy * 100:.2f}%"
)


print("=" * 60)