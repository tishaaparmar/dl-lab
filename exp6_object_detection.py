# ============================================================
# EXP 6 : Object Detection
# CIFAR-10 Object Detection using MobileNetV2
# ============================================================

# ------------------------------------------------------------
# Import Libraries
# ------------------------------------------------------------

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import tensorflow as tf

from tensorflow.keras.applications import MobileNetV2

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    GlobalAveragePooling2D,
    Dense,
    Dropout
)

from tensorflow.keras.datasets import cifar10

from tensorflow.keras.utils import to_categorical

import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Load CIFAR-10 Dataset
# ------------------------------------------------------------

print("\nLoading Dataset...")

(x_train, y_train), (x_test, y_test) = cifar10.load_data()


# CIFAR-10 Class Names

class_names = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
]


# ------------------------------------------------------------
# Use Smaller Dataset
# ------------------------------------------------------------

x_train = x_train[:6000]

y_train = y_train[:6000]

x_test = x_test[:1000]

y_test = y_test[:1000]


# ------------------------------------------------------------
# Data Preprocessing
# ------------------------------------------------------------

x_train = x_train / 255.0

x_test = x_test / 255.0


# Convert Labels into Categorical Format

y_train_cat = to_categorical(y_train, 10)


# Resize Images for MobileNetV2

x_train_resized = tf.image.resize(
    x_train,
    (96,96)
) / 255.0

x_test_resized = tf.image.resize(
    x_test,
    (96,96)
) / 255.0


# ============================================================
# Build Transfer Learning Model
# ============================================================

print("\nBuilding MobileNetV2 Model...")


# Load Pretrained MobileNetV2

base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(96,96,3)
)


# Freeze Pretrained Layers

base_model.trainable = False


# Create Model

model = Sequential([

    base_model,

    GlobalAveragePooling2D(),

    Dense(64, activation='relu'),

    Dropout(0.3),

    Dense(10, activation='softmax')

])


# Compile Model

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# Print Model Summary

model.summary()


# ============================================================
# Train Model
# ============================================================

print("\nTraining Model...")

model.fit(
    x_train_resized,
    y_train_cat,
    epochs=5,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)


# ============================================================
# Object Detection on Test Images
# ============================================================

print("\nDetecting Objects...")


# Select Sample Images

sample_images = x_test[:6]

sample_images_resized = x_test_resized[:6]


# Predict Objects

predictions = model.predict(
    sample_images_resized,
    verbose=0
)


# ============================================================
# Display Predictions
# ============================================================

plt.figure(figsize=(15,6))

for i in range(6):

    plt.subplot(2,3,i+1)

    plt.imshow(sample_images[i])

    # Predicted Class

    pred_class = np.argmax(predictions[i])

    pred_name = class_names[pred_class]

    # Confidence Score

    confidence = predictions[i][pred_class]

    # True Label

    true_name = class_names[
        y_test[i][0]
    ]


    # Correct or Incorrect

    color = 'green'

    if pred_name != true_name:

        color = 'red'


    plt.title(
        f"True : {true_name}\n"
        f"Pred : {pred_name}\n"
        f"Conf : {confidence:.2f}",
        color=color,
        fontsize=9
    )

    plt.axis('off')

plt.tight_layout()

plt.show()


# ============================================================
# Confidence Score Analysis
# ============================================================

print("\n================================================")
print("CONFIDENCE SCORE ANALYSIS")
print("================================================")


for i in range(6):

    pred_class = np.argmax(predictions[i])

    pred_name = class_names[pred_class]

    confidence = predictions[i][pred_class]

    true_name = class_names[y_test[i][0]]

    print(f"\nImage {i+1}")

    print(f"True Label      : {true_name}")

    print(f"Predicted Label : {pred_name}")

    print(f"Confidence      : {confidence:.4f}")


# ============================================================
# Image Quality Analysis
# ============================================================

print("\n================================================")
print("IMAGE QUALITY ANALYSIS")
print("================================================")


# Take One Test Image

test_image = x_test_resized[0:1]


# Different Noise Levels

noise_levels = [0, 0.05, 0.1, 0.2, 0.4]

confidence_scores = []


for noise in noise_levels:

    # Add Noise

    noisy_image = np.clip(

        test_image +

        np.random.normal(
            0,
            noise,
            test_image.shape
        ),

        0,
        1
    )


    # Predict

    prediction = model.predict(
        noisy_image,
        verbose=0
    )

    confidence = np.max(prediction)

    confidence_scores.append(confidence)

    print(f"Noise = {noise}  --> Confidence = {confidence:.4f}")


# ============================================================
# Plot Quality vs Confidence
# ============================================================

plt.figure(figsize=(8,5))

plt.plot(
    noise_levels,
    confidence_scores,
    marker='o'
)

plt.title("Image Quality vs Confidence")

plt.xlabel("Noise Level")

plt.ylabel("Confidence Score")

plt.grid(True)

plt.show()
