# ============================================================
# EXP 3 : Transfer Learning
# MobileNetV2 on CIFAR-10 (Cats vs Dogs)
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

import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Load CIFAR-10 Dataset
# ------------------------------------------------------------

print("\nLoading Dataset...")

(x_train_all, y_train_all), (x_test_all, y_test_all) = cifar10.load_data()


# ------------------------------------------------------------
# Filter Only Cats and Dogs
# Cat = 3
# Dog = 5
# ------------------------------------------------------------

def filter_data(x, y):

    mask = (y == 3) | (y == 5)

    x = x[mask.flatten()]

    y = y[mask.flatten()]

    # Convert labels:
    # cat = 0
    # dog = 1

    y = np.where(y == 3, 0, 1)

    return x, y


x_train, y_train = filter_data(x_train_all, y_train_all)

x_test, y_test = filter_data(x_test_all, y_test_all)


# ------------------------------------------------------------
# Use Smaller Dataset for Faster Training
# ------------------------------------------------------------

x_train = x_train[:2000]

y_train = y_train[:2000]

x_test = x_test[:500]

y_test = y_test[:500]


# ------------------------------------------------------------
# Resize Images for MobileNetV2
# ------------------------------------------------------------

x_train = tf.image.resize(x_train, (96,96)) / 255.0

x_test = tf.image.resize(x_test, (96,96)) / 255.0


# ============================================================
# MODEL 1 : Frozen Base Model
# Feature Extraction
# ============================================================

print("\n================================================")
print("MODEL 1 : FEATURE EXTRACTION")
print("================================================")


# Load Pretrained MobileNetV2

base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(96,96,3)
)

# Freeze all pretrained layers

base_model.trainable = False


# Create New Model

model_frozen = Sequential([

    base_model,

    GlobalAveragePooling2D(),

    Dense(64, activation='relu'),

    Dropout(0.3),

    Dense(1, activation='sigmoid')

])


# Compile Model

model_frozen.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)


# Print Model Summary

model_frozen.summary()


# Train Model

print("\nTraining Frozen Model...")

history_frozen = model_frozen.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)


# Evaluate Model

loss_frozen, acc_frozen = model_frozen.evaluate(
    x_test,
    y_test,
    verbose=0
)

print(f"\nFrozen Model Accuracy : {acc_frozen:.4f}")


# ============================================================
# MODEL 2 : Partial Fine Tuning
# ============================================================

print("\n================================================")
print("MODEL 2 : PARTIAL FINE TUNING")
print("================================================")


# Load Pretrained MobileNetV2

base_model_ft = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(96,96,3)
)

# Unfreeze model

base_model_ft.trainable = True


# Freeze first layers

for layer in base_model_ft.layers[:-20]:

    layer.trainable = False


# Create New Model

model_ft = Sequential([

    base_model_ft,

    GlobalAveragePooling2D(),

    Dense(64, activation='relu'),

    Dropout(0.3),

    Dense(1, activation='sigmoid')

])


# Compile Model

model_ft.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)


# Print Model Summary

model_ft.summary()


# Train Model

print("\nTraining Fine Tuned Model...")

history_ft = model_ft.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)


# Evaluate Model

loss_ft, acc_ft = model_ft.evaluate(
    x_test,
    y_test,
    verbose=0
)

print(f"\nFine Tuned Model Accuracy : {acc_ft:.4f}")


# ============================================================
# Accuracy Graph
# ============================================================

plt.figure(figsize=(10,5))

plt.plot(
    history_frozen.history['val_accuracy'],
    label='Frozen Model'
)

plt.plot(
    history_ft.history['val_accuracy'],
    label='Fine Tuned Model'
)

plt.title("Validation Accuracy Comparison")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.grid(True)

plt.show()


# ============================================================
# Final Results
# ============================================================

print("\n================================================")
print("FINAL RESULTS")
print("================================================")

print(f"Frozen Model Accuracy      : {acc_frozen:.4f}")

print(f"Fine Tuned Model Accuracy  : {acc_ft:.4f}")
