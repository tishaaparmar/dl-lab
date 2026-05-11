# ============================================================
# EXP 4 : CNN vs Transfer Learning
# CIFAR-10 (Cats vs Dogs)
# ============================================================

# ------------------------------------------------------------
# Import Libraries
# ------------------------------------------------------------

import warnings
warnings.filterwarnings("ignore")

import time
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    GlobalAveragePooling2D
)

from tensorflow.keras.applications import MobileNetV2

from tensorflow.keras.datasets import cifar10

import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Load Dataset
# ------------------------------------------------------------

print("\nLoading Dataset...")

(x_train_all, y_train_all), (x_test_all, y_test_all) = cifar10.load_data()


# ------------------------------------------------------------
# Filter Cats and Dogs
# Cat = 3
# Dog = 5
# ------------------------------------------------------------

def filter_data(x, y):

    mask = (y == 3) | (y == 5)

    x = x[mask.flatten()]

    y = y[mask.flatten()]

    # Convert labels
    # Cat = 0
    # Dog = 1

    y = np.where(y == 3, 0, 1)

    return x, y


x_train_full, y_train_full = filter_data(
    x_train_all,
    y_train_all
)

x_test, y_test = filter_data(
    x_test_all,
    y_test_all
)


# ============================================================
# Compare Different Dataset Sizes
# ============================================================

dataset_sizes = [500, 2000]

results = {}


# ============================================================
# Run Experiment for Each Dataset Size
# ============================================================

for size in dataset_sizes:

    print("\n================================================")
    print(f"DATASET SIZE = {size}")
    print("================================================")


    # --------------------------------------------------------
    # Prepare Dataset
    # --------------------------------------------------------

    x_train = x_train_full[:size]

    y_train = y_train_full[:size]

    x_test_small = x_test[:500]

    y_test_small = y_test[:500]


    # CNN Input

    x_train_cnn = x_train / 255.0

    x_test_cnn = x_test_small / 255.0


    # Transfer Learning Input

    x_train_tl = tf.image.resize(
        x_train,
        (96,96)
    ) / 255.0

    x_test_tl = tf.image.resize(
        x_test_small,
        (96,96)
    ) / 255.0


    # ========================================================
    # MODEL 1 : CNN FROM SCRATCH
    # ========================================================

    print("\nTraining CNN From Scratch...")


    cnn = Sequential([

        Conv2D(
            32,
            (3,3),
            activation='relu',
            input_shape=(32,32,3)
        ),

        MaxPooling2D((2,2)),

        Conv2D(
            64,
            (3,3),
            activation='relu'
        ),

        MaxPooling2D((2,2)),

        Flatten(),

        Dense(128, activation='relu'),

        Dropout(0.3),

        Dense(1, activation='sigmoid')

    ])


    # Compile Model

    cnn.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )


    # Record Training Time

    start = time.time()


    # Train Model

    history_cnn = cnn.fit(
        x_train_cnn,
        y_train,
        epochs=5,
        batch_size=32,
        validation_split=0.1,
        verbose=0
    )


    # Training Time

    cnn_time = time.time() - start


    # Evaluate Model

    loss_cnn, acc_cnn = cnn.evaluate(
        x_test_cnn,
        y_test_small,
        verbose=0
    )


    print(f"CNN Accuracy : {acc_cnn:.4f}")

    print(f"CNN Time     : {cnn_time:.2f} sec")


    # ========================================================
    # MODEL 2 : TRANSFER LEARNING
    # ========================================================

    print("\nTraining Transfer Learning Model...")


    # Load MobileNetV2

    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(96,96,3)
    )


    # Freeze Layers

    base_model.trainable = False


    # Create Transfer Learning Model

    tl_model = Sequential([

        base_model,

        GlobalAveragePooling2D(),

        Dense(64, activation='relu'),

        Dropout(0.3),

        Dense(1, activation='sigmoid')

    ])


    # Compile Model

    tl_model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )


    # Record Training Time

    start = time.time()


    # Train Model

    history_tl = tl_model.fit(
        x_train_tl,
        y_train,
        epochs=5,
        batch_size=32,
        validation_split=0.1,
        verbose=0
    )


    # Training Time

    tl_time = time.time() - start


    # Evaluate Model

    loss_tl, acc_tl = tl_model.evaluate(
        x_test_tl,
        y_test_small,
        verbose=0
    )


    print(f"Transfer Learning Accuracy : {acc_tl:.4f}")

    print(f"Transfer Learning Time     : {tl_time:.2f} sec")


    # Save Results

    results[size] = {
        'cnn_acc': acc_cnn,
        'cnn_time': cnn_time,
        'tl_acc': acc_tl,
        'tl_time': tl_time
    }


# ============================================================
# Accuracy Comparison Graph
# ============================================================

sizes = list(results.keys())

cnn_acc = [results[s]['cnn_acc'] for s in sizes]

tl_acc = [results[s]['tl_acc'] for s in sizes]


plt.figure(figsize=(8,5))

plt.plot(
    sizes,
    cnn_acc,
    marker='o',
    label='CNN From Scratch'
)

plt.plot(
    sizes,
    tl_acc,
    marker='o',
    label='Transfer Learning'
)

plt.title("Accuracy Comparison")

plt.xlabel("Dataset Size")

plt.ylabel("Accuracy")

plt.legend()

plt.grid(True)

plt.show()


# ============================================================
# Training Time Graph
# ============================================================

cnn_time = [results[s]['cnn_time'] for s in sizes]

tl_time = [results[s]['tl_time'] for s in sizes]


plt.figure(figsize=(8,5))

plt.plot(
    sizes,
    cnn_time,
    marker='o',
    label='CNN From Scratch'
)

plt.plot(
    sizes,
    tl_time,
    marker='o',
    label='Transfer Learning'
)

plt.title("Training Time Comparison")

plt.xlabel("Dataset Size")

plt.ylabel("Time (seconds)")

plt.legend()

plt.grid(True)

plt.show()


# ============================================================
# Final Results
# ============================================================

print("\n================================================")
print("FINAL RESULTS")
print("================================================")

for size in sizes:

    print(f"\nDataset Size : {size}")

    print(f"CNN Accuracy : {results[size]['cnn_acc']:.4f}")

    print(f"TL Accuracy  : {results[size]['tl_acc']:.4f}")

    print(f"CNN Time     : {results[size]['cnn_time']:.2f} sec")

    print(f"TL Time      : {results[size]['tl_time']:.2f} sec")
