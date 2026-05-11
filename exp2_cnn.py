# ============================================================
# EXP 2 : CNN (Convolutional Neural Network)
# Fashion-MNIST Image Classification
# ============================================================

# ------------------------------------------------------------
# Import Libraries
# ------------------------------------------------------------

import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Input
)

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical

import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Load Dataset
# ------------------------------------------------------------

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Use smaller dataset for faster execution

x_train = x_train[:10000]
y_train = y_train[:10000]

x_test = x_test[:2000]
y_test = y_test[:2000]


# ------------------------------------------------------------
# Data Preprocessing
# ------------------------------------------------------------

# CNN input shape = (28,28,1)

x_train_cnn = x_train.reshape(-1,28,28,1) / 255.0
x_test_cnn = x_test.reshape(-1,28,28,1) / 255.0

# MLP input shape = (784)

x_train_mlp = x_train.reshape(-1,784) / 255.0
x_test_mlp = x_test.reshape(-1,784) / 255.0

# Convert labels into categorical format

y_train = to_categorical(y_train,10)
y_test = to_categorical(y_test,10)


# ============================================================
# MODEL 1 : Fully Connected Network (MLP)
# ============================================================

mlp = Sequential()

mlp.add(Dense(
    128,
    activation='relu',
    input_shape=(784,)
))

mlp.add(Dense(64, activation='relu'))

mlp.add(Dense(10, activation='softmax'))


# Compile Model

mlp.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# Print Model Summary

print("\n================================================")
print("MLP MODEL")
print("================================================")

mlp.summary()


# Train Model

print("\nTraining MLP Model...")

history_mlp = mlp.fit(
    x_train_mlp,
    y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.1,
    verbose=0
)


# Evaluate Model

loss_mlp, acc_mlp = mlp.evaluate(
    x_test_mlp,
    y_test,
    verbose=0
)

print(f"\nMLP Accuracy : {acc_mlp:.4f}")


# ============================================================
# MODEL 2 : CNN with MaxPooling
# ============================================================

cnn = Sequential()

# Input Layer

cnn.add(Input(shape=(28,28,1)))

# First Convolution Layer

cnn.add(Conv2D(
    32,
    (3,3),
    activation='relu'
))

# First Pooling Layer

cnn.add(MaxPooling2D((2,2)))

# Second Convolution Layer

cnn.add(Conv2D(
    64,
    (3,3),
    activation='relu'
))

# Second Pooling Layer

cnn.add(MaxPooling2D((2,2)))

# Flatten Layer

cnn.add(Flatten())

# Fully Connected Layer

cnn.add(Dense(128, activation='relu'))

# Output Layer

cnn.add(Dense(10, activation='softmax'))


# Compile Model

cnn.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# Print Model Summary

print("\n================================================")
print("CNN MODEL")
print("================================================")

cnn.summary()


# Train Model

print("\nTraining CNN Model...")

history_cnn = cnn.fit(
    x_train_cnn,
    y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.1,
    verbose=0
)


# Evaluate Model

loss_cnn, acc_cnn = cnn.evaluate(
    x_test_cnn,
    y_test,
    verbose=0
)

print(f"\nCNN Accuracy : {acc_cnn:.4f}")


# ============================================================
# MODEL 3 : CNN with Different Kernel Size (5x5)
# ============================================================

cnn_5x5 = Sequential()

cnn_5x5.add(Input(shape=(28,28,1)))

cnn_5x5.add(Conv2D(
    32,
    (5,5),
    activation='relu'
))

cnn_5x5.add(MaxPooling2D((2,2)))

cnn_5x5.add(Flatten())

cnn_5x5.add(Dense(128, activation='relu'))

cnn_5x5.add(Dense(10, activation='softmax'))


# Compile Model

cnn_5x5.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# Print Model Summary

print("\n================================================")
print("CNN 5x5 MODEL")
print("================================================")

cnn_5x5.summary()


# Train Model

print("\nTraining CNN with 5x5 Kernel...")

history_5x5 = cnn_5x5.fit(
    x_train_cnn,
    y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.1,
    verbose=0
)


# Evaluate Model

loss_5x5, acc_5x5 = cnn_5x5.evaluate(
    x_test_cnn,
    y_test,
    verbose=0
)

print(f"\nCNN 5x5 Accuracy : {acc_5x5:.4f}")


# ============================================================
# Feature Map Visualization
# ============================================================

print("\nGenerating Feature Maps...")

# Create Feature Map Model

feature_model = Model(
    inputs=cnn.inputs,
    outputs=cnn.layers[0].output
)

# Take one sample image

sample = x_test_cnn[0:1]

# Generate Feature Maps

feature_maps = feature_model.predict(sample, verbose=0)


# Plot Feature Maps

plt.figure(figsize=(10,10))

for i in range(16):

    plt.subplot(4,4,i+1)

    plt.imshow(
        feature_maps[0,:,:,i],
        cmap='viridis'
    )

    plt.axis('off')

plt.suptitle("Feature Maps from First Conv Layer")

plt.tight_layout()

plt.show()


# ============================================================
# Accuracy Comparison Graph
# ============================================================

models = ['MLP', 'CNN', 'CNN 5x5']

accuracies = [
    acc_mlp,
    acc_cnn,
    acc_5x5
]

plt.figure(figsize=(8,5))

plt.bar(models, accuracies)

plt.title("Model Accuracy Comparison")

plt.ylabel("Accuracy")

plt.ylim(0.8,1.0)

plt.grid(True)

plt.show()


# ============================================================
# Final Results
# ============================================================

print("\n================================================")
print("FINAL RESULTS")
print("================================================")

print(f"MLP Accuracy       : {acc_mlp:.4f}")

print(f"CNN Accuracy       : {acc_cnn:.4f}")

print(f"CNN 5x5 Accuracy   : {acc_5x5:.4f}")
