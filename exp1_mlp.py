# ============================================================
# EXP 1 : MLP (Multilayer Perceptron)
# MNIST Digit Classification
# ============================================================

# ------------------------------------------------------------
# Import Libraries
# ------------------------------------------------------------


import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Load Dataset
# ------------------------------------------------------------

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Use smaller dataset for faster execution

x_train = x_train[:10000]
y_train = y_train[:10000]

x_test = x_test[:2000]
y_test = y_test[:2000]


# ------------------------------------------------------------
# Data Preprocessing
# ------------------------------------------------------------

# Normalize pixel values (0-255 → 0-1)

x_train = x_train / 255.0
x_test = x_test / 255.0

# Convert output labels into categorical format

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


# ============================================================
# EXPERIMENT 1A
# Effect of Activation Functions
# ============================================================

print("\n================================================")
print("ACTIVATION FUNCTION ANALYSIS")
print("================================================")


# Different activation functions

activations = ['relu', 'sigmoid', 'tanh']

# Store validation accuracy for plotting

activation_history = {}


# ------------------------------------------------------------
# Train Model using Different Activation Functions
# ------------------------------------------------------------

for act in activations:

    print(f"\nTraining using {act} activation function")

    # Create Model

    model = Sequential()

    model.add(Input(shape=(28,28)))

    model.add(Flatten())

    model.add(Dense(128, activation=act))

    model.add(Dense(64, activation=act))

    model.add(Dense(10, activation='softmax'))

    
    # Compile Model

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )


    # Train Model

    history = model.fit(
        x_train,
        y_train,
        epochs=5,
        batch_size=128,
        validation_split=0.1,
        verbose=0
    )


    # Evaluate Model

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Loss     : {loss:.4f}")

    
    # Store history for graph

    activation_history[act] = history.history['val_accuracy']


# ============================================================
# EXPERIMENT 1B
# Effect of Hidden Layers
# ============================================================

print("\n================================================")
print("HIDDEN LAYER ANALYSIS")
print("================================================")


# Different hidden layer configurations

layer_configs = {
    "1 Hidden Layer": [128],
    "2 Hidden Layers": [128, 64],
    "3 Hidden Layers": [128, 64, 32]
}

# Store validation accuracy

layer_history = {}


# ------------------------------------------------------------
# Train Model using Different Hidden Layers
# ------------------------------------------------------------

for name, layers in layer_configs.items():

    print(f"\nTraining using {name}")

    
    # Create Model

    model = Sequential()

    model.add(Input(shape=(28,28)))

    model.add(Flatten())


    # Add hidden layers

    for units in layers:

        model.add(Dense(units, activation='relu'))


    # Output Layer

    model.add(Dense(10, activation='softmax'))


    # Compile Model

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )


    # Train Model

    history = model.fit(
        x_train,
        y_train,
        epochs=5,
        batch_size=128,
        validation_split=0.1,
        verbose=0
    )


    # Evaluate Model

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Loss     : {loss:.4f}")


    # Store history for graph

    layer_history[name] = history.history['val_accuracy']


# ============================================================
# Plot Graphs
# ============================================================

plt.figure(figsize=(12,5))


# ------------------------------------------------------------
# Activation Function Graph
# ------------------------------------------------------------

plt.subplot(1,2,1)

for act, values in activation_history.items():

    plt.plot(values, label=act)

plt.title("Activation Function Comparison")
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.legend()
plt.grid(True)


# ------------------------------------------------------------
# Hidden Layer Graph
# ------------------------------------------------------------

plt.subplot(1,2,2)

for name, values in layer_history.items():

    plt.plot(values, label=name)

plt.title("Hidden Layer Comparison")
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.legend()
plt.grid(True)


plt.tight_layout()

plt.savefig("exp1_mlp_output.png")

plt.show()


