# ============================================================
# EXP 5 : Performance Analysis of CNN
# CIFAR-10 Classification
# ============================================================

# ------------------------------------------------------------
# Import Libraries
# ------------------------------------------------------------

import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)

from tensorflow.keras.datasets import cifar10

from tensorflow.keras.utils import to_categorical

import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Load Dataset
# ------------------------------------------------------------

print("\nLoading CIFAR-10 Dataset...")

(x_train, y_train), (x_test, y_test) = cifar10.load_data()


# Use Smaller Dataset

x_train = x_train[:8000]

y_train = y_train[:8000]

x_test = x_test[:2000]

y_test = y_test[:2000]


# ------------------------------------------------------------
# Data Preprocessing
# ------------------------------------------------------------

x_train = x_train / 255.0

x_test = x_test / 255.0


# Convert Labels into Categorical Format

y_train = to_categorical(y_train, 10)

y_test = to_categorical(y_test, 10)


# ============================================================
# CNN Model Function
# ============================================================

def create_model(learning_rate=0.001, dropout_rate=0.3):

    model = Sequential([

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

        Dropout(dropout_rate),

        Dense(10, activation='softmax')

    ])


    # Compile Model

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate
        ),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# ============================================================
# EXPERIMENT 1 : Effect of Learning Rate
# ============================================================

print("\n================================================")
print("EXPERIMENT 1 : LEARNING RATE ANALYSIS")
print("================================================")


learning_rates = [0.1, 0.01, 0.001, 0.0001]

lr_results = {}

lr_history = {}


for lr in learning_rates:

    print(f"\nTraining with Learning Rate = {lr}")

    model = create_model(learning_rate=lr)

    history = model.fit(
        x_train,
        y_train,
        epochs=10,
        batch_size=128,
        validation_split=0.1,
        verbose=0
    )


    # Evaluate Model

    loss, accuracy = model.evaluate(
        x_test,
        y_test,
        verbose=0
    )

    print(f"Accuracy : {accuracy:.4f}")

    print(f"Loss     : {loss:.4f}")


    # Store Results

    lr_results[lr] = accuracy

    lr_history[lr] = history


# ============================================================
# EXPERIMENT 2 : Overfitting Analysis
# ============================================================

print("\n================================================")
print("EXPERIMENT 2 : OVERFITTING ANALYSIS")
print("================================================")


# Model Without Dropout

print("\nTraining Model Without Dropout...")

model_overfit = create_model(
    learning_rate=0.001,
    dropout_rate=0.0
)

history_overfit = model_overfit.fit(
    x_train,
    y_train,
    epochs=30,
    batch_size=128,
    validation_split=0.1,
    verbose=0
)


# Model With Dropout

print("\nTraining Model With Dropout...")

model_regularized = create_model(
    learning_rate=0.001,
    dropout_rate=0.5
)

history_regularized = model_regularized.fit(
    x_train,
    y_train,
    epochs=30,
    batch_size=128,
    validation_split=0.1,
    verbose=0
)


# ============================================================
# EXPERIMENT 3 : Hyperparameter Comparison
# ============================================================

print("\n================================================")
print("EXPERIMENT 3 : HYPERPARAMETER COMPARISON")
print("================================================")


configs = {

    "LR=0.001 Drop=0.3":
    {
        "learning_rate":0.001,
        "dropout_rate":0.3
    },

    "LR=0.001 Drop=0.5":
    {
        "learning_rate":0.001,
        "dropout_rate":0.5
    },

    "LR=0.01 Drop=0.3":
    {
        "learning_rate":0.01,
        "dropout_rate":0.3
    }

}


hp_results = {}


for name, config in configs.items():

    print(f"\n{name}")

    model = create_model(
        learning_rate=config["learning_rate"],
        dropout_rate=config["dropout_rate"]
    )

    model.fit(
        x_train,
        y_train,
        epochs=10,
        batch_size=128,
        validation_split=0.1,
        verbose=0
    )


    # Evaluate Model

    loss, accuracy = model.evaluate(
        x_test,
        y_test,
        verbose=0
    )

    print(f"Accuracy : {accuracy:.4f}")

    hp_results[name] = accuracy


# ============================================================
# Plot 1 : Learning Rate Comparison
# ============================================================

plt.figure(figsize=(8,5))

for lr, history in lr_history.items():

    plt.plot(
        history.history['val_accuracy'],
        label=f"LR={lr}"
    )

plt.title("Learning Rate vs Validation Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.grid(True)

plt.show()


# ============================================================
# Plot 2 : Training vs Validation Loss
# ============================================================

plt.figure(figsize=(8,5))

plt.plot(
    history_overfit.history['loss'],
    label='Training Loss'
)

plt.plot(
    history_overfit.history['val_loss'],
    label='Validation Loss'
)

plt.title("Overfitting Analysis")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.grid(True)

plt.show()


# ============================================================
# Plot 3 : Hyperparameter Comparison
# ============================================================

names = list(hp_results.keys())

accuracies = list(hp_results.values())


plt.figure(figsize=(8,5))

plt.bar(names, accuracies)

plt.title("Hyperparameter Accuracy Comparison")

plt.ylabel("Accuracy")

plt.xticks(rotation=10)

plt.grid(True)

plt.show()


# ============================================================
# Final Results
# ============================================================

print("\n================================================")
print("FINAL RESULTS")
print("================================================")


print("\nLearning Rate Results:")

for lr, acc in lr_results.items():

    print(f"LR = {lr}  --> Accuracy = {acc:.4f}")


print("\nHyperparameter Results:")

for name, acc in hp_results.items():

    print(f"{name} --> Accuracy = {acc:.4f}")

