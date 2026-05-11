# ============================================================
# EXP 9 : GRU
# GRU vs LSTM vs Simple RNN
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
    Embedding,
    SimpleRNN,
    LSTM,
    GRU,
    Dense
)

import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Text Dataset
# ------------------------------------------------------------

TEXT = """
Deep learning uses neural networks.
RNN works on sequential data.
LSTM handles long-term dependencies.
GRU is faster than LSTM.
Machine learning learns patterns from data.
Real-time systems require fast prediction.
""" * 20


TEXT = TEXT.lower()

print("\nText Length :", len(TEXT))


# ============================================================
# Character Vocabulary
# ============================================================

chars = sorted(set(TEXT))

vocab_size = len(chars)

print("Vocabulary Size :", vocab_size)


# Character Mapping

char_to_index = {
    char:index
    for index, char in enumerate(chars)
}

index_to_char = {
    index:char
    for char, index in char_to_index.items()
}


# ============================================================
# Hyperparameters
# ============================================================

SEQ_LENGTH = 25

EPOCHS = 10

UNITS = 64

BATCH_SIZE = 128


# ============================================================
# Create Sequences
# ============================================================

X = []

y = []


# Convert Characters into Numbers

encoded = [
    char_to_index[c]
    for c in TEXT
]


# Create Input and Output Sequences

for i in range(len(encoded) - SEQ_LENGTH):

    X.append(
        encoded[i:i+SEQ_LENGTH]
    )

    y.append(
        encoded[i+SEQ_LENGTH]
    )


X = np.array(X)

y = tf.keras.utils.to_categorical(
    y,
    num_classes=vocab_size
)


print("Training Samples :", X.shape)


# ============================================================
# Build Simple RNN Model
# ============================================================

def build_rnn():

    model = Sequential([

        Embedding(
            vocab_size,
            16,
            input_length=SEQ_LENGTH
        ),

        SimpleRNN(UNITS),

        Dense(
            vocab_size,
            activation='softmax'
        )

    ])


    # Compile Model

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# ============================================================
# Build LSTM Model
# ============================================================

def build_lstm():

    model = Sequential([

        Embedding(
            vocab_size,
            16,
            input_length=SEQ_LENGTH
        ),

        LSTM(UNITS),

        Dense(
            vocab_size,
            activation='softmax'
        )

    ])


    # Compile Model

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# ============================================================
# Build GRU Model
# ============================================================

def build_gru():

    model = Sequential([

        Embedding(
            vocab_size,
            16,
            input_length=SEQ_LENGTH
        ),

        GRU(UNITS),

        Dense(
            vocab_size,
            activation='softmax'
        )

    ])


    # Compile Model

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# ============================================================
# Train Models
# ============================================================

models = {

    "Simple RNN": build_rnn(),

    "LSTM": build_lstm(),

    "GRU": build_gru()

}


results = {}


for name, model in models.items():

    print("\n================================================")

    print(f"TRAINING {name}")

    print("================================================")


    # Record Training Time

    start_time = time.time()


    # Train Model

    history = model.fit(
        X,
        y,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        verbose=0
    )


    # Training Time

    training_time = time.time() - start_time


    # Evaluate Model

    loss, accuracy = model.evaluate(
        X,
        y,
        verbose=0
    )


    # Save Results

    results[name] = {

        "history": history,

        "accuracy": accuracy,

        "time": training_time,

        "params": model.count_params()
    }


    print(f"Accuracy : {accuracy:.4f}")

    print(f"Time     : {training_time:.2f} seconds")

    print(f"Parameters: {model.count_params()}")


# ============================================================
# Plot Validation Accuracy
# ============================================================

plt.figure(figsize=(10,5))

for name, result in results.items():

    plt.plot(
        result["history"].history['val_accuracy'],
        label=name
    )

plt.title("Validation Accuracy Comparison")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.grid(True)

plt.show()


# ============================================================
# Plot Training Time
# ============================================================

model_names = list(results.keys())

times = [
    results[m]["time"]
    for m in model_names
]


plt.figure(figsize=(8,5))

plt.bar(model_names, times)

plt.title("Training Time Comparison")

plt.ylabel("Time (seconds)")

plt.grid(True)

plt.show()


# ============================================================
# Plot Parameter Count
# ============================================================

params = [
    results[m]["params"]
    for m in model_names
]


plt.figure(figsize=(8,5))

plt.bar(model_names, params)

plt.title("Parameter Comparison")

plt.ylabel("Number of Parameters")

plt.grid(True)

plt.show()


# ============================================================
# Final Results
# ============================================================

print("\n================================================")
print("FINAL RESULTS")
print("================================================")


for name, result in results.items():

    print(f"\n{name}")

    print(f"Accuracy  : {result['accuracy']:.4f}")

    print(f"Time      : {result['time']:.2f} sec")

    print(f"Parameters: {result['params']}")
