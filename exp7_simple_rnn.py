# ============================================================
# EXP 7 : Simple RNN
# Next Character Prediction using Text Data
# ============================================================

# ------------------------------------------------------------
# Import Libraries
# ------------------------------------------------------------

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    Embedding,
    SimpleRNN,
    Dense
)

import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Text Dataset
# ------------------------------------------------------------

TEXT = """
Deep learning is powerful.
Artificial intelligence is the future.
Machine learning helps solve problems.
Neural networks learn patterns from data.
Simple RNN works on sequence data.
""" * 20


TEXT = TEXT.lower()

print("\nText Length :", len(TEXT))


# ============================================================
# Character Vocabulary
# ============================================================

chars = sorted(set(TEXT))

vocab_size = len(chars)

print("Vocabulary Size :", vocab_size)


# Character to Index Mapping

char_to_index = {
    char:index
    for index, char in enumerate(chars)
}

index_to_char = {
    index:char
    for char, index in char_to_index.items()
}


# ============================================================
# Create Sequences
# ============================================================

def create_sequences(text, seq_length):

    X = []

    y = []


    # Convert Characters into Numbers

    encoded = [
        char_to_index[c]
        for c in text
    ]


    # Create Input and Output Sequences

    for i in range(len(encoded) - seq_length):

        X.append(
            encoded[i:i+seq_length]
        )

        y.append(
            encoded[i+seq_length]
        )


    X = np.array(X)

    y = tf.keras.utils.to_categorical(
        y,
        num_classes=vocab_size
    )

    return X, y


# ============================================================
# Build Simple RNN Model
# ============================================================

def build_rnn(seq_length):

    model = Sequential([

        Embedding(
            vocab_size,
            16,
            input_length=seq_length
        ),

        SimpleRNN(64),

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
# Generate Text Function
# ============================================================

def generate_text(model, seed_text, seq_length, num_chars=100):

    result = seed_text


    # Convert Seed Text into Numbers

    input_seq = [
        char_to_index[c]
        for c in seed_text[-seq_length:]
    ]


    for _ in range(num_chars):

        x = np.array([input_seq])

        prediction = model.predict(
            x,
            verbose=0
        )[0]


        # Predicted Character

        next_index = np.argmax(prediction)

        next_char = index_to_char[next_index]


        # Add Character

        result += next_char


        # Update Sequence

        input_seq = input_seq[1:] + [next_index]


    return result


# ============================================================
# EXPERIMENT 1 : Effect of Sequence Length
# ============================================================

print("\n================================================")
print("EXPERIMENT 1 : SEQUENCE LENGTH ANALYSIS")
print("================================================")


sequence_lengths = [5, 15, 30]

results = {}


for seq_length in sequence_lengths:

    print(f"\nTraining with Sequence Length = {seq_length}")


    # Create Sequences

    X, y = create_sequences(
        TEXT,
        seq_length
    )


    # Build Model

    model = build_rnn(seq_length)


    # Train Model

    history = model.fit(
        X,
        y,
        epochs=20,
        batch_size=64,
        validation_split=0.1,
        verbose=0
    )


    # Final Accuracy

    final_accuracy = history.history[
        'val_accuracy'
    ][-1]


    # Store Results

    results[seq_length] = {
        'history':history,
        'accuracy':final_accuracy,
        'model':model
    }


    print(f"Validation Accuracy : {final_accuracy:.4f}")


    # Generate Sample Text

    seed = TEXT[:seq_length]

    generated = generate_text(
        model,
        seed,
        seq_length
    )

    print("\nGenerated Text:")

    print(generated[:100])


# ============================================================
# EXPERIMENT 2 : Short vs Long Sequences
# ============================================================

print("\n================================================")
print("EXPERIMENT 2 : SHORT vs LONG SEQUENCES")
print("================================================")


# Short Sequence

X_short, y_short = create_sequences(
    TEXT,
    5
)

# Long Sequence

X_long, y_long = create_sequences(
    TEXT,
    40
)


# Build Models

model_short = build_rnn(5)

model_long = build_rnn(40)


# Train Models

print("\nTraining Short Sequence Model...")

history_short = model_short.fit(
    X_short,
    y_short,
    epochs=20,
    batch_size=64,
    validation_split=0.1,
    verbose=0
)


print("\nTraining Long Sequence Model...")

history_long = model_long.fit(
    X_long,
    y_long,
    epochs=20,
    batch_size=64,
    validation_split=0.1,
    verbose=0
)


# ============================================================
# Plot Accuracy Graph
# ============================================================

plt.figure(figsize=(10,5))

for seq_length, result in results.items():

    plt.plot(
        result['history'].history['val_accuracy'],
        label=f"Seq Length = {seq_length}"
    )

plt.title("Sequence Length vs Validation Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.grid(True)

plt.show()


# ============================================================
# Plot Loss Graph
# ============================================================

plt.figure(figsize=(10,5))

plt.plot(
    history_short.history['val_loss'],
    label='Short Sequence'
)

plt.plot(
    history_long.history['val_loss'],
    label='Long Sequence'
)

plt.title("Short vs Long Sequence Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.grid(True)

plt.show()


# ============================================================
# Final Results
# ============================================================

print("\n================================================")
print("FINAL RESULTS")
print("================================================")


for seq_length, result in results.items():

    print(
        f"Sequence Length = {seq_length}"
        f" --> Accuracy = {result['accuracy']:.4f}"
    )

