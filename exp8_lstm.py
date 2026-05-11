# ============================================================
# EXP 8 : LSTM
# Text Prediction using LSTM and Simple RNN
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
    LSTM,
    Dense,
    Dropout
)

import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Text Dataset
# ------------------------------------------------------------

TEXT = """
Deep learning is powerful.
Artificial intelligence is growing rapidly.
Machine learning helps solve problems.
Neural networks learn from data.
LSTM handles long-term dependencies.
Simple RNN struggles with long sequences.
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
# Sequence Parameters
# ============================================================

SEQ_LENGTH = 25

LONG_SEQ = 50

EPOCHS = 20


# ============================================================
# Create Sequences Function
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
# Create Short and Long Sequences
# ============================================================

X_short, y_short = create_sequences(
    TEXT,
    SEQ_LENGTH
)

X_long, y_long = create_sequences(
    TEXT,
    LONG_SEQ
)


# ============================================================
# Build Simple RNN Model
# ============================================================

def build_rnn(seq_length):

    model = Sequential([

        Embedding(
            vocab_size,
            32,
            input_length=seq_length
        ),

        SimpleRNN(128),

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

def build_lstm(seq_length):

    model = Sequential([

        Embedding(
            vocab_size,
            32,
            input_length=seq_length
        ),

        LSTM(
            128,
            return_sequences=True
        ),

        Dropout(0.2),

        LSTM(128),

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
# Train Simple RNN
# ============================================================

print("\n================================================")
print("TRAINING SIMPLE RNN")
print("================================================")


rnn_model = build_rnn(SEQ_LENGTH)

history_rnn = rnn_model.fit(
    X_short,
    y_short,
    epochs=EPOCHS,
    batch_size=128,
    validation_split=0.1,
    verbose=0
)


# ============================================================
# Train LSTM (Short Sequence)
# ============================================================

print("\n================================================")
print("TRAINING LSTM (SHORT SEQUENCE)")
print("================================================")


lstm_model = build_lstm(SEQ_LENGTH)

history_lstm = lstm_model.fit(
    X_short,
    y_short,
    epochs=EPOCHS,
    batch_size=128,
    validation_split=0.1,
    verbose=0
)


# ============================================================
# Train LSTM (Long Sequence)
# ============================================================

print("\n================================================")
print("TRAINING LSTM (LONG SEQUENCE)")
print("================================================")


lstm_long_model = build_lstm(LONG_SEQ)

history_lstm_long = lstm_long_model.fit(
    X_long,
    y_long,
    epochs=EPOCHS,
    batch_size=128,
    validation_split=0.1,
    verbose=0
)


# ============================================================
# Text Generation Function
# ============================================================

def generate_text(model, seed_text, seq_length, num_chars=100):

    result = seed_text


    # Convert Seed into Numbers

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
# Generate Sample Text
# ============================================================

seed = TEXT[:SEQ_LENGTH]

print("\n================================================")
print("TEXT GENERATION")
print("================================================")


print("\nSimple RNN Output:\n")

print(
    generate_text(
        rnn_model,
        seed,
        SEQ_LENGTH
    )[:150]
)


print("\nLSTM Output:\n")

print(
    generate_text(
        lstm_model,
        seed,
        SEQ_LENGTH
    )[:150]
)


# ============================================================
# Evaluate Models
# ============================================================

loss_rnn, acc_rnn = rnn_model.evaluate(
    X_short,
    y_short,
    verbose=0
)

loss_lstm, acc_lstm = lstm_model.evaluate(
    X_short,
    y_short,
    verbose=0
)

loss_long, acc_long = lstm_long_model.evaluate(
    X_long,
    y_long,
    verbose=0
)


# ============================================================
# Plot Loss Comparison
# ============================================================

plt.figure(figsize=(10,5))

plt.plot(
    history_rnn.history['val_loss'],
    label='Simple RNN'
)

plt.plot(
    history_lstm.history['val_loss'],
    label='LSTM'
)

plt.title("Validation Loss Comparison")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.grid(True)

plt.show()


# ============================================================
# Plot Accuracy Comparison
# ============================================================

plt.figure(figsize=(10,5))

models = [
    'Simple RNN',
    'LSTM Short',
    'LSTM Long'
]

accuracies = [
    acc_rnn,
    acc_lstm,
    acc_long
]

plt.bar(models, accuracies)

plt.title("Model Accuracy Comparison")

plt.ylabel("Accuracy")

plt.grid(True)

plt.show()


# ============================================================
# Final Results
# ============================================================

print("\n================================================")
print("FINAL RESULTS")
print("================================================")


print(f"Simple RNN Accuracy   : {acc_rnn:.4f}")

print(f"LSTM Short Accuracy   : {acc_lstm:.4f}")

print(f"LSTM Long Accuracy    : {acc_long:.4f}")
