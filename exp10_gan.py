# ============================================================
# EXP 10 : GAN
# Generating Synthetic MNIST Digits
# ============================================================

# ------------------------------------------------------------
# Import Libraries
# ------------------------------------------------------------

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential, Model

from tensorflow.keras.layers import (
    Dense,
    Flatten,
    Reshape,
    LeakyReLU,
    Dropout,
    Input
)

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.datasets import mnist

import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Hyperparameters
# ------------------------------------------------------------

LATENT_DIM = 100

IMG_SIZE = 28 * 28

BATCH_SIZE = 64

EPOCHS = 3000

SAMPLE_INTERVAL = 500


# ------------------------------------------------------------
# Load MNIST Dataset
# ------------------------------------------------------------

print("\nLoading Dataset...")

(x_train, _), (_, _) = mnist.load_data()


# Use Smaller Dataset

x_train = x_train[:10000]


# Normalize Images

x_train = (x_train.astype('float32') - 127.5) / 127.5


# Flatten Images

x_train = x_train.reshape(-1, IMG_SIZE)

print("Training Data Shape :", x_train.shape)


# ============================================================
# Build Generator
# ============================================================

def build_generator():

    model = Sequential([

        Dense(256, input_dim=LATENT_DIM),

        LeakyReLU(0.2),

        Dense(512),

        LeakyReLU(0.2),

        Dense(1024),

        LeakyReLU(0.2),

        Dense(
            IMG_SIZE,
            activation='tanh'
        )

    ])

    return model


# ============================================================
# Build Discriminator
# ============================================================

def build_discriminator():

    model = Sequential([

        Dense(512, input_dim=IMG_SIZE),

        LeakyReLU(0.2),

        Dropout(0.3),

        Dense(256),

        LeakyReLU(0.2),

        Dropout(0.3),

        Dense(
            1,
            activation='sigmoid'
        )

    ])

    return model


# ============================================================
# Create Models
# ============================================================

generator = build_generator()

discriminator = build_discriminator()


# Compile Discriminator

discriminator.compile(
    optimizer=Adam(0.0002, 0.5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)


# Freeze Discriminator while Training GAN

discriminator.trainable = False


# GAN Model

gan_input = Input(shape=(LATENT_DIM,))

generated_image = generator(gan_input)

gan_output = discriminator(generated_image)

gan = Model(gan_input, gan_output)


# Compile GAN

gan.compile(
    optimizer=Adam(0.0002, 0.5),
    loss='binary_crossentropy'
)


# Print Model Summary

print("\n================================================")
print("GENERATOR MODEL")
print("================================================")

generator.summary()


print("\n================================================")
print("DISCRIMINATOR MODEL")
print("================================================")

discriminator.summary()


# ============================================================
# Training Variables
# ============================================================

d_losses = []

g_losses = []

sample_images = {}


# Real and Fake Labels

real = np.ones((BATCH_SIZE,1))

fake = np.zeros((BATCH_SIZE,1))


# ============================================================
# Function to Save Generated Images
# ============================================================

def save_generated_images(step):

    noise = np.random.normal(
        0,
        1,
        (25, LATENT_DIM)
    )


    # Generate Images

    generated = generator.predict(
        noise,
        verbose=0
    )


    # Rescale Images

    generated = 0.5 * generated + 0.5

    generated = generated.reshape(-1,28,28)

    sample_images[step] = generated


# ============================================================
# GAN Training Loop
# ============================================================

print("\nStarting GAN Training...\n")


for epoch in range(EPOCHS + 1):


    # --------------------------------------------------------
    # Train Discriminator
    # --------------------------------------------------------

    idx = np.random.randint(
        0,
        x_train.shape[0],
        BATCH_SIZE
    )

    real_images = x_train[idx]


    # Generate Fake Images

    noise = np.random.normal(
        0,
        1,
        (BATCH_SIZE, LATENT_DIM)
    )

    fake_images = generator.predict(
        noise,
        verbose=0
    )


    # Train on Real Images

    d_loss_real = discriminator.train_on_batch(
        real_images,
        real
    )


    # Train on Fake Images

    d_loss_fake = discriminator.train_on_batch(
        fake_images,
        fake
    )


    # Average Loss

    d_loss = 0.5 * (
        d_loss_real[0] + d_loss_fake[0]
    )


    # --------------------------------------------------------
    # Train Generator
    # --------------------------------------------------------

    noise = np.random.normal(
        0,
        1,
        (BATCH_SIZE, LATENT_DIM)
    )


    g_loss = gan.train_on_batch(
        noise,
        real
    )


    # Save Losses

    d_losses.append(d_loss)

    g_losses.append(g_loss)


    # --------------------------------------------------------
    # Print Progress
    # --------------------------------------------------------

    if epoch % 500 == 0:

        print(
            f"Epoch {epoch}"
            f" | D Loss = {d_loss:.4f}"
            f" | G Loss = {g_loss:.4f}"
        )


    # Save Generated Samples

    if epoch % SAMPLE_INTERVAL == 0:

        save_generated_images(epoch)


# ============================================================
# Plot Generated Images
# ============================================================

steps = list(sample_images.keys())

fig, axes = plt.subplots(
    1,
    len(steps),
    figsize=(15,5)
)


if len(steps) == 1:

    axes = [axes]


for ax, step in zip(axes, steps):

    images = sample_images[step]


    # Create 5x5 Grid

    grid = np.zeros((28*5, 28*5))


    for r in range(5):

        for c in range(5):

            grid[
                r*28:(r+1)*28,
                c*28:(c+1)*28
            ] = images[r*5 + c]


    ax.imshow(grid, cmap='gray')

    ax.set_title(f"Step {step}")

    ax.axis('off')


plt.suptitle("Generated Images During Training")

plt.tight_layout()

plt.show()


# ============================================================
# Plot Loss Curves
# ============================================================

plt.figure(figsize=(10,5))

plt.plot(
    d_losses,
    label='Discriminator Loss'
)

plt.plot(
    g_losses,
    label='Generator Loss'
)

plt.title("Generator vs Discriminator Loss")

plt.xlabel("Training Step")

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


print(
    f"Final Generator Loss     : "
    f"{np.mean(g_losses[-100:]):.4f}"
)

print(
    f"Final Discriminator Loss : "
    f"{np.mean(d_losses[-100:]):.4f}"
)

