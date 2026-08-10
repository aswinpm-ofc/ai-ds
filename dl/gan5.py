# ================================================================
# Deep Convolutional GAN (DCGAN) — MNIST Handwritten Digit Generation
# ================================================================
# Single, consolidated implementation: one Generator, one
# Discriminator, one training run. Tracks losses, checkpoints
# periodically, and saves sample images + final trained models.

import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.initializers import RandomNormal

# ----------------------------------------------------------------
# GPU info
# ----------------------------------------------------------------
print("=" * 60)
print("TensorFlow version:", tf.__version__)
print("=" * 60)
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    print("GPU detected:")
    for gpu in gpus:
        print(" ", gpu)
else:
    print("No GPU detected -- running on CPU (training will be slower).")

# ----------------------------------------------------------------
# Hyperparameters
# ----------------------------------------------------------------
BUFFER_SIZE = 60000
BATCH_SIZE = 128
EPOCHS = 100
NOISE_DIM = 100
LEARNING_RATE = 2e-4
BETA_1 = 0.5
NUM_EXAMPLES_TO_GENERATE = 16
CHECKPOINT_INTERVAL = 10       # every N epochs: save checkpoint + image grid + latest model
KEEP_N_CHECKPOINTS = 3         # older checkpoints beyond this are auto-pruned

# ----------------------------------------------------------------
# Output folders
# ----------------------------------------------------------------
IMAGE_DIR = "generated_images"
MODEL_DIR = "saved_model"
CHECKPOINT_DIR = "training_checkpoints"

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ----------------------------------------------------------------
# Load & preprocess MNIST
# ----------------------------------------------------------------
(x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype("float32")
x_train = (x_train - 127.5) / 127.5           # scale to [-1, 1] to match generator's tanh output
x_train = np.expand_dims(x_train, axis=-1)    # (N, 28, 28) -> (N, 28, 28, 1)

print("\nDataset shape:", x_train.shape)

train_dataset = (
    tf.data.Dataset.from_tensor_slices(x_train)
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(tf.data.AUTOTUNE)
)

# ----------------------------------------------------------------
# Weight initializer (DCGAN paper: N(0, 0.02)) -- applied to every
# conv/dense layer in both networks, consistently.
# ----------------------------------------------------------------
init = RandomNormal(mean=0.0, stddev=0.02)

# ----------------------------------------------------------------
# Generator: 100-d noise -> 28x28x1 image
# Dense -> reshape to 7x7x512, then upsample 7->14->28 with
# Conv2DTranspose, "same" padding throughout.
# ----------------------------------------------------------------
def build_generator():
    model = tf.keras.Sequential(name="Generator")

    model.add(layers.Input(shape=(NOISE_DIM,)))
    model.add(layers.Dense(7 * 7 * 512, use_bias=False, kernel_initializer=init))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Reshape((7, 7, 512)))

    model.add(layers.Conv2DTranspose(256, kernel_size=5, strides=1, padding="same",
                                      use_bias=False, kernel_initializer=init))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))

    model.add(layers.Conv2DTranspose(128, kernel_size=5, strides=2, padding="same",
                                      use_bias=False, kernel_initializer=init))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))

    model.add(layers.Conv2DTranspose(64, kernel_size=5, strides=2, padding="same",
                                      use_bias=False, kernel_initializer=init))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))

    model.add(layers.Conv2D(1, kernel_size=5, padding="same",
                             activation="tanh", kernel_initializer=init))
    return model

# ----------------------------------------------------------------
# Discriminator: 28x28x1 image -> real/fake logit
# Channels (192/384/768) are widened 3x over a "minimal" DCGAN
# discriminator so its capacity (~4.5M params) is in the same
# ballpark as the generator (~6.9M) rather than a small fraction
# of it -- helps keep the adversarial game balanced.
# GaussianNoise on the input (instance noise) + dropout help keep
# the discriminator from overpowering the generator early on.
# ----------------------------------------------------------------
def build_discriminator():
    model = tf.keras.Sequential(name="Discriminator")

    model.add(layers.Input(shape=(28, 28, 1)))
    model.add(layers.GaussianNoise(0.05))

    model.add(layers.Conv2D(192, kernel_size=5, strides=2, padding="same",
                             kernel_initializer=init))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(384, kernel_size=5, strides=2, padding="same",
                             kernel_initializer=init))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(768, kernel_size=3, strides=2, padding="same",
                             kernel_initializer=init))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1, kernel_initializer=init))
    return model

generator = build_generator()
discriminator = build_discriminator()

generator.summary()
print()
discriminator.summary()

print(f"\nGenerator params:     {generator.count_params():,}")
print(f"Discriminator params: {discriminator.count_params():,}")
print(f"Combined total:       {generator.count_params() + discriminator.count_params():,}")

# ----------------------------------------------------------------
# Losses (one-sided label smoothing on real labels -- a standard
# DCGAN stability trick) and optimizers
# ----------------------------------------------------------------
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_labels = tf.ones_like(real_output) * 0.9
    fake_labels = tf.zeros_like(fake_output)
    real_loss = cross_entropy(real_labels, real_output)
    fake_loss = cross_entropy(fake_labels, fake_output)
    return real_loss + fake_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=BETA_1)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=BETA_1)

# ----------------------------------------------------------------
# Checkpointing (CheckpointManager auto-prunes old checkpoints;
# epoch is stored INSIDE the checkpoint so a resume is exact
# regardless of CHECKPOINT_INTERVAL)
# ----------------------------------------------------------------
epoch_var = tf.Variable(0, dtype=tf.int64, trainable=False)

checkpoint = tf.train.Checkpoint(
    epoch=epoch_var,
    generator_optimizer=generator_optimizer,
    discriminator_optimizer=discriminator_optimizer,
    generator=generator,
    discriminator=discriminator,
)
checkpoint_manager = tf.train.CheckpointManager(
    checkpoint, CHECKPOINT_DIR, max_to_keep=KEEP_N_CHECKPOINTS
)

# ----------------------------------------------------------------
# Fixed seed: reused every save so you can watch the SAME 16
# latent vectors evolve across training, instead of a fresh
# random draw each time.
# ----------------------------------------------------------------
seed = tf.random.normal([NUM_EXAMPLES_TO_GENERATE, NOISE_DIM])

generator_losses = []
discriminator_losses = []
d_real_accuracy = []   # mean D(real) probability per epoch
d_fake_accuracy = []   # mean D(fake) probability per epoch

# ----------------------------------------------------------------
# Loss-history persistence -- saved alongside each checkpoint so a
# resume restores the full loss curves, not just the model weights.
# ----------------------------------------------------------------
HISTORY_PATH = os.path.join(CHECKPOINT_DIR, "history.json")

def save_history():
    with open(HISTORY_PATH, "w") as f:
        json.dump({
            "generator_losses": generator_losses,
            "discriminator_losses": discriminator_losses,
            "d_real_accuracy": d_real_accuracy,
            "d_fake_accuracy": d_fake_accuracy,
        }, f)

def load_history():
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH) as f:
            data = json.load(f)
        generator_losses.extend(data["generator_losses"])
        discriminator_losses.extend(data["discriminator_losses"])
        d_real_accuracy.extend(data["d_real_accuracy"])
        d_fake_accuracy.extend(data["d_fake_accuracy"])

# ----------------------------------------------------------------
# Resume from the latest checkpoint if one exists
# ----------------------------------------------------------------
start_epoch = 0
if checkpoint_manager.latest_checkpoint:
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    start_epoch = int(epoch_var.numpy())
    load_history()
    print(f"\nResumed from checkpoint at epoch {start_epoch} "
          f"({len(generator_losses)} epochs of history loaded)")
else:
    print("\nNo checkpoint found -- starting from scratch.")

# ----------------------------------------------------------------
# Save a 4x4 grid of generated images
# ----------------------------------------------------------------
def save_image_grid(model, epoch, test_input, tag="seed"):
    predictions = model(test_input, training=False)
    predictions = (predictions + 1.0) / 2.0   # [-1,1] -> [0,1] for display

    fig = plt.figure(figsize=(6, 6))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0], cmap="gray")
        plt.axis("off")
    plt.suptitle(f"Epoch {epoch}")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, f"{tag}_epoch_{epoch:03d}.png"), dpi=150)
    plt.close(fig)

# ----------------------------------------------------------------
# One training step
# ----------------------------------------------------------------
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))

    d_real_prob = tf.reduce_mean(tf.sigmoid(real_output))
    d_fake_prob = tf.reduce_mean(tf.sigmoid(fake_output))

    return gen_loss, disc_loss, d_real_prob, d_fake_prob

# ----------------------------------------------------------------
# Full training loop (resumes from start_epoch if set)
# ----------------------------------------------------------------
def train(dataset, epochs, start_epoch=0):
    for epoch in range(start_epoch, epochs):
        start = time.time()

        g_loss_sum = d_loss_sum = d_real_sum = d_fake_sum = 0.0
        n_batches = 0

        for image_batch in dataset:
            g_loss, d_loss, d_real, d_fake = train_step(image_batch)
            g_loss_sum += g_loss.numpy()
            d_loss_sum += d_loss.numpy()
            d_real_sum += d_real.numpy()
            d_fake_sum += d_fake.numpy()
            n_batches += 1

        g_loss_avg = g_loss_sum / n_batches
        d_loss_avg = d_loss_sum / n_batches
        d_real_avg = d_real_sum / n_batches
        d_fake_avg = d_fake_sum / n_batches

        generator_losses.append(float(g_loss_avg))
        discriminator_losses.append(float(d_loss_avg))
        d_real_accuracy.append(float(d_real_avg))
        d_fake_accuracy.append(float(d_fake_avg))

        print(
            f"Epoch {epoch + 1:3d}/{epochs} | "
            f"G loss: {g_loss_avg:.4f} | D loss: {d_loss_avg:.4f} | "
            f"D(real): {d_real_avg:.3f} | D(fake): {d_fake_avg:.3f} | "
            f"{time.time() - start:.1f}s"
        )

        if (epoch + 1) % CHECKPOINT_INTERVAL == 0:
            epoch_var.assign(epoch + 1)
            checkpoint_manager.save()
            save_image_grid(generator, epoch + 1, seed, tag="checkpoint")
            generator.save(os.path.join(MODEL_DIR, "generator_latest.keras"))
            discriminator.save(os.path.join(MODEL_DIR, "discriminator_latest.keras"))
            save_history()
            print(f"  -> checkpoint + image + latest model saved (epoch {epoch + 1})")

# ----------------------------------------------------------------
# Run training
# ----------------------------------------------------------------
start_time = time.time()
train(train_dataset, EPOCHS, start_epoch=start_epoch)
print(f"\nTotal training time this run: {(time.time() - start_time) / 60:.1f} minutes")

# ----------------------------------------------------------------
# Final outputs
# ----------------------------------------------------------------

# 1. Progress grid from the tracked seed (same 16 vectors used throughout training)
save_image_grid(generator, EPOCHS, seed, tag="final_seed")

# 2. A fresh grid of 16 brand-new random samples, confirming the generator
#    produces varied digits beyond just the tracked seed vectors
fresh_noise = tf.random.normal([NUM_EXAMPLES_TO_GENERATE, NOISE_DIM])
save_image_grid(generator, EPOCHS, fresh_noise, tag="final_fresh")

print(f"\nSample grids saved to '{IMAGE_DIR}/'")

# Save trained models
generator.save(os.path.join(MODEL_DIR, "generator.keras"))
discriminator.save(os.path.join(MODEL_DIR, "discriminator.keras"))
print(f"Models saved to '{MODEL_DIR}/'")

# Generator loss plot
plt.figure(figsize=(10, 5))
plt.plot(generator_losses, linewidth=2, color="tab:blue", label="Generator Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Generator Loss During Training")
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig(os.path.join(IMAGE_DIR, "generator_loss.png"), dpi=150)
plt.show()

# Discriminator loss plot
plt.figure(figsize=(10, 5))
plt.plot(discriminator_losses, linewidth=2, color="tab:red", label="Discriminator Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Discriminator Loss During Training")
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig(os.path.join(IMAGE_DIR, "discriminator_loss.png"), dpi=150)
plt.show()

# Summary
print("=" * 50)
print("Training complete")
print("=" * 50)
print(f"Final Generator Loss     : {generator_losses[-1]:.4f}")
print(f"Final Discriminator Loss : {discriminator_losses[-1]:.4f}")
print(f"Final D(real) / D(fake)  : {d_real_accuracy[-1]:.3f} / {d_fake_accuracy[-1]:.3f}")
print("=" * 50)