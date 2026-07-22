
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, roc_auc_score
)

RANDOM_STATE = 42
tf.random.set_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

DATA_DIR = "data"          # folder containing "yes" and "no" subfolders
IMG_SIZE = (150, 150)
BATCH_SIZE = 32
CLASS_NAMES = ["no", "yes"]  # alphabetical order matches Keras' default

# --------------------------------------------------------------------------
# Task 1: Data Understanding
# --------------------------------------------------------------------------

if not os.path.isdir(DATA_DIR):
    raise FileNotFoundError(
        f"'{DATA_DIR}' not found. Download the dataset from Kaggle "
        f"(navoneel/brain-mri-images-for-brain-tumor-detection), unzip it, "
        f"and point DATA_DIR at the folder containing 'yes' and 'no'."
    )

n_yes = len(os.listdir(os.path.join(DATA_DIR, "yes")))
n_no = len(os.listdir(os.path.join(DATA_DIR, "no")))
print(f"Tumor ('yes') images   : {n_yes}")
print(f"No tumor ('no') images : {n_no}")
print(f"Total images           : {n_yes + n_no}")
print(f"Classes                : {CLASS_NAMES}")

# 80% train / 20% validation split straight from the directory structure
# (Keras handles the shuffling and splitting for us here)
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    labels="inferred",
    label_mode="binary",
    class_names=CLASS_NAMES,
    color_mode="grayscale",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="training",
    seed=RANDOM_STATE,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    labels="inferred",
    label_mode="binary",
    class_names=CLASS_NAMES,
    color_mode="grayscale",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="validation",
    seed=RANDOM_STATE,
)

# The validation split above is used as our held-out test set for
# evaluation in Task 4.
test_ds = val_ds

# Display a handful of sample MRI images with their labels
plt.figure(figsize=(12, 4))
for images, labels in train_ds.take(1):
    for i in range(min(5, images.shape[0])):
        ax = plt.subplot(1, 5, i + 1)
        plt.imshow(images[i].numpy().astype("uint8").squeeze(), cmap="gray")
        plt.title(CLASS_NAMES[int(labels[i].numpy()[0])])
        plt.axis("off")
plt.tight_layout()
plt.savefig("sample_mri_images.png", dpi=150)
print("\nSaved sample MRI images to sample_mri_images.png")

# --------------------------------------------------------------------------
# Task 2: Data Preprocessing
# --------------------------------------------------------------------------

# Normalize pixel values from [0, 255] to [0, 1]
normalization_layer = layers.Rescaling(1.0 / 255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# Light augmentation on the training set only, since medical imaging
# datasets are often small -- this helps reduce overfitting
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.05),
])
train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))

# Cache/prefetch for training efficiency
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# --------------------------------------------------------------------------
# Task 3: Model Development (CNN)
# --------------------------------------------------------------------------

model = models.Sequential([
    layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 1)),

    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(1, activation="sigmoid"),   # binary: tumor probability
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.Precision(name="precision"),
             tf.keras.metrics.Recall(name="recall")],
)

model.summary()

EPOCHS = 25

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
)

# --------------------------------------------------------------------------
# Task 4: Model Evaluation
# --------------------------------------------------------------------------

test_results = model.evaluate(test_ds, verbose=0)
print("\n=== Test set results ===")
for name, value in zip(model.metrics_names, test_results):
    print(f"{name}: {value:.4f}")

y_true, y_pred, y_prob = [], [], []
for images, labels in test_ds:
    probs = model.predict(images, verbose=0).flatten()
    y_prob.extend(probs)
    y_pred.extend((probs >= 0.5).astype(int))
    y_true.extend(labels.numpy().flatten().astype(int))

print("\n=== Classification report ===")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

auc = roc_auc_score(y_true, y_prob)
print(f"ROC AUC: {auc:.4f}")

# Training curves
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(history.history["accuracy"], label="Train Accuracy")
axes[0].plot(history.history["val_accuracy"], label="Validation Accuracy")
axes[0].set_title("Accuracy over Epochs")
axes[0].set_xlabel("Epoch")
axes[0].legend()

axes[1].plot(history.history["loss"], label="Train Loss")
axes[1].plot(history.history["val_loss"], label="Validation Loss")
axes[1].set_title("Loss over Epochs")
axes[1].set_xlabel("Epoch")
axes[1].legend()
plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, cmap="Blues", colorbar=False)
plt.title("Confusion Matrix -- MRI Tumor Detection")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)

# ROC curve
fpr, tpr, _ = roc_curve(y_true, y_prob)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve -- MRI Tumor Detection")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=150)

print("\nSaved training_curves.png, confusion_matrix.png, roc_curve.png")

with open("metrics.txt", "w") as f:
    for name, value in zip(model.metrics_names, test_results):
        f.write(f"{name},{value:.4f}\n")
    f.write(f"roc_auc,{auc:.4f}\n")

print("\nDone.")
