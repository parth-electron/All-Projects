

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

RANDOM_STATE = 42
tf.random.set_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

# --------------------------------------------------------------------------
# Task 1: Data Understanding
# --------------------------------------------------------------------------

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

print("Training data shape :", X_train.shape)  # (50000, 32, 32, 3)
print("Training labels shape:", y_train.shape)  # (50000, 1)
print("Test data shape     :", X_test.shape)    # (10000, 32, 32, 3)
print("Test labels shape   :", y_test.shape)     # (10000, 1)
print("Number of classes   :", len(CLASS_NAMES))
print("Class names         :", CLASS_NAMES)

# Show the first 5 training images with their labels
fig, axes = plt.subplots(1, 5, figsize=(12, 3))
for i, ax in enumerate(axes):
    ax.imshow(X_train[i])
    ax.set_title(CLASS_NAMES[int(y_train[i])])
    ax.axis("off")
plt.tight_layout()
plt.savefig("sample_images.png", dpi=150)
print("\nSaved first-five sample images to sample_images.png")

# --------------------------------------------------------------------------
# Task 2: Data Preprocessing
# --------------------------------------------------------------------------

# Check for missing/corrupt values (CIFAR-10 is a clean, complete dataset,
# but we verify anyway as good practice)
print("\nAny NaNs in training images?", np.isnan(X_train).any())
print("Any NaNs in test images?    ", np.isnan(X_test).any())

# Normalize pixel values from [0, 255] to [0, 1]
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# One-hot encode the labels (10 classes)
y_train_cat = to_categorical(y_train, num_classes=10)
y_test_cat = to_categorical(y_test, num_classes=10)

# Keras already provides a fixed 50,000 / 10,000 (train/test) split for
# CIFAR-10, which serves the 80/20-style train/test purpose of this task.
print(f"\nTraining set size: {X_train.shape[0]} images")
print(f"Testing set size : {X_test.shape[0]} images")

# --------------------------------------------------------------------------
# Task 3: Model Development (CNN)
# --------------------------------------------------------------------------

model = models.Sequential([
    layers.Input(shape=(32, 32, 3)),

    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),

    layers.Flatten(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(10, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

EPOCHS = 20
BATCH_SIZE = 64

history = model.fit(
    X_train, y_train_cat,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=2,
)

# --------------------------------------------------------------------------
# Task 4: Model Evaluation
# --------------------------------------------------------------------------

test_loss, test_accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"\nTest Loss    : {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = y_test.flatten()

print("\n=== Classification report ===")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

# Training curves: accuracy and loss vs. epoch
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(history.history["accuracy"], label="Train Accuracy")
axes[0].plot(history.history["val_accuracy"], label="Validation Accuracy")
axes[0].set_title("Accuracy over Epochs")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()

axes[1].plot(history.history["loss"], label="Train Loss")
axes[1].plot(history.history["val_loss"], label="Validation Loss")
axes[1].set_title("Loss over Epochs")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()

plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
print("\nSaved training curves to training_curves.png")

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
fig, ax = plt.subplots(figsize=(9, 8))
disp.plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation=45)
plt.title("Confusion Matrix -- CIFAR-10 CNN")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
print("Saved confusion matrix to confusion_matrix.png")

with open("metrics.txt", "w") as f:
    f.write(f"test_loss,{test_loss:.4f}\n")
    f.write(f"test_accuracy,{test_accuracy:.4f}\n")

print("\nDone.")
