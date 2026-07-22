

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

RANDOM_STATE = 42
tf.random.set_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# --------------------------------------------------------------------------
# Task 1: Data Understanding
# --------------------------------------------------------------------------

# min_faces_per_person keeps only identities with enough images to learn
# from, which keeps the problem tractable as a classroom-scale assignment.
# resize < 1.0 shrinks the images to keep the CNN small and fast to train.
lfw = fetch_lfw_people(min_faces_per_person=70, resize=0.5, color=False)

X = lfw.images                      # shape: (n_samples, h, w)
y = lfw.target                      # integer-encoded identity labels
target_names = lfw.target_names     # actual person names per class
n_classes = len(target_names)
img_h, img_w = X.shape[1], X.shape[2]

print("Number of samples :", X.shape[0])
print("Image size         :", (img_h, img_w))
print("Number of identities (classes):", n_classes)
print("Identity names     :", list(target_names))

# Show a few sample faces with their identity label
fig, axes = plt.subplots(1, 5, figsize=(12, 3))
for i, ax in enumerate(axes):
    ax.imshow(X[i], cmap="gray")
    ax.set_title(target_names[y[i]], fontsize=9)
    ax.axis("off")
plt.tight_layout()
plt.savefig("sample_faces.png", dpi=150)
print("\nSaved sample face images to sample_faces.png")

# --------------------------------------------------------------------------
# Task 2: Data Preprocessing
# --------------------------------------------------------------------------

print("\nAny NaNs in image data?", np.isnan(X).any())

# Normalize pixel values from [0, 255] to [0, 1]
X = X.astype("float32") / 255.0

# Add the channel dimension the CNN expects: (n_samples, h, w, 1)
X = X[..., np.newaxis]

# One-hot encode the identity labels
y_cat = to_categorical(y, num_classes=n_classes)

# 80% training / 20% testing split, stratified so every identity is
# represented proportionally in both sets
X_train, X_test, y_train, y_test, y_train_int, y_test_int = train_test_split(
    X, y_cat, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

print(f"\nTraining set size: {X_train.shape[0]} images")
print(f"Testing set size : {X_test.shape[0]} images")

# --------------------------------------------------------------------------
# Task 3: Model Development (CNN)
# --------------------------------------------------------------------------

model = models.Sequential([
    layers.Input(shape=(img_h, img_w, 1)),

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
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(n_classes, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

EPOCHS = 30
BATCH_SIZE = 32

history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=2,
)

# --------------------------------------------------------------------------
# Task 4: Model Evaluation
# --------------------------------------------------------------------------

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Loss    : {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

print("\n=== Classification report ===")
print(classification_report(y_test_int, y_pred, target_names=target_names))

# Training curves
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
cm = confusion_matrix(y_test_int, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
fig, ax = plt.subplots(figsize=(9, 8))
disp.plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation=45)
plt.title("Confusion Matrix -- LFW Face Recognition CNN")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
print("Saved confusion matrix to confusion_matrix.png")

with open("metrics.txt", "w") as f:
    f.write(f"test_loss,{test_loss:.4f}\n")
    f.write(f"test_accuracy,{test_accuracy:.4f}\n")

print("\nDone.")
