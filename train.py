import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

# ==========================================
# SETTINGS
# ==========================================

DATASET_DIR = "Dataset"
IMG_SIZE = 160
BATCH_SIZE = 32
EPOCHS = 10

MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "face_mask_detector.keras")

os.makedirs(MODEL_DIR, exist_ok=True)

# ==========================================
# LOAD DATASET
# ==========================================

print("======================================")
print("Loading dataset...")
print("======================================")

train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_dataset.class_names

print()
print("Classes:", class_names)
print()

# ==========================================
# DATA PERFORMANCE
# ==========================================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(AUTOTUNE)
validation_dataset = validation_dataset.prefetch(AUTOTUNE)

# ==========================================
# DATA AUGMENTATION
# ==========================================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.10),
    layers.RandomContrast(0.10)
])

# ==========================================
# MOBILE NET V2
# ==========================================

print("Loading MobileNetV2...")

base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze pretrained layers initially
base_model.trainable = False

# ==========================================
# BUILD MODEL
# ==========================================

inputs = layers.Input(
    shape=(IMG_SIZE, IMG_SIZE, 3)
)

x = data_augmentation(inputs)

# MobileNetV2 expects pixels in [-1, 1]
x = layers.Rescaling(
    1.0 / 127.5,
    offset=-1
)(x)

x = base_model(
    x,
    training=False
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dense(
    128,
    activation="relu"
)(x)

x = layers.Dropout(0.4)(x)

outputs = layers.Dense(
    1,
    activation="sigmoid"
)(x)

model = models.Model(
    inputs,
    outputs
)

# ==========================================
# COMPILE
# ==========================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print()
print("======================================")
print("MODEL SUMMARY")
print("======================================")

model.summary()

# ==========================================
# CALLBACKS
# ==========================================

callbacks = [

    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=3,
        mode="max",
        restore_best_weights=True
    ),

    tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATH,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True
    )
]

# ==========================================
# TRAIN
# ==========================================

print()
print("======================================")
print("Starting training...")
print("======================================")

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ==========================================
# SAVE MODEL
# ==========================================

model.save(MODEL_PATH)

print()
print("======================================")
print("Training completed successfully!")
print("======================================")

print()
print("Model saved to:")
print(MODEL_PATH)

# ==========================================
# ACCURACY GRAPH
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(MODEL_DIR, "accuracy.png")
)

plt.close()

# ==========================================
# LOSS GRAPH
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(MODEL_DIR, "loss.png")
)

plt.close()

print()
print("Graphs saved successfully!")
print("model/accuracy.png")
print("model/loss.png")

print()
print("======================================")
print("Class mapping:")
print("0 =", class_names[0])
print("1 =", class_names[1])
print("======================================")