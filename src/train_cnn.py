import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load the spectrograms saved in the previous step
X = np.load("data/spectrograms_X.npy")
y = np.load("data/spectrograms_y.npy")

print(f"Loaded X shape: {X.shape}, y shape: {y.shape}")

# Convert labels ("real"/"fake") into numbers (0/1) the model can use
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
print(f"Label classes: {encoder.classes_}")

# Add a "channel" dimension - CNNs expect images to have a color channel, even if grayscale
X = X[..., np.newaxis]
print(f"X shape after adding channel dimension: {X.shape}")

# Split into train/test, same 80/20 split as before
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

print(f"Training on {len(X_train)} clips, testing on {len(X_test)} clips")

# Build the CNN
model = keras.Sequential([
    layers.Input(shape=(128, 63, 1)),
    layers.Conv2D(16, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

# Stop training early if validation performance stops improving
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=4,
    restore_best_weights=True
)

# Train it
history = model.fit(
    X_train, y_train,
    epochs=30,
    validation_data=(X_test, y_test),
    batch_size=16,
    callbacks=[early_stop]
)

# Final evaluation
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"\nFinal test accuracy: {test_accuracy:.2%}")

# Save the trained model
model.save("models/cnn_model.keras")
print("Model saved to models/cnn_model.keras")
