import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

X = np.load("data/spectrograms_X.npy")
y = np.load("data/spectrograms_y.npy")

print(f"Loaded X: {X.shape}, y: {y.shape}")

# turn "real"/"fake" into 0/1 since the model needs numbers not text
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
print(f"classes: {encoder.classes_}")

# CNN wants a channel dim even tho this is basically grayscale
X = X[..., np.newaxis]
print(f"X after channel dim: {X.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

print(f"train: {len(X_train)}, test: {len(X_test)}")

model = keras.Sequential([
    layers.Input(shape=(128, 63, 1)),
    layers.Conv2D(16, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dropout(0.5),  # added this bc it was overfitting hard before
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

# stop early if it's not actually improving anymore
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=4,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    epochs=30,
    validation_data=(X_test, y_test),
    batch_size=16,
    callbacks=[early_stop]
)

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"\nFinal accuracy: {test_accuracy:.2%}")

model.save("models/cnn_model.keras")
print("saved model to models/cnn_model.keras")
