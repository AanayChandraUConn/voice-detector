import librosa
import numpy as np
import os
from tensorflow import keras

TARGET_SHAPE = (128, 63)
folder = "data/unseen_fake"  # clips from elevenlabs, model never trained on these

X_new = []
filenames = []

files = os.listdir(folder)
print(f"testing on {len(files)} unseen clips...")

for filename in files:
    filepath = os.path.join(folder, filename)

    try:
        audio, sr = librosa.load(filepath, sr=16000)
        spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        spec_db = librosa.power_to_db(spec, ref=np.max)

        if spec_db.shape[1] < TARGET_SHAPE[1]:
            pad_width = TARGET_SHAPE[1] - spec_db.shape[1]
            spec_db = np.pad(spec_db, ((0, 0), (0, pad_width)), mode='constant')
        else:
            spec_db = spec_db[:, :TARGET_SHAPE[1]]

        X_new.append(spec_db)
        filenames.append(filename)

    except Exception as e:
        print(f"skipped {filename}: {e}")

X_new = np.array(X_new)
X_new = X_new[..., np.newaxis]

print(f"shape: {X_new.shape}")

model = keras.models.load_model("models/cnn_model.keras")
predictions_prob = model.predict(X_new)

print("\nresults (these are ALL supposed to be fake):")
for fname, prob in zip(filenames, predictions_prob):
    prob_value = float(prob[0])
    guess = "real" if prob_value > 0.5 else "fake"
    print(f"{fname}: guessed '{guess}' ({prob_value:.2f})")

correct = sum(1 for prob in predictions_prob if prob[0] <= 0.5)
print(f"\ngot {correct}/{len(predictions_prob)} right")
