import librosa
import numpy as np
import os

folders = {
    "real": "data/real",
    "fake": "data/fake"
}

X = []  # will hold all spectrograms
y = []  # will hold all labels

TARGET_SHAPE = (128, 63)  # (frequency bins, time steps) - fixed size for every clip

for label, folder in folders.items():
    files = os.listdir(folder)
    print(f"Processing {len(files)} files from {folder}...")

    for i, filename in enumerate(files):
        filepath = os.path.join(folder, filename)

        try:
            audio, sr = librosa.load(filepath, sr=16000)

            # Generate a mel spectrogram (image-like representation of audio)
            spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
            spec_db = librosa.power_to_db(spec, ref=np.max)

            # Pad or trim to make sure every spectrogram is the exact same size
            if spec_db.shape[1] < TARGET_SHAPE[1]:
                pad_width = TARGET_SHAPE[1] - spec_db.shape[1]
                spec_db = np.pad(spec_db, ((0, 0), (0, pad_width)), mode='constant')
            else:
                spec_db = spec_db[:, :TARGET_SHAPE[1]]

            X.append(spec_db)
            y.append(label)

        except Exception as e:
            print(f"Skipped {filename} due to error: {e}")

        if (i + 1) % 50 == 0:
            print(f"  ...processed {i + 1}/{len(files)}")

X = np.array(X)
y = np.array(y)

print(f"Final shape of X: {X.shape}")
print(f"Final shape of y: {y.shape}")

# Save so we don't have to redo this every time
np.save("data/spectrograms_X.npy", X)
np.save("data/spectrograms_y.npy", y)
print("Saved spectrograms to data/spectrograms_X.npy and data/spectrograms_y.npy")
