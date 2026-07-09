import librosa
import numpy as np
import os

folders = {
    "real": "data/real",
    "fake": "data/fake"
}

X = []
y = []

TARGET_SHAPE = (128, 63)  # gotta keep every spectrogram the same size for the CNN

for label, folder in folders.items():
    files = os.listdir(folder)
    print(f"Processing {len(files)} files from {folder}...")

    for i, filename in enumerate(files):
        filepath = os.path.join(folder, filename)

        try:
            audio, sr = librosa.load(filepath, sr=16000)

            spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
            spec_db = librosa.power_to_db(spec, ref=np.max)

            # pad with silence if too short, chop off the end if too long
            if spec_db.shape[1] < TARGET_SHAPE[1]:
                pad_width = TARGET_SHAPE[1] - spec_db.shape[1]
                spec_db = np.pad(spec_db, ((0, 0), (0, pad_width)), mode='constant')
            else:
                spec_db = spec_db[:, :TARGET_SHAPE[1]]

            X.append(spec_db)
            y.append(label)

        except Exception as e:
            print(f"Skipped {filename}: {e}")

        if (i + 1) % 50 == 0:
            print(f"  ...{i + 1}/{len(files)}")

X = np.array(X)
y = np.array(y)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

np.save("data/spectrograms_X.npy", X)
np.save("data/spectrograms_y.npy", y)
print("saved to data/spectrograms_X.npy and data/spectrograms_y.npy")
