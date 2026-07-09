import librosa
import numpy as np
import pandas as pd
import os

folders = {
    "real": "data/real",
    "fake": "data/fake"
}

rows = []

for label, folder in folders.items():
    files = os.listdir(folder)
    print(f"Processing {len(files)} files from {folder}...")

    for i, filename in enumerate(files):
        filepath = os.path.join(folder, filename)

        try:
            y, sr = librosa.load(filepath, sr=16000)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfccs, axis=1)  # avg over time so every clip = 13 numbers

            row = list(mfcc_mean) + [label]
            rows.append(row)
        except Exception as e:
            print(f"Skipped {filename}: {e}")

        if (i + 1) % 50 == 0:
            print(f"  ...{i + 1}/{len(files)}")

columns = [f"mfcc_{i}" for i in range(13)] + ["label"]
df = pd.DataFrame(rows, columns=columns)

df.to_csv("data/features.csv", index=False)
print(f"Saved {len(df)} rows to data/features.csv")
