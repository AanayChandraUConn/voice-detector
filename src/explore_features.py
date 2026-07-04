import librosa
import numpy as np
import os

# Pick one real file to explore
real_folder = "data/real"
first_file = os.listdir(real_folder)[0]
filepath = os.path.join(real_folder, first_file)

print(f"Looking at file: {first_file}")

# Load the audio
y, sr = librosa.load(filepath, sr=16000)

print(f"Audio loaded. Number of samples: {len(y)}, Sample rate: {sr}")

# Extract MFCCs
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

print(f"MFCC shape: {mfccs.shape}")

# Average across time to get ONE fixed-length vector per clip
mfcc_mean = np.mean(mfccs, axis=1)

print(f"Final feature vector (13 numbers): {mfcc_mean}")
