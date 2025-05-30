import numpy as np
import librosa
from pathlib import Path

# This file contains methods for computing mfcc

# Computes mfcc for a recording at file_path. The default average type is the mean.
def compute_mfcc(file_path: Path, n_mfcc: int, avg_type: str):
    y, sr = librosa.load(file_path)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    if avg_type == "median":
        return np.median(mfcc, axis=1)
    else:
        return np.mean(mfcc, axis=1)
