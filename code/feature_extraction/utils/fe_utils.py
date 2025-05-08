import numpy as np
import librosa
from pathlib import Path

# This file contains methods for computing mfcc


def compute_mean_mfcc(file_path: Path, n_mfcc: int):
    y, sr = librosa.load(file_path)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc, axis=1)
