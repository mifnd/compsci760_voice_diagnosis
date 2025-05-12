import numpy as np
import librosa
from pathlib import Path

# This file contains methods for computing mfcc


def compute_mean_mfcc(file_path: Path, n_mfcc: int):
    y, sr = librosa.load(file_path)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc, axis=1)


def compute_median_mfcc(file_path: Path, n_mfcc: int):
    y, sr = librosa.load(file_path)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.median(mfcc, axis=1)


def compute_mfcc(file_path: Path, n_mfcc: int, avg_type: str):
    if avg_type == "median":
        compute_median_mfcc(file_path=file_path, n_mfcc=n_mfcc)
    else:
        compute_mean_mfcc(file_path=file_path, n_mfcc=n_mfcc)
