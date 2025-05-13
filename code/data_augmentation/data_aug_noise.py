import os
import soundfile as sf
import librosa
import numpy as np
from pathlib import Path
from utils.recording_name_utils_copy import get_is_egg

# This script makes a new dataset with added random background noise.

# Set level of background noise:
noise_level = 0.005

# Data paths
root = Path("../..").resolve()
data_input_folder = root / "data/raw/patient-vocal-dataset"
data_output_folder = root / "data/interim/aug_noise"

# For each disease folder, create a new folder (same disease name) in the output folder.
# For each recording, add random background noise. Ensure that the numpy array representing
# the recording with background noise only has values in the interval [-1, 1].
# Save the new recording as an .wav-file in the output disease folder.
# Skip egg-recordings.
for disease_folder in data_input_folder.iterdir():
    if not disease_folder.is_dir():
        continue
    disease_output_folder = data_output_folder / disease_folder.name
    os.makedirs(name=disease_output_folder, exist_ok=True)
    for file in disease_folder.iterdir():
        if file.suffix != ".wav":
            continue
        if get_is_egg(file):
            continue
        y, sr = librosa.load(file)
        random_noise = noise_level * np.random.normal(loc=0, scale=0.4, size=y.shape)
        y_with_noise = np.clip(a=y + random_noise, a_min=-1.0, a_max=1.0)
        output_path = data_output_folder / disease_folder.name / file.name
        sf.write(output_path, y_with_noise, sr, format='WAV')

