import os
import soundfile as sf
import librosa
from pathlib import Path
from utils.recording_name_utils_copy import get_is_egg

# Divide octave in 24 steps and set pitch shift to 1 step up.
bins_per_octave = 24
n_steps = -1

# Data paths
root = Path("../..").resolve()
data_input_folder = root / "data/raw/patient-vocal-dataset"
data_output_folder = root / "data/interim/aug_pitch_down"

# For each disease folder, create a new folder (same disease name) in the output folder.
# For each recording, shift the pitch down and save it as an .wav-file in the output disease folder.
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
        y_new = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps, bins_per_octave=bins_per_octave)
        output_path = data_output_folder / disease_folder.name / file.name
        sf.write(output_path, y_new, sr, format='WAV')
