import numpy as np
import pandas as pd
from pathlib import Path
from utils.recording_name_utils import get_patient_number, get_sound_type, get_is_egg
from utils.fe_utils import compute_mean_mfcc

root = Path("../..").resolve()
data_path = root / "data/raw/patient-vocal-dataset"
n_mfcc = 10

# Initialise lists
recording_names = []
disease_labels = []
patient_numbers = []
sound_types = []
is_eggs = []
mfccs = []

# Extract patient information and calculate mfcc
for disease_folder in data_path.iterdir():
    if not disease_folder.is_dir():
        continue
    for file in disease_folder.iterdir():
        if file.suffix != ".wav":
            continue
        recording_names.append(file.name)
        disease_labels.append(disease_folder.name)
        patient_numbers.append(get_patient_number(file))
        sound_types.append(get_sound_type(file))
        is_eggs.append(get_is_egg(file))
        mfccs.append(compute_mean_mfcc(file_path=file, n_mfcc=n_mfcc))

# Compile everything in a pandas dataframe
mfcc_matrix = np.array(mfccs)
df_voice_data = pd.DataFrame({
            "id": range(len(recording_names)),
            "patient_number": patient_numbers,
            "disease_label": disease_labels,
            "file_name": recording_names,
            "sound_type": sound_types,
            "is_egg": is_eggs,
            "mfcc_1": mfcc_matrix[:, 0],
            "mfcc_2": mfcc_matrix[:, 1],
            "mfcc_3": mfcc_matrix[:, 2],
            "mfcc_4": mfcc_matrix[:, 3],
            "mfcc_5": mfcc_matrix[:, 4],
            "mfcc_6": mfcc_matrix[:, 5],
            "mfcc_7": mfcc_matrix[:, 6],
            "mfcc_8": mfcc_matrix[:, 7],
            "mfcc_9": mfcc_matrix[:, 8],
            "mfcc_10": mfcc_matrix[:, 9]})

# Write to .csv file
df_voice_data.to_csv(root / "data/processed/10mfcc_mean.csv", index=False)

