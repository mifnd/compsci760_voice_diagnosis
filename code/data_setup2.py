import numpy as np
import pandas as pd
from pathlib import Path
from feature_extraction import compute_mean_mfcc

root = Path("..").resolve()
data_path = root / "data/raw/patient-vocal-dataset"
n_mfcc = 10

# Initialise lists
recording_names = []
disease_labels = []
patient_numbers = []
sound_types = []
is_eggs = []
mfccs = []

for disease_folder in data_path.iterdir():
    if not disease_folder.is_dir():
        continue
    for file in disease_folder.iterdir():
        if file.suffix != ".wav":
            continue
        recording_names.append(file.name)
        disease_labels.append(disease_folder.name)
        file_stem_parts = file.stem.split("-")
        patient_numbers.append(file_stem_parts[0])
        sound_types.append(file_stem_parts[1])
        is_egg = len(file_stem_parts) > 2 and file_stem_parts[2] == "egg"
        is_eggs.append(is_egg)
        mfccs.append(compute_mean_mfcc(file_path=file, n_mfcc=n_mfcc))

# compile everything in a pandas data frame
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

# write to .csv file
df_voice_data.to_csv(root / "data/interim/recording_data2.csv", index=False)

