import numpy as np
import pandas as pd
from pathlib import Path
from utils.recording_name_utils import get_patient_number, get_sound_type, get_is_egg
from utils.fe_utils import compute_mfcc


def create_csv_with_mfcc(recordings_path: Path, csv_output_path: Path, n_mfcc: int, avg_type: str):
    # Initialise lists
    recording_names = []
    disease_labels = []
    patient_numbers = []
    sound_types = []
    is_eggs = []
    mfccs = []

    # Extract patient information and calculate mfcc
    for disease_folder in recordings_path.iterdir():
        if not disease_folder.is_dir():
            continue
        for file in disease_folder.iterdir():
            if file.suffix != ".wav":
                continue
            if get_is_egg(file):
                continue
            recording_names.append(file.name)
            disease_labels.append(disease_folder.name)
            patient_numbers.append(get_patient_number(file))
            sound_types.append(get_sound_type(file))
            is_eggs.append(get_is_egg(file))
            mfccs.append(compute_mfcc(file_path=file, n_mfcc=n_mfcc, avg_type=avg_type))

    # Compile everything in a pandas dataframe
    mfcc_matrix = np.array(mfccs)
    df_voice_data = pd.DataFrame({
                "id": range(len(recording_names)),
                "patient_number": patient_numbers,
                "disease_label": disease_labels,
                "file_name": recording_names,
                "sound_type": sound_types,
                "is_egg": is_eggs})
    for i in range(0, n_mfcc):
        next_mfcc_column = pd.DataFrame({f"mfcc_{i+1}": mfcc_matrix[:, i]})
        df_voice_data = df_voice_data.join(next_mfcc_column)

    # Write to .csv file
    df_voice_data.to_csv(csv_output_path, index=False)
