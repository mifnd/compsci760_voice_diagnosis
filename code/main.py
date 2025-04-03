import numpy as np
import pandas as pd
import librosa
from pathlib import Path


def compute_mfcc_ah():
    directory = Path("../data/raw/patient-vocal-dataset/")
    number_of_mfcc = 10

    patient_numbers = []
    mfcc_all_patients = []
    for file in directory.glob('**/*-a_h.wav'):
        patient_numbers.append(file.name[:file.name.index('-')])
        y, sr = librosa.load(file)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=number_of_mfcc)
        mfcc_all_patients.append(np.mean(mfcc, axis=1))

    column_names = ['mean_mfcc_' + str(i) for i in range(1, 11)]
    df = pd.DataFrame(data=mfcc_all_patients, index=patient_numbers, columns=column_names)
    print(df)




