# Package imports
from pathlib import Path
import pandas as pd
import numpy as np

# Function to perform train / test split
# parameter file_path takes the file path relative to compsci760_voice_diagnosis to a .csv file with a column named patient_number
# function returns two pandas data frames with train / test data
def train_test_split(file_path = ""):
    root = Path().resolve().parents[2]
    pn_train = np.loadtxt(root/"data/train.txt", dtype=int)
    pn_test = np.loadtxt(root/"data/test.txt", dtype=int)
    all_data = pd.read_csv(root/file_path, index_col="id")
    train = all_data[all_data["patient_number"] .isin(pn_train)]
    test = all_data[all_data["patient_number"].isin(pn_test)]
    return (train, test)


