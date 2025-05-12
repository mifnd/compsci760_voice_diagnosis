# Library imports
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn import model_selection

# Function to perform train / test split
# Parameters:
#   file_path: takes the file path relative to compsci760_voice_diagnosis to a .csv file with a column named patient_number
# function returns two pandas data frames with train / test data (similar to the original dataframe)
def train_test_split(file_path = ""):
    root = Path().resolve().parents[1]
    pn_train = np.loadtxt(root/"data/train.txt", dtype=int)
    pn_test = np.loadtxt(root/"data/test.txt", dtype=int)
    all_data = pd.read_csv(root/file_path, index_col="id")
    train = all_data[all_data["patient_number"] .isin(pn_train)]
    test = all_data[all_data["patient_number"].isin(pn_test)]
    return (train, test)

# Function to encode string classes as integers
def encode_y(labels, classes):
    y = np.zeros(len(labels), dtype=int)
    for i in range(len(labels)):
        y[i] = classes.index(labels[i])
    return y

# Function to cross validate according to patient number
# Parameters:
#   data: a pandas dataframe
#   y: str of response column name
#   X: numpy array of str of predictor column names
#   model: a classifier model with model.fit(X,y) and model.predict(y) methods (according to the style of sklearn)
#           or can be False to output lists with split indeces
#   scoring: a function of the form function(test, pred) and returns an integer
#   n_folds: an integer for the number of cross validation folds
#   y_classes: list of strings that contain the unique classes for prediction

def cross_val_by_pnum(data, y_col, X_cols, model = None, scoring = None, n_folds = 5, y_classes = ['Vox_senilis', 'Laryngozele', 'Normal']):
    k_fold = model_selection.GroupKFold(n_splits = n_folds)
    cross_val = enumerate(k_fold.split(data[X_cols], data[y_col], data["patient_number"]))
    # Return list with train/test indices for each fold if no model or scoring is given
    if model is None or scoring is None:
        return [item[1:2][0] for item in cross_val]
    # Otherwise caluclate crossval scores using provided scoring function and model
    scores = np.zeros(n_folds, dtype=float)
    for i, (train_index, test_index) in cross_val:
        y_train = encode_y(np.array(data[y_col].iloc[train_index]), y_classes)
        X_train = data[X_cols].iloc[train_index]
        y_test = encode_y(np.array(data[y_col].iloc[test_index]), y_classes)
        X_test = data[X_cols].iloc[test_index]
        model.fit(X_train, y_train)
        scores[i] = scoring(y_test, model.predict(X_test))
    return scores