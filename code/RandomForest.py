import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from feature_extraction import compute_mean_mfcc

# Read the dataset
git_root = Path().resolve().parent
data = pd.read_csv(git_root / "recording_data.csv", index_col=0)

# Construct the list of file paths
recording_paths = [git_root / Path(p) for p in data["file_path"]]


# Define the feature extraction function
def compute_x(paths: list[Path], num_mfcc: int):
    mfcc = np.zeros((len(paths), num_mfcc), dtype=float)
    for i in range(len(paths)):
        mfcc[i, :] = compute_mean_mfcc(paths[i], num_mfcc)
    return mfcc


# Compute the MFCC feature matrix (using 10 MFCC features)
X = compute_x(recording_paths, 10)

# Process labels by encoding the string labels to integer labels
# It is assumed that the possible labels are 'Normal', 'Laryngozele', 'Vox senilis'
y_classes = ['Normal', 'Laryngozele', 'Vox senilis']
y_labels = data["label"]
y = np.array([y_classes.index(label) for label in y_labels])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict on the test set
y_pred = rf_model.predict(X_test)

# Compute evaluation metrics
accuracy = metrics.accuracy_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred, average="weighted")
recall = metrics.recall_score(y_test, y_pred, average="weighted")
f1 = metrics.f1_score(y_test, y_pred, average="weighted")

print("Accuracy: ", round(accuracy, 4))
print("Average Precision: ", round(precision, 4))
print("Average Recall: ", round(recall, 4))
print("Average F1: ", round(f1, 4))
