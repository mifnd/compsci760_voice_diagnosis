# Package imports
import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBClassifier
from sklearn import metrics

# Function imports
from feature_extraction import compute_mean_mfcc
from sklearn.model_selection import train_test_split, cross_val_score

# Data import
git_root = Path().resolve().parent
data = pd.read_csv( git_root / "recording_data.csv", index_col = 0)

# Calculate X matrix of MFCC vectors
recording_paths = []
for path in data.loc[:, "file_path"]:
    recording_paths.append(git_root / Path(path))

def compute_x(paths: list[Path], num_mfcc: int):
    mfcc = np.zeros((len(paths), num_mfcc), dtype=float)
    for i in range(len(paths)):
        mfcc[i, :] = compute_mean_mfcc(paths[i], num_mfcc)
    return mfcc

X = compute_x(recording_paths, 10)

# Compute y vector of responses
y_labels = data.loc[:, "label"]

y_classes = ['Normal', 'Laryngozele', 'Vox senilis'] # Encode 0 = Normal, 1 = Laryngozele, 2 = Vox senilis

y = np.zeros(len(y_labels), dtype=int)
for i in range(len(y_labels)):
    y[i] = y_classes.index(y_labels[i])

# Test/Train split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

# Hyper-Parameter tuning
# I'm commenting out this part as it took mare than 4 hours to run
'''
# Recall scorer
def recall_scorer(estimator, X, y):
    return metrics.recall_score(y, estimator.predict(X), average="weighted")

# Crossval model with certain parameters
def xgb_cross( X, y, learning_rate: float, max_depth: int, min_child_weight: float, subsample: float, colsample_bytree:float, cv: int = 10, scoring = recall_scorer):
    xgb = XGBClassifier(learning_rate = learning_rate,
                         max_depth = max_depth, 
                         min_child_weight = min_child_weight,
                         subsample = subsample,
                         colsample_bytree = colsample_bytree,
                         objective = "multi: softmax")
    return cross_val_score(xgb, X, y, scoring=scoring, cv=cv)

# Tune learning_rate + max_depth + min_child_weight - Exhaustive search
# WARNING!! this took 247min to run
lr_vals = (np.array(range(20)) + 1) * 0.01 
md_vals = np.array(range(20)) + 1
mcw_vals = (np.array(range(10)) + 1) * 0.1
average_recall = np.zeros((len(lr_vals), len(md_vals), len(mcw_vals)))

for i in range(len(lr_vals)):
    for j in range(len(md_vals)):
        for k in range(len(mcw_vals)):
            average_recall[i, j, k] = xgb_cross(X_train, y_train, learning_rate=lr_vals[i], max_depth=md_vals[j], min_child_weight=mcw_vals[k], subsample=1, colsample_bytree=1).mean()

# Tune subsample, and colsample_bytree - Exhaustive search
# WARNING!! this took 2m 30s to run
ss_vals = (np.array(range(10)) + 1) * 0.1
cbt_vals = (np.array(range(10)) + 1) * 0.1
average_recall_2 = np.zeros((len(ss_vals), len(cbt_vals)))

for i in range(len(ss_vals)):
    for j in range(len(cbt_vals)):
        average_recall_2[i, j] = xgb_cross(X_train, y_train, learning_rate=0.2, max_depth=6, min_child_weight=0.9, subsample=ss_vals[i], colsample_bytree=cbt_vals[j]).mean()
        #print(i, j)

best_index = np.unravel_index(np.argmax(average_recall), average_recall.shape)
best_lr = lr_vals[best_index[0]]
best_md = md_vals[best_index[1]]
best_mcw = mcw_vals[best_index[2]]

print(best_lr, best_md, best_mcw) # lr = 0.2, md = 6, mcw = 0.9

best_index_2 = np.unravel_index(np.argmax(average_recall_2), average_recall_2.shape)
best_ss = ss_vals[best_index_2[0]]
best_cbt = cbt_vals[best_index_2[1]]

print(best_ss, best_cbt) # ss = 0.7, cbt = 0.8
average_recall_2.max()

'''

# Building and testing final model

xgb_final = XGBClassifier(learning_rate = 0.2, 
                          max_depth = 6, 
                          min_child_weight = 0.9, 
                          subsample = 0.7, 
                          colsample_bytree = 0.8, 
                          objective = "multi: softmax")

xgb_final.fit(X_train, y_train)
xgb_pred = xgb_final.predict(X_test)

# Metrics
print("Accuracy: ", metrics.accuracy_score(y_test, xgb_pred))
print("Average Precision: ", metrics.precision_score(y_test, xgb_pred, average="weighted"))
print("Average Recall: ", metrics.recall_score(y_test, xgb_pred, average="weighted"))
print("Average F1: ", metrics.f1_score(y_test, xgb_pred, average="weighted"))