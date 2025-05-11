# Library imports
import numpy as np

# Test libraries
from sklearn import metrics
from xgboost import XGBClassifier
from train_test_split import cross_val_by_pnum, train_test_split

# Test Example
# Create model
xgb = xgb_final = XGBClassifier(learning_rate = 0.2, 
                          max_depth = 6, 
                          min_child_weight = 0.9, 
                          subsample = 0.7, 
                          colsample_bytree = 0.8, 
                          objective = "multi: softmax")

# Read dataset
dataset, temp = train_test_split("data/processed/10mfcc_mean.csv")

# Correct column names 
y_lab = "disease_label"
X_lab = np.array([f"mfcc_{i + 1}" for i in range(10)])

# Use the cross_val_by_pnum function
cval = cross_val_by_pnum(data = dataset, y_col= y_lab, X_cols= X_lab, model = xgb, scoring = metrics.accuracy_score)
print(cval)