import os
from itertools import product
import numpy as np
import pandas as pd
import xgboost as xgb
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import compute_class_weight
from utils.train_test_split import train_test_split, find_root_path


def run_xgb_pipeline(main_csv, train_txt, test_txt, aug_csv_list=None, script_name=None):
    train_df, test_df = train_test_split(
        file_path=main_csv,
        train_file=train_txt,
        test_file=test_txt
    )

    if aug_csv_list:
        for aug_csv in aug_csv_list:
            aug_train, _ = train_test_split(
                file_path=aug_csv,
                train_file=train_txt,
                test_file=test_txt
            )
            train_df = pd.concat([train_df, aug_train], ignore_index=True)

    # Make the label column categorical
    y_col = "disease_label"
    X_cols = [c for c in train_df.columns if c.startswith("mfcc_")]
    le = LabelEncoder()
    y_train_enc = le.fit_transform(train_df[y_col])
    y_test_enc = le.transform(test_df[y_col])
    n_classes = len(le.classes_)

    # Weights for each class
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train_enc),
        y=y_train_enc
    )
    sample_weights = np.array([class_weights[i] for i in y_train_enc])

    # Hyperparameter tuning using GroupKFold
    # Hyperparameter grid
    groups = train_df["patient_number"].values
    param_grid = {
        "max_depth": [5, 7, 9],
        "learning_rate": [0.01, 0.05, 0.1],
        "n_estimators": [100, 200, 300],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0]
    }
    cv = GroupKFold(n_splits=5)
    best_score, best_params = grouped_hyperparam_search(
        train_df, y_train_enc, sample_weights,
        X_cols, n_classes, param_grid, groups, cv
    )

    # Train the final model with the best parameters
    final_model = xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric="mlogloss",
        objective="multi:softprob",
        num_class=n_classes,
        **best_params
    )
    final_model.fit(
        train_df[X_cols], y_train_enc,
        sample_weight=sample_weights
    )

    y_proba = final_model.predict_proba(test_df[X_cols])
    best_thresholds = optimize_thresholds(y_test_enc, y_proba)

    y_pred_opt = []
    for probs in y_proba:
        candidates = [i for i, p in enumerate(probs) if p >= best_thresholds[i]]
        if candidates:
            y_pred_opt.append(max(candidates, key=lambda i: probs[i]))
        else:
            y_pred_opt.append(np.argmax(probs))
    y_pred_opt = np.array(y_pred_opt)

    # Evaluate the model
    print("\n=== Test Set Evaluation ===")
    print(f"Accuracy:           {accuracy_score(y_test_enc, y_pred_opt):.4f}")
    print(f"Precision (macro):  {precision_score(y_test_enc, y_pred_opt, average='macro'):.4f}")
    print(f"Recall (macro):     {recall_score(y_test_enc, y_pred_opt, average='macro'):.4f}")
    print(f"F1 Score (macro):   {f1_score(y_test_enc, y_pred_opt, average='macro'):.4f}")
    print(f"ROC AUC (ovr):      {roc_auc_score(y_test_enc, y_proba, multi_class='ovr'):.4f}")

    results_path = find_root_path() / "results" / f"predictions_{script_name}_{datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")}.txt"
    results_txt = open(results_path, "w")
    results_txt.write("Accuracy,Precision(macro),Recall(macro),F1_Score(macro),ROC_AUC(ovr)\n")
    results_txt.write((f"{accuracy_score(y_test_enc, y_pred_opt)},"
                       f"{precision_score(y_test_enc, y_pred_opt, average='macro')},"
                       f"{recall_score(y_test_enc, y_pred_opt, average='macro')},"
                       f"{f1_score(y_test_enc, y_pred_opt, average='macro')},"
                       f"{roc_auc_score(y_test_enc, y_proba, multi_class='ovr')}"))
    results_txt.close()

    # Record predictions in csv file
    pd.DataFrame(data={"id": test_df.index,
                       "patient_number": test_df["patient_number"],
                       "disease_label": test_df["disease_label"],
                       "sound_type": test_df["sound_type"],
                       "y_real": y_test_enc,
                       "y_pred": y_pred_opt}).to_csv(
        find_root_path() / "results" / f"results_{datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")}.csv", index=False)


# Function to optimize thresholds for each class
# Parameters:
#   y_true: true labels
#   y_proba: predicted probabilities for each class
def optimize_thresholds(y_true, y_proba):
    best_thresholds = {}
    n_classes = y_proba.shape[1]
    for idx in range(n_classes):
        precision, recall, thresh = precision_recall_curve(
            (y_true == idx).astype(int), y_proba[:, idx]
        )
        # compute F1 value for each threshold
        f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
        # select threshold with largest F1, the default is 0.5 if there is no thresholds
        best_thresholds[idx] = thresh[np.argmax(f1_scores)] if thresh.size else 0.5
    return best_thresholds


# Function to perform grouped hyperparameter search
# Parameters:
#   combined_train: training data, y_train_enc: encoded labels, sample_weights: sample weights
#   X_cols: feature columns, n_classes: number of classes, param_grid: hyperparameter grid
#   groups: patient groups for cross-validation, cv: cross-validation strategy
def grouped_hyperparam_search(combined_train, y_train_enc, sample_weights, X_cols,
                              n_classes, param_grid, groups, cv):
    best_score = -np.inf
    best_params = None
    # Iterate over all combinations of hyperparameters
    for combo in product(*param_grid.values()):
        params = dict(zip(param_grid.keys(), combo))
        model = xgb.XGBClassifier(
            use_label_encoder=False,
            eval_metric="mlogloss",
            objective="multi:softprob",
            num_class=n_classes,
            **params
        )
        fold_scores = []
        # Perform GroupKFold cross-validation
        for train_idx, val_idx in cv.split(combined_train, y_train_enc, groups):
            X_tr = combined_train.iloc[train_idx][X_cols]
            y_tr = y_train_enc[train_idx]
            w_tr = sample_weights[train_idx]

            X_val = combined_train.iloc[val_idx][X_cols]
            y_val = y_train_enc[val_idx]

            # Fit the model
            model.fit(X_tr, y_tr, sample_weight=w_tr)
            y_pred = model.predict(X_val)
            fold_scores.append(f1_score(y_val, y_pred, average='macro'))

        # Calculate the mean score for this combination of hyperparameters
        mean_score = np.mean(fold_scores)
        if mean_score > best_score:
            best_score, best_params = mean_score, params

    return best_score, best_params
