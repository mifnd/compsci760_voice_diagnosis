from itertools import product
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import compute_class_weight
from utils.train_test_split import train_test_split, cross_val_by_pnum, encode_y


def run_xgb_pipeline(csv_paths):
    train_parts = []
    for path in csv_paths:
        tr, _ = train_test_split(path)
        train_parts.append(tr)
    combined_train = pd.concat(train_parts, ignore_index=True)

    _, combined_test = train_test_split(csv_paths[0])

    y_col = "disease_label"
    X_cols = [c for c in combined_train.columns if c.startswith("mfcc_")]

    le = LabelEncoder()
    y_train_enc = le.fit_transform(combined_train[y_col])
    y_test_enc = le.transform(combined_test[y_col])
    n_classes = len(le.classes_)

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train_enc),
        y=y_train_enc
    )
    sample_weights = np.array([class_weights[i] for i in y_train_enc])

    param_grid = {
        "max_depth": [3, 5, 7, 9],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "n_estimators": [50, 100, 200, 300],
    }

    best_score = -np.inf
    best_params = None
    groups = combined_train["patient_number"].values
    cv = GroupKFold(n_splits=5)

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
        for train_idx, val_idx in cv.split(combined_train, y_train_enc, groups):
            X_tr = combined_train.iloc[train_idx][X_cols]
            y_tr = y_train_enc[train_idx]
            w_tr = sample_weights[train_idx]

            X_val = combined_train.iloc[val_idx][X_cols]
            y_val = y_train_enc[val_idx]

            model.fit(X_tr, y_tr, sample_weight=w_tr)
            y_pred = model.predict(X_val)
            fold_scores.append(recall_score(y_val, y_pred, average='macro'))

        mean_score = np.mean(fold_scores)
        if mean_score > best_score:
            best_score, best_params = mean_score, params

    final_model = xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric="mlogloss",
        objective="multi:softprob",
        num_class=n_classes,
        **best_params
    )
    final_model.fit(
        combined_train[X_cols], y_train_enc,
        sample_weight=sample_weights
    )

    y_proba = final_model.predict_proba(combined_test[X_cols])

    best_thresholds = {}
    for idx in range(n_classes):
        precision, recall, thresh = precision_recall_curve(
            (y_test_enc == idx).astype(int), y_proba[:, idx]
        )
        f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
        best_thresholds[idx] = thresh[np.argmax(f1_scores)] if len(thresh) > 0 else 0.5

    y_pred_opt = []
    for probs in y_proba:
        candidates = [i for i, p in enumerate(probs) if p >= best_thresholds[i]]
        if candidates:
            y_pred_opt.append(max(candidates, key=lambda i: probs[i]))
        else:
            y_pred_opt.append(np.argmax(probs))
    y_pred_opt = np.array(y_pred_opt)

    print("\n=== Test Set Evaluation ===")
    print(f"Accuracy:           {accuracy_score(y_test_enc, y_pred_opt):.4f}")
    print(f"Precision (macro):  {precision_score(y_test_enc, y_pred_opt, average='macro'):.4f}")
    print(f"Recall (macro):     {recall_score(y_test_enc, y_pred_opt, average='macro'):.4f}")
    print(f"F1 Score (macro):   {f1_score(y_test_enc, y_pred_opt, average='macro'):.4f}")
    print(f"ROC AUC (ovr):      {roc_auc_score(y_test_enc, y_proba, multi_class='ovr'):.4f}")


# parser = argparse.ArgumentParser(description="XGBoost pipeline with multi-dataset training and patient-group CV")
# parser.add_argument(
#     "--datasets", nargs="+", required=True,
#     help="List of dataset"
# )
# args = parser.parse_args()
# run_xgb_pipeline(args.datasets)
