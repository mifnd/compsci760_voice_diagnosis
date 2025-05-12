import numpy as np
import xgboost as xgb
import argparse
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    precision_recall_curve
from sklearn.preprocessing import LabelEncoder

from utils.train_test_split import train_test_split, cross_val_by_pnum, encode_y


def run_xgb_pipeline(csv_path):
    train_df, test_df = train_test_split(csv_path)

    y_col = "disease_label"
    X_cols = [c for c in train_df.columns if c.startswith("mfcc_")]

    le = LabelEncoder()
    y_train_enc = le.fit_transform(train_df[y_col])  # ['Normal','Vox_senilis','Laryngozele'] -> [1,2,0]
    y_test_enc = le.transform(test_df[y_col])

    model = xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        objective="multi:softprob",
        num_class=len(le.classes_))
    param_grid = {
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1],
        "n_estimators": [50, 100],
    }

    grid = GridSearchCV(model, param_grid, cv=5, scoring="roc_auc_ovr", n_jobs=-1, verbose=1)
    grid.fit(train_df[X_cols], y_train_enc)

    # y_pred_enc = grid.predict(test_df[X_cols])
    y_proba = grid.predict_proba(test_df[X_cols])

    best_thresholds = {}
    for idx, cls in enumerate(le.classes_):
        precision, recall, thresh = precision_recall_curve(
            (y_test_enc == idx).astype(int),
            y_proba[:, idx]
        )
        f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
        best_idx = f1_scores.argmax()
        best_thresholds[idx] = thresh[best_idx]

    y_pred_opt = []
    for probs in y_proba:
        candidates = [i for i, p in enumerate(probs) if p >= best_thresholds[i]]
        if candidates:
            chosen = max(candidates, key=lambda i: probs[i])
        else:
            chosen = probs.argmax()
        y_pred_opt.append(chosen)
    y_pred_opt = np.array(y_pred_opt)

    print("Best params:", grid.best_params_)
    print("Accuracy:", accuracy_score(y_test_enc, y_pred_opt))
    print("Precision (macro):", precision_score(y_test_enc, y_pred_opt, average="macro"))
    print("Recall (macro):",    recall_score(y_test_enc, y_pred_opt, average="macro"))
    print("F1 (macro):",        f1_score(y_test_enc, y_pred_opt, average="macro"))
    print("ROC AUC (ovr):",     roc_auc_score(y_test_enc, y_proba, multi_class="ovr"))


parser = argparse.ArgumentParser()
parser.add_argument(
    "--datasets", nargs="+",
    default=[
        "data/processed/10mfcc_mean.csv",
        "data/processed/aug_pitch_down_10mfcc_mean.csv",
        "data/processed/aug_pitch_up_10mfcc_mean.csv"
    ],
    help="List of dataset"
)
args = parser.parse_args()

for ds in args.datasets:
    run_xgb_pipeline(ds)
