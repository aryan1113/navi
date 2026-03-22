from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features import FeatureConfig, assemble_master_table


def pick_threshold_max_f1(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    prec, rec, thresh = precision_recall_curve(y_true, y_prob)
    # last point has no trheshold, so ignore
    f1_score = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
    best_idx = int(np.argmax(f1_score))
    return float(thresh[best_idx])


def build_pipeline(model_name: str, numeric_features: list[str], random_state: int) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            )
        ],
        remainder="drop",
    )

    if model_name == "lr":
        clf = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=random_state,
        )
    elif model_name == "rf":
        clf = RandomForestClassifier(
            n_estimators=500,
            max_depth=8,
            min_samples_leaf=25,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown model_name={model_name!r}. Use 'lr' or 'rf'.")

    return Pipeline([("pre", pre), ("clf", clf)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--model", choices=["lr", "rf"], default="lr")
    ap.add_argument("--val-size", type=float, default=0.2)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    flag_base = pd.read_csv(
        data_dir / "assignment_flag_base.csv",
        parse_dates=["decision_date", "decision_date_full"],
    )
    txn_ledger = pd.read_csv(
        data_dir / "assignment_txn_ledger.csv",
        parse_dates=["txnDate"],
    )

    master = assemble_master_table(flag_base, txn_ledger, cfg=FeatureConfig())

    target = "default"
    train_df = master[(master["type"] == "train") & master[target].notna()].copy()

    drop_cols = ["decision_id", "decision_date", "decision_date_full", "type", target]
    X = train_df.drop(columns=drop_cols, errors="ignore")
    y = train_df[target].astype(int).to_numpy()

    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    X = X[numeric_features]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=args.val_size,
        random_state=args.random_state,
        stratify=y,
    )

    pipe = build_pipeline(args.model, numeric_features=numeric_features, random_state=args.random_state)
    pipe.fit(X_train, y_train)

    y_prob = pipe.predict_proba(X_val)[:, 1]
    metrics = {
        "roc_auc": float(roc_auc_score(y_val, y_prob)),
        "pr_auc": float(average_precision_score(y_val, y_prob)),
        "val_size": float(args.val_size),
        "model": args.model,
    }
    threshold = pick_threshold_max_f1(y_val, y_prob)

    # fit on all train rows
    pipe.fit(X, y)

    model_path = models_dir / "pd_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(
            {
                "model": pipe,
                "threshold": float(threshold),
                "features": numeric_features,
                "metrics": metrics,
            },
            f,
        )

    threshold_path = models_dir / "threshold.json"
    threshold_path.write_text(json.dumps({"threshold": float(threshold)}, indent=2))

    metrics_path = models_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print(f"Saved model → {model_path}")
    print(f"Saved threshold → {threshold_path} (threshold={threshold:.4f})")
    print(f"Saved metrics → {metrics_path} (roc_auc={metrics['roc_auc']:.4f}, pr_auc={metrics['pr_auc']:.4f})")


if __name__ == "__main__":
    main()
