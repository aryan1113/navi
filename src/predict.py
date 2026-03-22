from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd

from src.features import FeatureConfig, assemble_master_table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--outputs-dir", default="outputs")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    models_dir = Path(args.models_dir)
    outputs_dir = Path(args.outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    with open(models_dir / "pd_model.pkl", "rb") as f:
        artifacts = pickle.load(f)
    model = artifacts["model"]
    features: list[str] = artifacts["features"]

    flag_base = pd.read_csv(
        data_dir / "assignment_flag_base.csv",
        parse_dates=["decision_date", "decision_date_full"],
    )
    txn_ledger = pd.read_csv(
        data_dir / "assignment_txn_ledger.csv",
        parse_dates=["txnDate"],
    )

    master = assemble_master_table(flag_base, txn_ledger, cfg=FeatureConfig())
    oot = master[master["type"] == "oot"].copy()

    X = oot.reindex(columns=features)
    probs = model.predict_proba(X)[:, 1]

    out = pd.DataFrame(
        {
            "decision_id": oot["decision_id"].values,
            "predicted_PD": probs.astype(float),
        }
    )
    out_path = outputs_dir / "oot_predictions.csv"
    out.to_csv(out_path, index=False)
    print(f"Wrote predictions → {out_path} ({len(out):,} rows)")


if __name__ == "__main__":
    main()
