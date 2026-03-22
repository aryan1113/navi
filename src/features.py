from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureConfig:
    window_days: tuple[int, ...] = (30, 90)


def coerce_flag_dates(flag_base: pd.DataFrame) -> pd.DataFrame:
    flag = flag_base.copy()
    flag["decision_date"] = pd.to_datetime(flag["decision_date"], errors="coerce")
    if "decision_date_full" in flag.columns:
        flag["decision_date_full"] = pd.to_datetime(flag["decision_date_full"], errors="coerce")
    return flag


def prepare_ledger_with_leakage_filter(
    txn_ledger: pd.DataFrame,
    decision_dates: pd.DataFrame,
) -> pd.DataFrame:
    # Attaches decision_date to ledger and filters out transactions after the decision date.
    ledger = txn_ledger.copy()
    ledger["txnDate"] = pd.to_datetime(ledger["txnDate"], errors="coerce")
    decision_dates = decision_dates.copy()
    decision_dates["decision_date"] = pd.to_datetime(decision_dates["decision_date"], errors="coerce")

    # The raw ledger file can contain a `decision_date` column, if yes drop
    if "decision_date" in ledger.columns:
        ledger = ledger.drop(columns=["decision_date"])

    decision_dates = decision_dates.rename(columns={"decision_date": "decision_date_flag"})
    ledger = ledger.merge(decision_dates, on="decision_id", how="left")

    ledger["days_before"] = (ledger["decision_date_flag"] - ledger["txnDate"]).dt.days
    ledger = ledger[ledger["days_before"].notna() & (ledger["days_before"] >= 0)]
    return ledger


def build_decision_features(ledger: pd.DataFrame, window_days: int | None = None) -> pd.DataFrame:
    """
    Aggregate ledger into one row per decision_id
    """
    df = ledger if window_days is None else ledger[ledger["days_before"] <= window_days]

    agg = (
        df.groupby("decision_id")
        .agg(
            n_txns=("txnDate", "count"),
            n_accounts=("bankAccount", "nunique"),
            n_debit_txns=("debit", lambda x: x.notna().sum()),
            n_credit_txns=("credit", lambda x: x.notna().sum()),
            total_debit=("debit", "sum"),
            mean_debit=("debit", "mean"),
            max_debit=("debit", "max"),
            total_credit=("credit", "sum"),
            mean_credit=("credit", "mean"),
            max_credit=("credit", "max"),
            min_balance=("balance", "min"),
            max_balance=("balance", "max"),
            mean_balance=("balance", "mean"),
            last_balance=("balance", "last"),
            std_balance=("balance", "std"),
            days_since_last_txn=("days_before", "min"),
            days_since_first_txn=("days_before", "max"),
            txn_span_days=("days_before", lambda x: x.max() - x.min()),
        )
        .reset_index()
    )

    eps = 1e-9
    agg["credit_debit_ratio"] = agg["total_credit"] / (agg["total_debit"] + eps)
    agg["balance_range"] = agg["max_balance"] - agg["min_balance"]
    agg["debit_txn_ratio"] = agg["n_debit_txns"] / (agg["n_txns"] + eps)
    agg["negative_balance_flag"] = (agg["min_balance"] < 0).astype(int)

    suffix = f"_w{window_days}d" if window_days is not None else ""
    rename_map = {c: f"{c}{suffix}" for c in agg.columns if c != "decision_id"}
    return agg.rename(columns=rename_map)


def assemble_master_table(
    flag_base: pd.DataFrame,
    txn_ledger: pd.DataFrame,
    cfg: FeatureConfig = FeatureConfig(),
) -> pd.DataFrame:
    flag = coerce_flag_dates(flag_base)

    decision_dates = flag[["decision_id", "decision_date"]].drop_duplicates()
    ledger = prepare_ledger_with_leakage_filter(txn_ledger, decision_dates)

    feat_all = build_decision_features(ledger, window_days=None)
    feat_windows = [build_decision_features(ledger, window_days=w) for w in cfg.window_days]

    master = flag.copy()
    for feat_df in [feat_all, *feat_windows]:
        master = master.merge(feat_df, on="decision_id", how="left")

    master["decision_month"] = pd.to_datetime(master["decision_date"], errors="coerce").dt.month
    master["decision_dayofweek"] = pd.to_datetime(master["decision_date"], errors="coerce").dt.dayofweek
    if "decision_date_full" in master.columns:
        master["decision_hour"] = pd.to_datetime(master["decision_date_full"], errors="coerce").dt.hour
    else:
        master["decision_hour"] = np.nan
    return master
