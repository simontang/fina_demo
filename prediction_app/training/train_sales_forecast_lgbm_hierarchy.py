#!/usr/bin/env python3
"""
Train a hierarchy-friendly global sales forecasting model (revenue) using LightGBM.

Why this script exists:
- The baseline model trained only on SKU-level series does not generalize well to
  category totals (value range is much larger).
- This script adds category-level and total-level entities to the training set so
  the model learns across multiple aggregation levels.

Outputs to repo root:
  models/{model_name}/{version}/model.pkl
  models/{model_name}/{version}/metadata.json

The saved model is wrapped in `shared.models.sales_forecast_wrapper.SalesForecastWrapper`
so inference automatically returns values in the original scale when a target transform
like log1p is used.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

try:
    import holidays  # type: ignore
except Exception as e:
    raise RuntimeError(f"Missing dependency 'holidays'. Install it first. Root cause: {e}")

try:
    import lightgbm as lgb  # type: ignore
except Exception as e:
    raise RuntimeError(f"Missing dependency 'lightgbm'. Install it first. Root cause: {e}")

# Ensure we can import from prediction_app/ when running from prediction_app/training/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.models.sales_forecast_wrapper import SalesForecastWrapper  # noqa: E402
from api.datasets import SALES_HIERARCHY_RULES, _sales_hierarchy_build_keyword_regex, _sales_hierarchy_norm  # noqa: E402


@dataclass
class TrainConfig:
    csv_path: str
    model_name: str
    version: str
    holiday_country: str
    val_days: int
    top_k_skus: int
    include_categories: bool
    include_total: bool
    target_transform: str
    lags: Tuple[int, ...]
    rolling_windows: Tuple[int, ...]
    random_state: int


def _detect_encoding(path: Path) -> str:
    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252", "iso-8859-1"]
    with open(path, "rb") as f:
        sample = f.read(4_000_000)
    for enc in encodings:
        try:
            sample.decode(enc)
            return enc
        except Exception:
            continue
    return "latin-1"


def _is_holiday_series(dates: pd.Series, country: str) -> pd.Series:
    code = (country or "").strip().upper()
    if not code:
        return pd.Series(0, index=dates.index, dtype="int64")

    years = sorted({int(d.year) for d in pd.to_datetime(dates, errors="coerce").dropna().dt.to_pydatetime()})
    cal = holidays.country_holidays(code, years=years)
    s = pd.to_datetime(dates, errors="coerce").dt.date
    return s.apply(lambda x: int(x in cal) if x else 0).astype("int64")


def _assign_category(desc_upper: pd.Series) -> pd.Series:
    """
    Assign a category using the same keyword rules as the API.

    Priority order follows SALES_HIERARCHY_RULES (excluding "Other", which is the fallback).
    """
    out = pd.Series("Other", index=desc_upper.index, dtype="object")
    matched = pd.Series(False, index=desc_upper.index)

    for r in SALES_HIERARCHY_RULES:
        cat = str(r.get("category") or "")
        if _sales_hierarchy_norm(cat) == "other":
            continue
        kws = list(r.get("keywords") or [])
        pat = _sales_hierarchy_build_keyword_regex(kws)
        if not pat:
            continue
        m = desc_upper.str.contains(pat, regex=True, na=False) & (~matched)
        if bool(m.any()):
            out.loc[m] = cat
            matched.loc[m] = True

    # Remaining rows stay as "Other".
    return out


def _make_daily_entity_frame(csv_path: Path, encoding: str, cfg: TrainConfig) -> pd.DataFrame:
    # Read once; dataset is small enough for memory (~280k rows).
    usecols = ["StockCode", "InvoiceDate", "Quantity", "UnitPrice", "Description"]
    df = pd.read_csv(csv_path, usecols=usecols, encoding=encoding, low_memory=False)

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df.dropna(subset=["StockCode", "InvoiceDate"])

    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce").fillna(0.0).astype(float)
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce").fillna(0.0).astype(float)
    df["revenue"] = (df["Quantity"] * df["UnitPrice"]).astype(float).clip(lower=0.0)

    df["StockCode"] = df["StockCode"].astype(str).str.strip()
    df["date"] = df["InvoiceDate"].dt.floor("D")

    parts: List[pd.DataFrame] = []

    # SKU entities
    daily_sku = (
        df.groupby(["StockCode", "date"], as_index=False)
        .agg(y=("revenue", "sum"))
        .sort_values(["StockCode", "date"])
    )
    if cfg.top_k_skus and cfg.top_k_skus > 0:
        top = (
            daily_sku.groupby("StockCode")["y"]
            .sum()
            .sort_values(ascending=False)
            .head(int(cfg.top_k_skus))
            .index
        )
        daily_sku = daily_sku[daily_sku["StockCode"].isin(top)].copy()

    daily_sku["entity"] = "SKU:" + daily_sku["StockCode"].astype(str)
    parts.append(daily_sku[["entity", "date", "y"]])

    # Category entities (based on Description keywords)
    if cfg.include_categories:
        desc_upper = df["Description"].astype(str).str.upper()
        df["category"] = _assign_category(desc_upper)
        daily_cat = (
            df.groupby(["category", "date"], as_index=False)
            .agg(y=("revenue", "sum"))
            .sort_values(["category", "date"])
        )
        daily_cat["entity"] = "CAT:" + daily_cat["category"].astype(str)
        parts.append(daily_cat[["entity", "date", "y"]])

    # Total entity
    if cfg.include_total:
        daily_all = df.groupby("date", as_index=False).agg(y=("revenue", "sum")).sort_values("date")
        daily_all["entity"] = "ALL"
        parts.append(daily_all[["entity", "date", "y"]])

    out = pd.concat(parts, ignore_index=True)
    out["entity"] = out["entity"].astype(str)
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.floor("D")
    out["y"] = pd.to_numeric(out["y"], errors="coerce").fillna(0.0).astype(float)
    out = out.dropna(subset=["date"])
    return out.sort_values(["entity", "date"]).reset_index(drop=True)


def _to_dense_daily_grid(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        raise RuntimeError("No daily revenue data after preprocessing.")

    min_date = pd.Timestamp(daily["date"].min()).floor("D")
    max_date = pd.Timestamp(daily["date"].max()).floor("D")
    date_index = pd.date_range(min_date, max_date, freq="D")
    entities = sorted(daily["entity"].astype(str).unique().tolist())

    mi = pd.MultiIndex.from_product([entities, date_index], names=["entity", "date"])
    dense = (
        daily.set_index(["entity", "date"])
        .reindex(mi, fill_value=0.0)
        .reset_index()
        .rename(columns={"y": "y"})
    )
    dense["y"] = pd.to_numeric(dense["y"], errors="coerce").fillna(0.0).astype(float)
    return dense


def _build_features(df: pd.DataFrame, cfg: TrainConfig) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    df = df.sort_values(["entity", "date"]).reset_index(drop=True)

    ts = pd.to_datetime(df["date"], errors="coerce")
    df["month"] = ts.dt.month.astype("int64")
    df["day_of_week"] = ts.dt.dayofweek.astype("int64")
    df["is_weekend"] = (ts.dt.dayofweek >= 5).astype("int64")
    df["day_of_year"] = ts.dt.dayofyear.astype("int64")
    df["is_holiday"] = _is_holiday_series(ts, cfg.holiday_country)

    grp = df.groupby("entity")["y"]
    for lag in cfg.lags:
        df[f"lag_{lag}"] = grp.shift(int(lag)).fillna(0.0).astype(float)

    for w in cfg.rolling_windows:
        df[f"rolling_mean_{w}"] = grp.transform(lambda s: s.shift(1).rolling(int(w)).mean()).fillna(0.0).astype(float)
        df[f"rolling_std_{w}"] = grp.transform(lambda s: s.shift(1).rolling(int(w)).std(ddof=0)).fillna(0.0).astype(float)

    df["mom_7"] = ((df["lag_1"] - df["lag_7"]) / (df["lag_1"].abs() + df["lag_7"].abs() + 1e-6)).astype(float)

    feature_cols: List[str] = [
        "month",
        "day_of_week",
        "is_weekend",
        "day_of_year",
        "is_holiday",
        *[f"lag_{l}" for l in cfg.lags],
        *[f"rolling_mean_{w}" for w in cfg.rolling_windows],
        *[f"rolling_std_{w}" for w in cfg.rolling_windows],
        "mom_7",
    ]

    X = df[feature_cols].copy()
    y = df["y"].copy()
    entity = df["entity"].copy()
    return X, y, entity


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true > 1e-6
    if not bool(mask.any()):
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train hierarchy-friendly global revenue forecasting model (LightGBM)")
    parser.add_argument("--csv-path", type=str, default="", help="Path to sales_data.csv")
    parser.add_argument("--model-name", type=str, default="sales_forecast", help="Model name (folder under models/)")
    parser.add_argument("--version", type=str, default="v1.1.0", help="Version folder name (e.g. v1.1.0)")
    parser.add_argument("--holiday-country", type=str, default="GB", help="holidays country code (e.g. GB)")
    parser.add_argument("--val-days", type=int, default=30, help="Validation window size (days)")
    parser.add_argument("--top-k-skus", type=int, default=0, help="Use only top-K SKUs for SKU entities (0 = all)")
    parser.add_argument("--include-categories", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-total", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--target-transform", type=str, default="log1p", choices=["none", "log1p"])
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    repo_root = project_root.parent

    csv_path = Path(args.csv_path) if args.csv_path else (repo_root / "raw_data" / "sales_data.csv")
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    cfg = TrainConfig(
        csv_path=str(csv_path),
        model_name=str(args.model_name),
        version=str(args.version),
        holiday_country=str(args.holiday_country),
        val_days=int(args.val_days),
        top_k_skus=int(args.top_k_skus),
        include_categories=bool(args.include_categories),
        include_total=bool(args.include_total),
        target_transform=str(args.target_transform),
        lags=(1, 7, 14, 28),
        rolling_windows=(7, 14, 28),
        random_state=int(args.random_state),
    )

    print("Loading data...")
    encoding = _detect_encoding(csv_path)
    daily = _make_daily_entity_frame(csv_path, encoding=encoding, cfg=cfg)
    dense = _to_dense_daily_grid(daily)

    print("Building features...")
    X, y_raw, entity = _build_features(dense, cfg)
    dates = pd.to_datetime(dense["date"]).reset_index(drop=True)

    max_date = pd.Timestamp(dates.max()).floor("D")
    val_start = max_date - pd.Timedelta(days=int(cfg.val_days) - 1)
    is_val = dates >= val_start

    X_train, X_val = X[~is_val], X[is_val]
    y_train_raw, y_val_raw = y_raw[~is_val].values, y_raw[is_val].values
    ent_train = np.asarray(entity[~is_val].astype(str).values, dtype="U")
    ent_val = np.asarray(entity[is_val].astype(str).values, dtype="U")

    # Sample weights: up-weight category/total rows so the model learns large-scale behavior too.
    w_train = np.ones(len(X_train), dtype=float)
    w_val = np.ones(len(X_val), dtype=float)
    w_train[np.char.startswith(ent_train, "CAT:")] = 50.0
    w_val[np.char.startswith(ent_val, "CAT:")] = 50.0
    w_train[ent_train == "ALL"] = 100.0
    w_val[ent_val == "ALL"] = 100.0

    def _transform(y: np.ndarray) -> np.ndarray:
        if cfg.target_transform == "log1p":
            return np.log1p(np.maximum(0.0, y))
        return y

    y_train = _transform(y_train_raw)
    y_val = _transform(y_val_raw)

    print(
        f"Train rows: {len(X_train):,} | Val rows: {len(X_val):,} | val_start: {val_start.date()} | max: {max_date.date()}"
    )

    model = lgb.LGBMRegressor(
        n_estimators=4000,
        learning_rate=0.03,
        num_leaves=128,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.0,
        reg_lambda=0.0,
        random_state=cfg.random_state,
        n_jobs=-1,
    )

    print("Training...")
    model.fit(
        X_train,
        y_train,
        sample_weight=w_train,
        eval_set=[(X_val, y_val)],
        eval_sample_weight=[w_val],
        eval_metric="l1",
        callbacks=[lgb.early_stopping(stopping_rounds=150, verbose=False)],
    )

    print("Evaluating...")
    pred_val = model.predict(X_val)
    if cfg.target_transform == "log1p":
        pred_val = np.expm1(pred_val)
    pred_val = np.maximum(0.0, pred_val)

    mae = float(np.mean(np.abs(y_val_raw - pred_val)))
    mape = _mape(y_val_raw, pred_val)
    print(f"MAE: {mae:.4f} | MAPE: {mape:.4f}%")

    out_dir = repo_root / "models" / cfg.model_name / cfg.version
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "model.pkl"
    meta_path = out_dir / "metadata.json"

    wrapped = SalesForecastWrapper(model, target_transform=cfg.target_transform)

    print(f"Saving model: {model_path}")
    joblib.dump(wrapped, model_path)

    entities_all = dense["entity"].astype(str)
    entities_unique = pd.Series(entities_all.unique())
    unique_skus = int(entities_unique[entities_unique.str.startswith("SKU:")].nunique())
    unique_cats = int(entities_unique[entities_unique.str.startswith("CAT:")].nunique())

    meta: Dict[str, object] = {
        "name": cfg.model_name,
        "version": cfg.version,
        "task": "sales_forecast",
        "target_metric": "revenue",
        "framework": "lightgbm",
        "target_transform": cfg.target_transform,
        "trained_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "dataset": {
            "csv_path": str(csv_path),
            "encoding": encoding,
            "date_min": str(pd.Timestamp(dates.min()).date()),
            "date_max": str(pd.Timestamp(dates.max()).date()),
            "unique_entities": int(entities_all.nunique()),
            "unique_skus": unique_skus,
            "unique_categories": unique_cats,
        },
        "feature_names": list(getattr(model, "feature_names_in_", X.columns)),
        "training": {
            **asdict(cfg),
            "sample_weights": {"CAT:*": 50.0, "ALL": 100.0},
        },
        "metrics": {"mae": mae, "mape": mape},
    }

    print(f"Saving metadata: {meta_path}")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=True, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
