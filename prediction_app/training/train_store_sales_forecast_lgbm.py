#!/usr/bin/env python3
"""
Train a lightweight demand forecasting model for the stock-allocation demo (store_sales_data.csv).

Dataset: raw_data/store_sales_data.csv (Superstore-like)
Target: daily units (order-line count) per Sub-Category.

Why Sub-Category?
- Product-level time series in this dataset are extremely sparse.
- Sub-Category provides enough signal for a stable demo forecast.
- The allocation service downscales Sub-Category forecast to a selected product via recent share.

Output:
  models/store_sales_forecast/v1.0.0/model.pkl
  models/store_sales_forecast/v1.0.0/metadata.json

The inference feature schema MUST match:
  prediction_app/api/datasets.py -> _stock_alloc_feature_row()
"""

import argparse
import json
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


@dataclass
class TrainConfig:
    csv_path: str
    model_name: str
    version: str
    holiday_country: str
    val_days: int
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


def _parse_order_date(series: pd.Series) -> pd.Series:
    # The dataset uses day-first dates (e.g., 08/11/2017 = 8 Nov 2017).
    dt = pd.to_datetime(series, errors="coerce", dayfirst=True)
    # Fallback: try month-first if parsing fails.
    if dt.isna().mean() > 0.5:
        dt2 = pd.to_datetime(series, errors="coerce", dayfirst=False)
        if dt2.notna().sum() > dt.notna().sum():
            dt = dt2
    return dt.dt.floor("D")


def _is_holiday_series(dates: pd.Series, country: str) -> pd.Series:
    code = (country or "").strip().upper()
    if not code:
        return pd.Series(0, index=dates.index, dtype="int64")

    years = sorted({int(d.year) for d in pd.to_datetime(dates, errors="coerce").dropna().dt.to_pydatetime()})
    cal = holidays.country_holidays(code, years=years)
    s = pd.to_datetime(dates, errors="coerce").dt.date
    return s.apply(lambda x: int(x in cal) if x else 0).astype("int64")


def _make_daily_units_frame(csv_path: Path, encoding: str) -> pd.DataFrame:
    usecols = ["Order Date", "Sub-Category"]
    df = pd.read_csv(csv_path, usecols=usecols, encoding=encoding, low_memory=False)
    df["Order Date"] = _parse_order_date(df["Order Date"])
    df = df.dropna(subset=["Order Date", "Sub-Category"])
    df["Sub-Category"] = df["Sub-Category"].astype(str).str.strip()
    df = df[df["Sub-Category"] != ""].copy()

    df["date"] = df["Order Date"]
    daily = (
        df.groupby(["Sub-Category", "date"], as_index=False)
        .size()
        .rename(columns={"size": "y"})
        .sort_values(["Sub-Category", "date"])
    )
    daily["y"] = pd.to_numeric(daily["y"], errors="coerce").fillna(0.0).astype(float)
    return daily


def _to_dense_daily_grid(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        raise RuntimeError("No daily units after preprocessing.")

    min_date = pd.Timestamp(daily["date"].min()).floor("D")
    max_date = pd.Timestamp(daily["date"].max()).floor("D")
    date_index = pd.date_range(min_date, max_date, freq="D")
    keys = sorted(daily["Sub-Category"].astype(str).unique().tolist())

    mi = pd.MultiIndex.from_product([keys, date_index], names=["Sub-Category", "date"])
    dense = (
        daily.set_index(["Sub-Category", "date"])
        .reindex(mi, fill_value=0.0)
        .reset_index()
    )
    dense["y"] = pd.to_numeric(dense["y"], errors="coerce").fillna(0.0).astype(float)
    return dense


def _build_features(df: pd.DataFrame, cfg: TrainConfig) -> tuple[pd.DataFrame, pd.Series]:
    df = df.sort_values(["Sub-Category", "date"]).reset_index(drop=True)

    ts = pd.to_datetime(df["date"], errors="coerce")
    df["month"] = ts.dt.month.astype("int64")
    df["day_of_week"] = ts.dt.dayofweek.astype("int64")
    df["is_weekend"] = (ts.dt.dayofweek >= 5).astype("int64")
    df["day_of_year"] = ts.dt.dayofyear.astype("int64")
    df["is_holiday"] = _is_holiday_series(ts, cfg.holiday_country)

    grp = df.groupby("Sub-Category")["y"]
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
    return X, y


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true > 1e-6
    if not bool(mask.any()):
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train store_sales demand forecast model (LightGBM)")
    parser.add_argument("--csv-path", type=str, default="", help="Path to store_sales_data.csv")
    parser.add_argument("--model-name", type=str, default="store_sales_forecast", help="Model name folder under models/")
    parser.add_argument("--version", type=str, default="v1.0.0", help="Version folder name (e.g. v1.0.0)")
    parser.add_argument("--holiday-country", type=str, default="US", help="holidays country code (e.g. US)")
    parser.add_argument("--val-days", type=int, default=60, help="Validation window size (days)")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    repo_root = project_root.parent

    csv_path = Path(args.csv_path) if args.csv_path else (repo_root / "raw_data" / "store_sales_data.csv")
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    cfg = TrainConfig(
        csv_path=str(csv_path),
        model_name=str(args.model_name),
        version=str(args.version),
        holiday_country=str(args.holiday_country),
        val_days=int(args.val_days),
        lags=(1, 7, 14, 28),
        rolling_windows=(7, 14, 28),
        random_state=int(args.random_state),
    )

    print("Loading data...")
    encoding = _detect_encoding(csv_path)
    daily = _make_daily_units_frame(csv_path, encoding=encoding)
    dense = _to_dense_daily_grid(daily)

    print("Building features...")
    X, y = _build_features(dense, cfg)
    dates = pd.to_datetime(dense["date"]).reset_index(drop=True)

    max_date = pd.Timestamp(dates.max()).floor("D")
    val_start = max_date - pd.Timedelta(days=int(cfg.val_days) - 1)
    is_val = dates >= val_start
    X_train, y_train = X[~is_val], y[~is_val]
    X_val, y_val = X[is_val], y[is_val]

    print(
        f"Train rows: {len(X_train):,} | Val rows: {len(X_val):,} | "
        f"val_start: {val_start.date()} | max: {max_date.date()}"
    )

    model = lgb.LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=64,
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
        eval_set=[(X_val, y_val)],
        eval_metric="l1",
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
    )

    print("Evaluating...")
    pred_val = model.predict(X_val)
    pred_val = np.maximum(0.0, pred_val)
    mae = float(np.mean(np.abs(y_val.values - pred_val)))
    mape = _mape(y_val.values, pred_val)
    print(f"MAE: {mae:.4f} | MAPE: {mape:.4f}%")

    out_dir = repo_root / "models" / cfg.model_name / cfg.version
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "model.pkl"
    meta_path = out_dir / "metadata.json"

    print(f"Saving model: {model_path}")
    joblib.dump(model, model_path)

    meta: Dict[str, object] = {
        "name": cfg.model_name,
        "version": cfg.version,
        "task": "stock_allocation_forecast",
        "target_metric": "daily_units",
        "framework": "lightgbm",
        "trained_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "dataset": {
            "csv_path": str(csv_path),
            "encoding": encoding,
            "rows_raw": int(len(daily)),
            "rows_dense": int(len(dense)),
            "date_min": str(pd.Timestamp(dates.min()).date()),
            "date_max": str(pd.Timestamp(dates.max()).date()),
            "unique_sub_categories": int(daily["Sub-Category"].nunique()),
        },
        "feature_names": list(getattr(model, "feature_names_in_", X.columns)),
        "training": {**asdict(cfg)},
        "metrics": {"mae": mae, "mape": mape},
    }

    print(f"Saving metadata: {meta_path}")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=True, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()

