#!/usr/bin/env python3
"""
Train a global sales forecasting model (revenue) using LightGBM.

Input: raw_data/sales_data.csv (Online Retail)
Output:
  models/{model_name}/{version}/model.pkl
  models/{model_name}/{version}/metadata.json

The model is trained as a 1-day-ahead regressor:
  y[t] = revenue[t]
  X[t] = calendar features of day t + lag/rolling features computed from revenue up to t-1

This matches the online inference feature generation in `prediction_app/api/datasets.py`.
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
    top_k_skus: int
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


def _make_daily_revenue_frame(csv_path: Path, encoding: str, top_k_skus: int) -> pd.DataFrame:
    usecols = ["StockCode", "InvoiceDate", "Quantity", "UnitPrice"]
    df = pd.read_csv(csv_path, usecols=usecols, encoding=encoding, low_memory=False)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df.dropna(subset=["StockCode", "InvoiceDate"])

    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce").fillna(0.0).astype(float)
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce").fillna(0.0).astype(float)
    df["revenue"] = (df["Quantity"] * df["UnitPrice"]).astype(float)
    # Sales can't be negative; clip returns to 0 to match online post-processing.
    df["revenue"] = df["revenue"].clip(lower=0.0)

    df["StockCode"] = df["StockCode"].astype(str).str.strip()
    df["date"] = df["InvoiceDate"].dt.floor("D")

    daily = (
        df.groupby(["StockCode", "date"], as_index=False)
        .agg(revenue=("revenue", "sum"))
        .sort_values(["StockCode", "date"])
    )

    if top_k_skus and top_k_skus > 0:
        top = (
            daily.groupby("StockCode")["revenue"]
            .sum()
            .sort_values(ascending=False)
            .head(int(top_k_skus))
            .index
        )
        daily = daily[daily["StockCode"].isin(top)].copy()

    return daily


def _to_dense_daily_grid(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        raise RuntimeError("No daily revenue data after preprocessing.")

    min_date = pd.Timestamp(daily["date"].min()).floor("D")
    max_date = pd.Timestamp(daily["date"].max()).floor("D")
    date_index = pd.date_range(min_date, max_date, freq="D")
    stock_codes = sorted(daily["StockCode"].astype(str).unique().tolist())

    mi = pd.MultiIndex.from_product([stock_codes, date_index], names=["StockCode", "date"])
    dense = (
        daily.set_index(["StockCode", "date"])
        .reindex(mi, fill_value=0.0)
        .reset_index()
        .rename(columns={"revenue": "y"})
    )
    dense["y"] = pd.to_numeric(dense["y"], errors="coerce").fillna(0.0).astype(float)
    return dense


def _build_features(df: pd.DataFrame, cfg: TrainConfig) -> Tuple[pd.DataFrame, pd.Series]:
    df = df.sort_values(["StockCode", "date"]).reset_index(drop=True)

    # Calendar features for the target day (date itself).
    ts = pd.to_datetime(df["date"], errors="coerce")
    df["month"] = ts.dt.month.astype("int64")
    df["day_of_week"] = ts.dt.dayofweek.astype("int64")
    df["is_weekend"] = (ts.dt.dayofweek >= 5).astype("int64")
    df["day_of_year"] = ts.dt.dayofyear.astype("int64")
    df["is_holiday"] = _is_holiday_series(ts, cfg.holiday_country)

    # Lag and rolling features computed only from history up to t-1.
    grp = df.groupby("StockCode")["y"]
    for lag in cfg.lags:
        df[f"lag_{lag}"] = grp.shift(int(lag)).fillna(0.0).astype(float)

    for w in cfg.rolling_windows:
        # shift(1) ensures we never use y[t] when predicting y[t].
        df[f"rolling_mean_{w}"] = grp.transform(lambda s: s.shift(1).rolling(int(w)).mean()).fillna(0.0).astype(float)
        df[f"rolling_std_{w}"] = grp.transform(lambda s: s.shift(1).rolling(int(w)).std(ddof=0)).fillna(0.0).astype(float)

    df["mom_7"] = ((df["lag_1"] - df["lag_7"]) / (df["lag_7"].abs() + 1e-6)).astype(float)

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


def main():
    parser = argparse.ArgumentParser(description="Train global revenue forecasting model (LightGBM)")
    parser.add_argument("--csv-path", type=str, default="", help="Path to sales_data.csv")
    parser.add_argument("--model-name", type=str, default="sales_forecast", help="Model name (folder under models/)")
    parser.add_argument("--version", type=str, default="v1.0.0", help="Version folder name (e.g. v1.0.0)")
    parser.add_argument("--holiday-country", type=str, default="GB", help="holidays country code (e.g. GB)")
    parser.add_argument("--val-days", type=int, default=30, help="Validation window size (days)")
    parser.add_argument("--top-k-skus", type=int, default=0, help="Use only top-K SKUs by revenue (0 = all)")
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
        lags=(1, 7, 14, 28),
        rolling_windows=(7, 14, 28),
        random_state=int(args.random_state),
    )

    print("📥 Loading data...")
    encoding = _detect_encoding(csv_path)
    daily = _make_daily_revenue_frame(csv_path, encoding=encoding, top_k_skus=cfg.top_k_skus)
    dense = _to_dense_daily_grid(daily)

    print("🧱 Building features...")
    X, y = _build_features(dense, cfg)
    dates = pd.to_datetime(dense["date"]).reset_index(drop=True)

    max_date = pd.Timestamp(dates.max()).floor("D")
    val_start = max_date - pd.Timedelta(days=int(cfg.val_days) - 1)
    is_val = dates >= val_start
    X_train, y_train = X[~is_val], y[~is_val]
    X_val, y_val = X[is_val], y[is_val]

    print(f"✅ Train rows: {len(X_train):,} | Val rows: {len(X_val):,} | val_start: {val_start.date()} | max: {max_date.date()}")

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

    print("🏋️  Training...")
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="l1",
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
    )

    print("📈 Evaluating...")
    pred_val = model.predict(X_val)
    pred_val = np.maximum(0.0, pred_val)

    mae = float(np.mean(np.abs(y_val.values - pred_val)))
    mape = _mape(y_val.values, pred_val)
    print(f"MAE: {mae:.4f} | MAPE: {mape:.4f}%")

    out_dir = repo_root / "models" / cfg.model_name / cfg.version
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "model.pkl"
    meta_path = out_dir / "metadata.json"

    print(f"💾 Saving model: {model_path}")
    joblib.dump(model, model_path)

    meta: Dict[str, object] = {
        "name": cfg.model_name,
        "version": cfg.version,
        "task": "sales_forecast",
        "target_metric": "revenue",
        "framework": "lightgbm",
        "trained_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "dataset": {
            "csv_path": str(csv_path),
            "encoding": encoding,
            "rows_raw": int(len(daily)),
            "rows_dense": int(len(dense)),
            "date_min": str(pd.Timestamp(dates.min()).date()),
            "date_max": str(pd.Timestamp(dates.max()).date()),
            "unique_stock_codes": int(daily["StockCode"].nunique()),
        },
        "feature_names": list(getattr(model, "feature_names_in_", X.columns)),
        "training": {
            **asdict(cfg),
        },
        "metrics": {
            "mae": mae,
            "mape": mape,
        },
    }

    print(f"🧾 Saving metadata: {meta_path}")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("✅ Done.")


if __name__ == "__main__":
    main()

