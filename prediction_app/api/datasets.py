"""
Dataset Management API
Provides dataset list/detail/preview/stats endpoints.
"""
import os
import sys
import json
import math
import hashlib
from datetime import datetime, timedelta
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Set
from urllib import request as urlrequest
from urllib import error as urlerror
import re
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
import joblib
import psycopg2
from psycopg2 import sql
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

# Add project root to sys.path so we can import shared modules.
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables (optional)
env_path = project_root / ".env"
if env_path.exists():
    load_dotenv(env_path)
    print(f"Loaded env file: {env_path}")
else:
    print(f"Env file not found (optional): {env_path}")

router = APIRouter(prefix="/api/v1/datasets", tags=["datasets"])

RFM_CACHE_MAX_ITEMS = 8
# analysis_id -> {"created_at": datetime, "dataset_id": str, "params": dict, "df": pd.DataFrame}
RFM_ANALYSIS_CACHE: Dict[str, Dict[str, Any]] = {}
RFM_DISK_CACHE_MAX_ITEMS = int(os.getenv("RFM_DISK_CACHE_MAX_ITEMS", "32"))
RFM_DISK_CACHE_DIR = Path(os.getenv("RFM_DISK_CACHE_DIR", "/tmp/fina_demo_rfm_cache")).resolve()

SALES_FORECAST_MODEL_CACHE_MAX_ITEMS = 8
# model_path -> {"created_at": datetime, "model": Any}
SALES_FORECAST_MODEL_CACHE: Dict[str, Dict[str, Any]] = {}
SALES_FORECAST_MODEL_CACHE_LOCK = threading.Lock()
# (dataset_id, sales_metric) -> mean_sales (float)
SALES_FORECAST_FALLBACK_MEAN_CACHE: Dict[str, float] = {}
SALES_FORECAST_FALLBACK_MEAN_LOCK = threading.Lock()
SALES_FORECAST_HOLIDAY_CACHE: Dict[str, Any] = {}
SALES_FORECAST_HOLIDAY_LOCK = threading.Lock()


def get_db_connection():
    """Create a new DB connection per request (avoids transaction conflicts)."""
    try:
        # Fail fast when DB is unreachable; otherwise frontend requests can hang indefinitely.
        connect_timeout = int(os.getenv("DB_CONNECT_TIMEOUT", "5"))
        statement_timeout_ms = int(os.getenv("DB_STATEMENT_TIMEOUT_MS", "30000"))

        kwargs: Dict[str, Any] = {
            "host": os.getenv("DB_HOST"),
            "port": int(os.getenv("DB_PORT", 5432)),
            "database": os.getenv("DB_NAME"),
            "user": os.getenv("DB_USER"),
            "password": os.getenv("DB_PASSWORD"),
        }
        if connect_timeout > 0:
            kwargs["connect_timeout"] = connect_timeout
        if statement_timeout_ms > 0:
            # Applies per statement on the server side; value is in milliseconds.
            kwargs["options"] = f"-c statement_timeout={statement_timeout_ms}"

        conn = psycopg2.connect(**kwargs)
        return conn
    except Exception as e:
        print(f"Database connection failed: {e}")
        raise


def load_datasets_config() -> Dict[str, Any]:
    """Load datasets config from prediction_app/config/datasets.json."""
    config_path = project_root / "config" / "datasets.json"
    
    # Try a few possible paths (keep compatible with older layouts).
    possible_paths = [
        config_path,
        Path(project_root) / "config" / "datasets.json",
        Path(__file__).parent.parent / "config" / "datasets.json",
    ]
    
    for path in possible_paths:
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                    print(f"Loaded datasets config: {path}")
                    return config
            except Exception as e:
                print(f"Failed to load config {path}: {e}")
    
    print("Datasets config not found; returning empty config.")
    return {"datasets": []}


def get_table_row_count(table_name: str) -> int:
    """获取表的记录数"""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(f'SELECT COUNT(*) as count FROM "{table_name}"')
            result = cur.fetchone()
            return result[0] if result else 0
    finally:
        conn.close()


def get_table_columns(table_name: str) -> List[Dict[str, str]]:
    """获取表的列信息"""
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT 
                    column_name,
                    data_type
                FROM information_schema.columns
                WHERE table_name = %s
                ORDER BY ordinal_position
                """,
                (table_name,),
            )
            return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


def get_time_range(table_name: str, time_column: str) -> Optional[Dict[str, Optional[str]]]:
    """获取时间范围"""
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            try:
                cur.execute(
                    sql.SQL("SELECT MIN({col}) as min, MAX({col}) as max FROM {table}").format(
                        col=sql.Identifier(time_column),
                        table=sql.Identifier(table_name),
                    )
                )
                result = cur.fetchone()
                if result:
                    return {
                        "min": str(result["min"]) if result["min"] else None,
                        "max": str(result["max"]) if result["max"] else None,
                    }
            except Exception as e:
                print(f"⚠️  获取时间范围失败: {e}")
                return None
        return None
    finally:
        conn.close()


def get_column_stats(table_name: str, column_name: str, data_type: str) -> Dict[str, Any]:
    """获取列的统计信息"""
    conn = get_db_connection()
    stats = {
        "columnName": column_name,
        "dataType": data_type,
        "nullCount": 0,
    }
    
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # 获取空值数量
            cur.execute(
                f'SELECT COUNT(*) as count FROM "{table_name}" WHERE "{column_name}" IS NULL'
            )
            result = cur.fetchone()
            stats["nullCount"] = result["count"] if result else 0
            
            # 根据数据类型计算不同的统计信息
            if "char" in data_type.lower() or "text" in data_type.lower() or data_type == "varchar" or data_type == "character varying":
                # 字符串类型：计算唯一值数量
                cur.execute(
                    f'SELECT COUNT(DISTINCT "{column_name}") as count FROM "{table_name}" WHERE "{column_name}" IS NOT NULL'
                )
                result = cur.fetchone()
                stats["uniqueCount"] = result["count"] if result else 0
            elif (
                "int" in data_type.lower()
                or "numeric" in data_type.lower()
                or "decimal" in data_type.lower()
                or "real" in data_type.lower()
                or "double" in data_type.lower()
                or data_type == "float"
            ):
                # 数值类型：计算分布统计
                cur.execute(
                    f"""
                    SELECT 
                        MIN("{column_name}")::text as min,
                        MAX("{column_name}")::text as max,
                        AVG("{column_name}")::text as mean,
                        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY "{column_name}")::text as median,
                        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY "{column_name}")::text as q25,
                        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY "{column_name}")::text as q75
                    FROM "{table_name}"
                    WHERE "{column_name}" IS NOT NULL
                    """
                )
                result = cur.fetchone()
                if result and result["min"]:
                    stats["distribution"] = {
                        "min": float(result["min"]),
                        "max": float(result["max"]),
                        "mean": float(result["mean"]),
                        "median": float(result["median"]),
                        "quartiles": [
                            float(result["q25"]),
                            float(result["median"]),
                            float(result["q75"]),
                        ],
                    }
    finally:
        conn.close()
    
    return stats


class RFMWeights(BaseModel):
    r: float = Field(1.0, gt=0)
    f: float = Field(1.0, gt=0)
    m: float = Field(1.0, gt=0)


class RFMRunRequest(BaseModel):
    time_window_days: int = Field(365, ge=1, le=3650)
    scoring_scale: int = Field(5, ge=3, le=10)
    segmentation_method: str = Field("quantiles")  # "quantiles" | "kmeans"
    weights: RFMWeights = Field(default_factory=RFMWeights)


class CustomerSegmentationRequest(BaseModel):
    """
    Customer Segmentation (K-Means).

    Default mode clusters customers by product-category mix:
    - Clean product descriptions into a keyword-based `category`
    - Build per-customer category revenue-share features
    - Run K-Means with automatic K selection
    """

    time_window_days: int = Field(365, ge=1, le=3650)
    # Backward compatible. When feature_mode="rfm", these are the allowed inputs.
    selected_features: List[str] = Field(default_factory=lambda: ["recency_days", "frequency", "monetary"])
    # "category_mix" (default) | "rfm"
    feature_mode: str = Field("category_mix")
    k_range: Tuple[int, int] = Field((3, 6))
    random_seed: int = Field(42)
    outlier_threshold: Optional[float] = Field(3.0)
    # When None: enable by default if an API key is available.
    enable_ai_insight: Optional[bool] = Field(None)


def _rfm_make_analysis_id(dataset_id: str, req: RFMRunRequest) -> str:
    payload = {
        "dataset_id": dataset_id,
        "time_window_days": req.time_window_days,
        "scoring_scale": req.scoring_scale,
        "segmentation_method": req.segmentation_method,
        "weights": req.weights.model_dump(),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def _rfm_cache_put(analysis_id: str, item: Dict[str, Any]) -> None:
    if analysis_id in RFM_ANALYSIS_CACHE:
        RFM_ANALYSIS_CACHE[analysis_id] = item
        return

    if len(RFM_ANALYSIS_CACHE) >= RFM_CACHE_MAX_ITEMS:
        # Drop the oldest analysis to avoid unbounded memory growth.
        oldest_id = min(
            RFM_ANALYSIS_CACHE.items(),
            key=lambda kv: kv[1].get("created_at", datetime.min),
        )[0]
        RFM_ANALYSIS_CACHE.pop(oldest_id, None)

    RFM_ANALYSIS_CACHE[analysis_id] = item

    # Best-effort disk persistence so segment drill-down still works after reload/restart.
    _rfm_disk_cache_write(analysis_id, item)


def _rfm_disk_cache_paths(analysis_id: str) -> tuple[Path, Path]:
    base = RFM_DISK_CACHE_DIR / analysis_id
    return base.with_suffix(".json"), base.with_suffix(".pkl")


def _rfm_disk_cache_prune(max_items: int) -> None:
    try:
        if max_items <= 0 or not RFM_DISK_CACHE_DIR.exists():
            return

        metas = sorted(RFM_DISK_CACHE_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if len(metas) <= max_items:
            return

        for meta_path in metas[max_items:]:
            df_path = meta_path.with_suffix(".pkl")
            meta_path.unlink(missing_ok=True)
            df_path.unlink(missing_ok=True)
    except Exception:
        # Ignore cleanup failures.
        return


def _rfm_disk_cache_write(analysis_id: str, item: Dict[str, Any]) -> None:
    try:
        RFM_DISK_CACHE_DIR.mkdir(parents=True, exist_ok=True)

        meta_path, df_path = _rfm_disk_cache_paths(analysis_id)
        df = item.get("df")
        if isinstance(df, pd.DataFrame):
            df.to_pickle(df_path)

        created_at = item.get("created_at")
        if isinstance(created_at, datetime):
            created_at_str = created_at.isoformat()
        else:
            created_at_str = datetime.utcnow().isoformat()

        meta = {
            "analysis_id": analysis_id,
            "created_at": created_at_str,
            "dataset_id": item.get("dataset_id"),
            "params": item.get("params"),
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

        _rfm_disk_cache_prune(RFM_DISK_CACHE_MAX_ITEMS)
    except Exception as e:
        print(f"⚠️  Failed to write RFM disk cache: {e}")


def _rfm_disk_cache_read(analysis_id: str) -> Optional[Dict[str, Any]]:
    meta_path, df_path = _rfm_disk_cache_paths(analysis_id)
    if not meta_path.exists() or not df_path.exists():
        return None

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8", errors="ignore") or "{}")
        df = pd.read_pickle(df_path)

        created_at: datetime
        raw_created = meta.get("created_at")
        if isinstance(raw_created, str):
            try:
                created_at = datetime.fromisoformat(raw_created)
            except Exception:
                created_at = datetime.utcfromtimestamp(meta_path.stat().st_mtime)
        else:
            created_at = datetime.utcfromtimestamp(meta_path.stat().st_mtime)

        return {
            "created_at": created_at,
            "dataset_id": meta.get("dataset_id"),
            "params": meta.get("params") or {},
            "df": df,
        }
    except Exception as e:
        print(f"⚠️  Failed to read RFM disk cache: {e}")
        return None


def _rfm_cache_get(analysis_id: str, dataset_id: str) -> Optional[Dict[str, Any]]:
    cached = RFM_ANALYSIS_CACHE.get(analysis_id)
    if cached and cached.get("dataset_id") == dataset_id:
        return cached

    disk = _rfm_disk_cache_read(analysis_id)
    if disk and disk.get("dataset_id") == dataset_id:
        # Rehydrate memory cache for faster subsequent reads.
        _rfm_cache_put(analysis_id, disk)
        return disk

    return None


def _resolve_csv_column(header_cols: List[str], requested: str) -> str:
    if requested in header_cols:
        return requested
    lower_map = {c.lower(): c for c in header_cols}
    hit = lower_map.get(requested.lower())
    if hit:
        return hit
    # Fuzzy match: ignore separators like "_" and match "invoice_date" -> "InvoiceDate".
    def _norm(s: str) -> str:
        return "".join(ch for ch in s.lower() if ch.isalnum())

    norm_map = {_norm(c): c for c in header_cols}
    hit = norm_map.get(_norm(requested))
    if hit:
        return hit
    raise HTTPException(status_code=400, detail=f"CSV missing column: {requested}")


def _read_csv_header(csv_path: Path) -> tuple[str, List[str]]:
    # IMPORTANT: header-only reads can succeed with utf-8 even when the body isn't utf-8.
    # Detect encoding by attempting to decode a small byte sample first.
    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252", "iso-8859-1"]
    sample: bytes
    try:
        with open(csv_path, "rb") as f:
            # Use a larger sample to catch non-utf8 bytes that may appear later in the file.
            sample = f.read(4_000_000)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {e}")

    last_err: Optional[Exception] = None
    for enc in encodings:
        try:
            sample.decode(enc)
            cols = pd.read_csv(csv_path, nrows=0, encoding=enc).columns.tolist()
            return enc, [str(c) for c in cols]
        except Exception as e:
            last_err = e
            continue
    raise HTTPException(status_code=400, detail=f"Failed to read CSV header: {last_err}")


def _apply_filters_df(df: pd.DataFrame, filters: List[Dict[str, Any]]) -> pd.DataFrame:
    allowed_ops = {">", ">=", "<", "<=", "=", "!=", "<>"}
    mask = pd.Series(True, index=df.index)

    for f in filters:
        col = f.get("column")
        op = f.get("op")
        val = f.get("value")
        if not isinstance(col, str) or not col:
            raise HTTPException(status_code=400, detail="RFM filters: invalid column")
        if op not in allowed_ops:
            raise HTTPException(status_code=400, detail=f"RFM filters: invalid op '{op}'")
        if col not in df.columns:
            raise HTTPException(status_code=400, detail=f"CSV filters refer to missing column: {col}")

        s = df[col]
        if isinstance(val, (int, float)):
            s = pd.to_numeric(s, errors="coerce")

        if op == ">":
            mask &= s > val
        elif op == ">=":
            mask &= s >= val
        elif op == "<":
            mask &= s < val
        elif op == "<=":
            mask &= s <= val
        elif op == "=":
            mask &= s == val
        elif op in {"!=", "<>"}:
            mask &= s != val

    return df[mask]


def _get_dataset_csv_path(dataset_id: str, dataset_config: Dict[str, Any]) -> Path:
    raw = dataset_config.get("csv_path")
    # Support a sensible default: repo_root/raw_data/{dataset_id}.csv
    repo_root = project_root.parent
    if isinstance(raw, str) and raw:
        p = Path(raw)
        if not p.is_absolute():
            p = (repo_root / p).resolve()
        return p
    return (repo_root / "raw_data" / f"{dataset_id}.csv").resolve()


def _csv_count_rows(csv_path: Path) -> int:
    """Count data rows in a CSV (excluding the header row).

    Notes:
    - We intentionally avoid DB dependencies for the demo.
    - This counts physical lines; it assumes no multi-line CSV fields (true for our demo datasets).
    """
    try:
        with open(csv_path, "rb") as f:
            # File iteration counts the last line even if it does not end with a newline.
            line_count = sum(1 for _ in f)
        return max(0, line_count - 1)
    except Exception:
        return 0


def _csv_infer_column_types(sample_df: pd.DataFrame, header_cols: List[str]) -> Dict[str, str]:
    """Infer a simple, UI-friendly type string per column from a sample dataframe."""
    types: Dict[str, str] = {}
    for c in header_cols:
        if c not in sample_df.columns:
            types[c] = "text"
            continue
        s = sample_df[c]
        try:
            if pd.api.types.is_datetime64_any_dtype(s.dtype):
                types[c] = "timestamp"
            elif pd.api.types.is_bool_dtype(s.dtype):
                types[c] = "boolean"
            elif pd.api.types.is_integer_dtype(s.dtype):
                types[c] = "int"
            elif pd.api.types.is_float_dtype(s.dtype) or pd.api.types.is_numeric_dtype(s.dtype):
                types[c] = "float"
            else:
                types[c] = "text"
        except Exception:
            types[c] = "text"
    return types


def _csv_time_range(csv_path: Path, *, encoding: str, time_col: str) -> Optional[Dict[str, Optional[str]]]:
    """Compute min/max timestamp for a CSV column without loading the full CSV into memory."""
    min_ts: Optional[pd.Timestamp] = None
    max_ts: Optional[pd.Timestamp] = None
    try:
        for chunk in pd.read_csv(
            csv_path,
            encoding=encoding,
            usecols=[time_col],
            chunksize=200_000,
            low_memory=False,
        ):
            s = pd.to_datetime(chunk[time_col], errors="coerce")
            if s.notna().any():
                cmin = s.min()
                cmax = s.max()
                if isinstance(cmin, pd.Timestamp) and not pd.isna(cmin):
                    min_ts = cmin if min_ts is None else min(min_ts, cmin)
                if isinstance(cmax, pd.Timestamp) and not pd.isna(cmax):
                    max_ts = cmax if max_ts is None else max(max_ts, cmax)
        return {
            "min": min_ts.isoformat() if min_ts is not None else None,
            "max": max_ts.isoformat() if max_ts is not None else None,
        }
    except Exception:
        return None


def _csv_column_stats(df: pd.DataFrame, column_name: str, data_type: str) -> Dict[str, Any]:
    """Compute basic column stats from an in-memory dataframe (CSV-first demo path)."""
    stats: Dict[str, Any] = {"columnName": column_name, "dataType": data_type, "nullCount": 0}
    if column_name not in df.columns:
        return stats

    s = df[column_name]
    stats["nullCount"] = int(s.isna().sum())

    # String-like columns: unique count
    if data_type in {"text", "string"} or pd.api.types.is_object_dtype(s.dtype):
        try:
            stats["uniqueCount"] = int(s.dropna().nunique())
        except Exception:
            pass
        return stats

    # Numeric columns: distribution
    try:
        num = pd.to_numeric(s, errors="coerce")
        num = num.dropna()
        if num.empty:
            return stats
        q25 = float(num.quantile(0.25))
        q50 = float(num.quantile(0.50))
        q75 = float(num.quantile(0.75))
        stats["distribution"] = {
            "min": float(num.min()),
            "max": float(num.max()),
            "mean": float(num.mean()),
            "median": q50,
            "quartiles": [q25, q50, q75],
        }
    except Exception:
        # Non-numeric columns can skip distribution.
        pass
    return stats


def _rfm_compute_amount(df: pd.DataFrame, amount_cfg: Dict[str, Any]) -> pd.Series:
    t = amount_cfg.get("type")
    if t == "column":
        col = amount_cfg.get("column")
        if not isinstance(col, str) or not col or col not in df.columns:
            raise HTTPException(status_code=400, detail="RFM monetary config: missing column")
        return pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(float)

    if t == "mul":
        left = amount_cfg.get("left")
        right = amount_cfg.get("right")
        if not isinstance(left, str) or not left or left not in df.columns:
            raise HTTPException(status_code=400, detail="RFM monetary config: missing left")
        if not isinstance(right, str) or not right or right not in df.columns:
            raise HTTPException(status_code=400, detail="RFM monetary config: missing right")

        l = pd.to_numeric(df[left], errors="coerce").fillna(0.0).astype(float)
        r = pd.to_numeric(df[right], errors="coerce").fillna(0.0).astype(float)
        return (l * r).astype(float)

    raise HTTPException(status_code=400, detail="RFM monetary config: unsupported type")


def _rfm_aggregate_users(
    df: pd.DataFrame,
    *,
    user_col: str,
    order_col: str,
    date_col: str,
    amount_cfg: Dict[str, Any],
    reference_date: pd.Timestamp,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["user_id", "recency_days", "frequency", "monetary"])

    df = df.copy()
    df["__amount"] = _rfm_compute_amount(df, amount_cfg)

    # Order-level aggregation first to ensure Frequency counts unique orders (not line items).
    orders = (
        df.groupby([user_col, order_col], dropna=False)
        .agg(last_order_date=(date_col, "max"), monetary=("__amount", "sum"))
        .reset_index()
    )

    users = (
        orders.groupby(user_col, dropna=False)
        .agg(
            last_order_date=("last_order_date", "max"),
            frequency=(order_col, "count"),
            monetary=("monetary", "sum"),
        )
        .reset_index()
        .rename(columns={user_col: "user_id"})
    )

    users["last_order_date"] = pd.to_datetime(users["last_order_date"], errors="coerce")
    users = users.dropna(subset=["user_id", "last_order_date"])
    if users.empty:
        return pd.DataFrame(columns=["user_id", "recency_days", "frequency", "monetary"])

    users["frequency"] = pd.to_numeric(users["frequency"], errors="coerce").fillna(0).astype("int64")
    users["monetary"] = pd.to_numeric(users["monetary"], errors="coerce").fillna(0.0).astype(float)
    users["recency_days"] = (reference_date - users["last_order_date"]).dt.days.astype("int64")
    return users[["user_id", "recency_days", "frequency", "monetary"]]


def _rfm_score_quantiles(series: pd.Series, scale: int, higher_is_better: bool) -> pd.Series:
    if series.empty:
        return pd.Series(dtype="int64")

    n = int(series.shape[0])
    bins = max(1, min(int(scale), n))
    ranked = series.rank(method="first")
    bin_idx = pd.qcut(ranked, q=bins, labels=False)

    if higher_is_better:
        return bin_idx.astype("int64") + 1

    # Recency: smaller is better.
    return bins - bin_idx.astype("int64")


def _kmeans_1d(values: np.ndarray, k: int, max_iter: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Simple 1D k-means (numpy-only) to avoid adding heavy runtime deps."""
    if values.ndim != 1:
        values = values.reshape(-1)

    n = int(values.shape[0])
    if n == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    k = max(1, min(int(k), n))
    if k == 1:
        return np.zeros(n, dtype=int), np.array([float(np.mean(values))], dtype=float)

    # Initialize with quantiles to get stable centers.
    qs = (np.arange(k) + 0.5) / k
    centers = np.quantile(values, qs).astype(float)
    # Ensure centers are not all identical.
    if np.allclose(centers, centers[0]):
        centers = np.linspace(float(values.min()), float(values.max()), k)

    for _ in range(max_iter):
        distances = np.abs(values[:, None] - centers[None, :])
        labels = np.argmin(distances, axis=1)
        new_centers = centers.copy()

        for j in range(k):
            mask = labels == j
            if np.any(mask):
                new_centers[j] = float(np.mean(values[mask]))
            else:
                # Re-seed empty cluster to a random point.
                new_centers[j] = float(values[np.random.randint(0, n)])

        if np.allclose(new_centers, centers):
            centers = new_centers
            break
        centers = new_centers

    return labels.astype(int), centers.astype(float)


def _rfm_score_kmeans(series: pd.Series, scale: int, higher_is_better: bool) -> pd.Series:
    if series.empty:
        return pd.Series(dtype="int64")

    x = series.to_numpy(dtype=float, copy=False)
    k = max(1, min(int(scale), int(x.shape[0])))
    if k == 1:
        return pd.Series(1, index=series.index, dtype="int64")

    labels, centers = _kmeans_1d(x, k=k)
    order = np.argsort(centers)  # ascending center -> low group
    rank_map = {int(cluster): int(rank + 1) for rank, cluster in enumerate(order)}
    ranks = np.vectorize(lambda lab: rank_map.get(int(lab), 1))(labels).astype(int)

    if higher_is_better:
        scores = ranks
    else:
        scores = (k - ranks + 1)

    return pd.Series(scores, index=series.index, dtype="int64")


def _rfm_score_level(score: int, *, scale: int) -> int:
    """Map a discrete score (1..scale) to 3 classic buckets: Low/Mid/High.

    We intentionally bucket by *score range* (not by customer quantiles) so:
    - the user changing `scoring_scale` is visible in the matrix boundaries
    - the 3x3 R×F matrix remains "classic" and stable to explain
    """

    s = int(score)
    sc = max(3, int(scale))
    s = max(1, min(s, sc))

    # For the common 1-5 scale: Low=1-2, Mid=3, High=4-5.
    low_end = max(1, int(math.floor(sc * 0.4)))
    high_start = min(sc, int(math.ceil(sc * 0.8)))

    if s <= low_end:
        return 1  # Low
    if s >= high_start:
        return 3  # High
    return 2  # Mid


# Classic 3x3 R (recency) × F (frequency) matrix segments.
# r_level / f_level: 1=Low, 2=Mid, 3=High
RFM_MATRIX_SEGMENT_BY_LEVEL: Dict[Tuple[int, int], str] = {
    (3, 3): "Champions",
    (3, 2): "Potential Loyalist",
    (3, 1): "New Customers",
    (2, 3): "Loyal Customers",
    (2, 2): "Need Attention",
    (2, 1): "Promising",
    (1, 3): "Can't Lose Them",
    (1, 2): "At Risk",
    (1, 1): "Hibernating",
}


def _rfm_assign_segment(r_score: int, f_score: int, *, scale: int) -> str:
    r_level = _rfm_score_level(r_score, scale=scale)
    f_level = _rfm_score_level(f_score, scale=scale)
    return RFM_MATRIX_SEGMENT_BY_LEVEL.get((r_level, f_level), "Standard")


def _rfm_segment_color(segment: str) -> str:
    # HOKA brand palette mapping (good / potential / at risk)
    if segment in {"Champions", "Loyal Customers"}:
        return "#1D70B8"  # Brand Blue
    if segment in {"Can't Lose Them", "At Risk", "Hibernating"}:
        return "#FF7043"  # Coral
    return "#D4E157"  # Neon Lime (Potential/Standard/New)


def _rfm_score_distribution(scores: pd.Series, scale: int) -> Dict[str, int]:
    counts = scores.value_counts(dropna=False).to_dict()
    return {str(i): int(counts.get(i, 0)) for i in range(1, int(scale) + 1)}


def _rfm_generate_insight_markdown(
    overview: Dict[str, Any],
    segments: List[Dict[str, Any]],
    mom: Optional[Dict[str, Any]] = None,
) -> str:
    total_users = int(overview.get("total_users") or 0)
    total_revenue = float(overview.get("total_revenue") or 0.0)

    seg_by_name = {s["segment"]: s for s in segments}
    champions = seg_by_name.get("Champions")
    cant_lose = seg_by_name.get("Can't Lose Them")
    at_risk = seg_by_name.get("At Risk")

    # Executive Summary
    exec_bits: List[str] = []
    if champions:
        exec_bits.append(
            "Champions represent "
            f'{champions.get("share_pct", 0):.1f}% of users and contribute '
            f'{champions.get("revenue_share_pct", 0):.1f}% of revenue.'
        )
    if cant_lose:
        exec_bits.append(
            "Can't Lose Them represent "
            f"{cant_lose.get('share_pct', 0):.1f}% of users and contribute "
            f"{cant_lose.get('revenue_share_pct', 0):.1f}% of revenue."
        )
    if at_risk:
        exec_bits.append(f'At Risk represent {at_risk.get("share_pct", 0):.1f}% of users.')
    if not exec_bits:
        exec_bits.append("RFM segmentation is ready to identify high-value and churn-risk cohorts.")

    # Opportunity & Risk
    opp_lines: List[str] = []
    if at_risk and at_risk.get("revenue_share_pct", 0) >= 10:
        opp_lines.append(
            "High-value churn risk: At Risk contributes a meaningful share of revenue; prioritize win-back."
        )
    if cant_lose and cant_lose.get("revenue_share_pct", 0) >= 10:
        opp_lines.append(
            "Key customers are slipping: Can't Lose Them has high frequency but low recency; use high-touch reactivation."
        )
    if champions and champions.get("revenue_share_pct", 0) < 20 and total_users > 0:
        opp_lines.append(
            "Low high-value concentration: Champions revenue share is low; focus on upgrading users into Champions."
        )

    if mom and isinstance(mom.get("total_revenue_change_pct"), (int, float)):
        tr = mom["total_revenue_change_pct"]
        opp_lines.append(f"MoM revenue change: {tr:+.1f}%.")

    if not opp_lines:
        opp_lines.append(
            "No major anomalies detected; focus on moving Potential Loyalist / Need Attention toward Loyal Customers and Champions."
        )

    # Action Plan (top 2 segments by revenue)
    top2 = sorted(segments, key=lambda s: float(s.get("revenue", 0) or 0), reverse=True)[:2]
    action_lines: List[str] = []
    for s in top2:
        name = s["segment"]
        if name == "Champions":
            action_lines.append("Champions: launch new arrivals and premium bundles; protect with VIP/loyalty benefits.")
        elif name == "Can't Lose Them":
            action_lines.append("Can't Lose Them: high-touch win-back (exclusive voucher + outreach); diagnose churn reasons.")
        elif name == "At Risk":
            action_lines.append("At Risk: limited-time incentives + personalized recommendations to prevent churn.")
        elif name == "Loyal Customers":
            action_lines.append("Loyal Customers: replenishment reminders + cross-sell; nurture into Champions.")
        elif name == "Potential Loyalist":
            action_lines.append("Potential Loyalist: drive 2nd/3rd purchase with bundles, thresholds, and points.")
        elif name == "New Customers":
            action_lines.append("New Customers: onboarding + early follow-up to build a repeat habit.")
        elif name == "Need Attention":
            action_lines.append("Need Attention: content/new arrivals + personalization to avoid drifting into churn segments.")
        elif name == "Promising":
            action_lines.append("Promising: post-first-purchase journey (7-21 days) to trigger repeat purchase.")
        elif name == "Hibernating":
            action_lines.append("Hibernating: low-cost outreach + low-barrier offers; measure reactivation rate.")
        else:
            action_lines.append("Standard: segmented remarketing to move users toward Loyal Customers and Champions.")

    return (
        "### Executive Summary\n"
        + " ".join(exec_bits)
        + "\n\n### Opportunity & Risk\n"
        + "\n".join([f"- {l}" for l in opp_lines])
        + "\n\n### Action Plan\n"
        + "\n".join([f"- {l}" for l in action_lines])
    )


def _read_dotenv(path: Path) -> Dict[str, str]:
    """Minimal .env reader (KEY=VALUE, ignores comments)."""
    if not path.exists():
        return {}
    out: Dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip("'").strip('"')
    return out


def _get_volcengine_api_key() -> Optional[str]:
    # Prefer the same env var used by the agent service.
    key = os.getenv("VOLCENGINE_API_KEY2") or os.getenv("VOLCENGINE_API_KEY")
    if key:
        return key

    # Fallback: reuse agent/.env if present in this monorepo (demo convenience).
    agent_env = project_root.parent / "agent" / ".env"
    env_map = _read_dotenv(agent_env)
    return env_map.get("VOLCENGINE_API_KEY2") or env_map.get("VOLCENGINE_API_KEY")


def _call_volcengine_chat_completion(
    *,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    max_tokens: int = 1024,
    timeout_seconds: int = 30,
) -> str:
    api_key = _get_volcengine_api_key()
    if not api_key:
        raise RuntimeError("VOLCENGINE_API_KEY2/VOLCENGINE_API_KEY 未配置")

    base_url = (os.getenv("VOLCENGINE_API_URL") or os.getenv("VOLCENGINE_BASE_URL") or "https://ark.cn-beijing.volces.com/api/v3").rstrip("/")
    url = f"{base_url}/chat/completions"

    payload = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "stream": False,
    }

    req = urlrequest.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urlrequest.urlopen(req, timeout=timeout_seconds) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except urlerror.HTTPError as e:
        err_body = ""
        try:
            err_body = e.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        raise RuntimeError(f"Volcengine HTTP {e.code}: {err_body or str(e)}")
    except Exception as e:
        raise RuntimeError(f"Volcengine request failed: {e}")

    data = json.loads(body or "{}")
    choices = data.get("choices") or []
    if not choices or not isinstance(choices, list):
        raise RuntimeError(f"Volcengine response missing choices: {data}")
    msg = choices[0].get("message") or {}
    content = msg.get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(f"Volcengine response missing content: {data}")
    return content.strip()


def _rfm_generate_insight_markdown_ai(
    *,
    dataset_id: str,
    scoring_scale: int,
    weights: Dict[str, Any],
    segmentation_method: str,
    overview: Dict[str, Any],
    segments: List[Dict[str, Any]],
    matrix: Optional[Dict[str, Any]] = None,
    mom: Optional[Dict[str, Any]],
) -> str:
    model = os.getenv("VOLCENGINE_MODEL") or os.getenv("RFM_INSIGHT_MODEL") or "kimi-k2-250905"
    timeout_seconds = int(os.getenv("RFM_AI_TIMEOUT_SECONDS") or "30")
    max_tokens = int(os.getenv("RFM_AI_MAX_TOKENS") or "1200")

    # Keep the prompt compact but explicit; pass structured context as JSON.
    context = {
        "dataset_id": dataset_id,
        "overview": overview,
        "segments": [
            {
                "segment": s.get("segment"),
                "count": s.get("count"),
                "share_pct": s.get("share_pct"),
                "avg_recency_days": s.get("avg_recency_days"),
                "avg_frequency": s.get("avg_frequency"),
                "avg_monetary": s.get("avg_monetary"),
                "revenue": s.get("revenue"),
                "revenue_share_pct": s.get("revenue_share_pct"),
            }
            for s in segments
        ],
        "matrix": (
            {
                "thresholds": (matrix or {}).get("thresholds"),
                "cells": [
                    {
                        "segment": c.get("segment"),
                        "r_level": c.get("r_level"),
                        "f_level": c.get("f_level"),
                        "count": c.get("count"),
                        "share_pct": c.get("share_pct"),
                        "avg_recency_days": c.get("avg_recency_days"),
                        "avg_frequency": c.get("avg_frequency"),
                        "avg_monetary": c.get("avg_monetary"),
                        "revenue": c.get("revenue"),
                        "revenue_share_pct": c.get("revenue_share_pct"),
                        "avg_rfm_score": c.get("avg_rfm_score"),
                    }
                    for c in ((matrix or {}).get("cells") or [])
                ],
            }
            if matrix
            else None
        ),
        "mom": mom,
        "scoring_scale": scoring_scale,
        "weights": weights,
        "segmentation_method": segmentation_method,
    }

    system = (
        "You are a senior growth analyst. Generate business insights and recommendations from RFM segmentation.\n"
        "Output MUST be Markdown and MUST contain exactly these three sections (use '### ' headings):\n"
        "1) ### Executive Summary\n"
        "2) ### Opportunity & Risk\n"
        "3) ### Action Plan\n"
        "Rules:\n"
        "- Write in English.\n"
        "- Keep it concise and actionable.\n"
        "- In Opportunity & Risk use bullet list.\n"
        "- In Action Plan provide concrete strategies for Top 2 key segments (by revenue contribution).\n"
        "- Do not include any other sections or preamble.\n"
    )

    user = (
        "RFM context (JSON):\n"
        + json.dumps(context, ensure_ascii=False, separators=(",", ":"))
    )

    return _call_volcengine_chat_completion(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.2,
        max_tokens=max_tokens,
        timeout_seconds=timeout_seconds,
    )


def _segmentation_generate_insight_markdown(
    *,
    feature_definitions: List[Dict[str, Any]],
    clusters_summary: List[Dict[str, Any]],
) -> str:
    """Deterministic fallback insight generator (English)."""
    clusters = [c for c in (clusters_summary or []) if isinstance(c, dict)]
    clusters.sort(key=lambda x: int(x.get("cluster_id", 0)))

    def top_features(chars: Dict[str, Any], *, prefix: str, n: int = 3) -> List[Tuple[str, float]]:
        out: List[Tuple[str, float]] = []
        for k, v in (chars or {}).items():
            if not isinstance(k, str) or not k.startswith(prefix):
                continue
            try:
                mean_v = float((v or {}).get("mean"))
            except Exception:
                continue
            if not np.isfinite(mean_v):
                continue
            out.append((k, mean_v))
        out.sort(key=lambda x: x[1], reverse=True)
        return out[:n]

    lines: List[str] = []
    lines.append("### Executive Summary")
    lines.append(f"- Generated {len(clusters)} customer clusters using K-Means.")
    lines.append("")
    for c in clusters:
        cid = int(c.get("cluster_id", 0))
        size = c.get("size")
        pct = c.get("percentage")
        label = c.get("label_suggestion") or f"Cluster {cid}"
        lines.append(f"### Cluster {cid}: {label}")
        lines.append(f"- Size: {size} ({pct})")
        chars = c.get("characteristics") or {}

        top_cats = top_features(chars, prefix="cat_share_", n=3)
        if top_cats:
            tops_s = ", ".join([f"`{k}`={v*100:.1f}%" for k, v in top_cats])
            lines.append(f"- Dominant categories: {tops_s}")

        top_seasons = top_features(chars, prefix="season_share_", n=1)
        if top_seasons:
            lines.append(f"- Seasonality: `{top_seasons[0][0]}`={top_seasons[0][1]*100:.1f}%")

        try:
            f_mean = float((chars.get("frequency") or {}).get("mean"))
        except Exception:
            f_mean = float("nan")
        try:
            m_mean = float((chars.get("monetary") or {}).get("mean"))
        except Exception:
            m_mean = float("nan")
        if np.isfinite(f_mean):
            lines.append(f"- Avg frequency: {f_mean:.2f}")
        if np.isfinite(m_mean):
            lines.append(f"- Avg monetary: {m_mean:.2f}")

        lines.append("- Recommended actions: tailor messaging, product bundles, and timing around the dominant mix.")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _segmentation_generate_insight_markdown_ai(
    *,
    dataset_id: str,
    feature_mode: str,
    feature_definitions: List[Dict[str, Any]],
    clusters_summary: List[Dict[str, Any]],
) -> str:
    """AI insight generator for segmentation clusters (English, Markdown)."""
    model = os.getenv("VOLCENGINE_MODEL") or os.getenv("SEGMENTATION_INSIGHT_MODEL") or "kimi-k2-250905"
    timeout_seconds = int(os.getenv("SEGMENTATION_AI_TIMEOUT_SECONDS") or "30")
    max_tokens = int(os.getenv("SEGMENTATION_AI_MAX_TOKENS") or "1200")

    clusters = [c for c in (clusters_summary or []) if isinstance(c, dict)]
    clusters.sort(key=lambda x: int(x.get("cluster_id", 0)))

    context = {
        "dataset_id": dataset_id,
        "feature_mode": feature_mode,
        "features": [
            {"name": d.get("name"), "description": d.get("description")}
            for d in (feature_definitions or [])
            if isinstance(d, dict)
        ],
        "clusters": [
            {
                "cluster_id": c.get("cluster_id"),
                "size": c.get("size"),
                "percentage": c.get("percentage"),
                "label_suggestion": c.get("label_suggestion"),
                "characteristics": c.get("characteristics"),
            }
            for c in clusters
        ],
    }

    system = (
        "You are a senior retail analyst. Interpret customer clusters produced by K-Means segmentation.\n"
        "Output MUST be Markdown.\n"
        "For each cluster, use this exact structure:\n"
        "### Cluster <id>: <short name>\n"
        "- Profile: <2-4 bullet points>\n"
        "- Key signals: <2-4 bullet points referencing provided feature names>\n"
        "- Recommended actions: <3 bullet points>\n"
        "Rules:\n"
        "- Write in English.\n"
        "- Do NOT invent data; only use the provided metrics.\n"
        "- Keep it concise but actionable.\n"
    )
    user = "Segmentation context (JSON):\n" + json.dumps(context, ensure_ascii=False, separators=(",", ":"))

    return _call_volcengine_chat_completion(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.2,
        max_tokens=max_tokens,
        timeout_seconds=timeout_seconds,
    )


@router.get("")
def get_datasets_list():
    """获取数据集列表"""
    try:
        config = load_datasets_config()
        datasets = []
        
        for dataset_config in config.get("datasets", []):
            dataset_id = dataset_config.get("id")
            csv_path = _get_dataset_csv_path(str(dataset_id), dataset_config)
            row_count = _csv_count_rows(csv_path) if csv_path.exists() else 0
            datasets.append(
                {
                    "id": dataset_config["id"],
                    "name": dataset_config["name"],
                    "description": dataset_config["description"],
                    "table_name": dataset_config.get("table_name", dataset_config["id"]),
                    "type": dataset_config["type"],
                    "row_count": row_count,
                    "created_at": dataset_config.get("created_at"),
                    "updated_at": dataset_config.get("updated_at"),
                    "tags": dataset_config.get("tags", []),
                }
            )
        
        return {
            "success": True,
            "data": datasets,
            "total": len(datasets),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取数据集列表失败: {str(e)}")


@router.get("/{dataset_id}/preview")
def preview_dataset(
    dataset_id: str,
    page: int = Query(1, ge=1),
    pageSize: int = Query(20, ge=1, le=100),
):
    """预览数据集数据（分页）"""
    try:
        config = load_datasets_config()
        dataset_config = next(
            (d for d in config.get("datasets", []) if d["id"] == dataset_id), None
        )
        
        if not dataset_config:
            raise HTTPException(status_code=404, detail="数据集不存在")

        csv_path = _get_dataset_csv_path(dataset_id, dataset_config)
        if not csv_path.exists():
            raise HTTPException(status_code=404, detail=f"CSV not found: {csv_path}")

        encoding, header_cols = _read_csv_header(csv_path)
        row_count = _csv_count_rows(csv_path)

        offset = (page - 1) * pageSize
        if offset < 0:
            offset = 0

        df_page = pd.read_csv(
            csv_path,
            encoding=encoding,
            header=None,
            names=header_cols,
            skiprows=1 + offset,
            nrows=pageSize,
            low_memory=False,
        )
        # Replace NaN, inf, and -inf with None for JSON serialization
        # Replace inf values first, then NaN values
        df_page = df_page.replace([np.inf, -np.inf], np.nan)
        df_page = df_page.where(pd.notna(df_page), None)
        rows = df_page.to_dict(orient="records")
        # Clean up any remaining invalid float values in the records
        for row in rows:
            for key, value in row.items():
                if isinstance(value, (float, np.floating)):
                    if np.isinf(value) or np.isnan(value):
                        row[key] = None

        return {
            "success": True,
            "data": {
                "records": rows,
                "total": row_count,
                "page": page,
                "pageSize": pageSize,
                "totalPages": (row_count + pageSize - 1) // pageSize,
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预览数据集失败: {str(e)}")


@router.get("/{dataset_id}")
def get_dataset_detail(dataset_id: str, include_stats: bool = False):
    """获取数据集详情
    include_stats: 是否包含列统计信息（默认为 False 以提高响应速度）
    """
    try:
        config = load_datasets_config()
        dataset_config = next(
            (d for d in config.get("datasets", []) if d["id"] == dataset_id), None
        )
        
        if not dataset_config:
            raise HTTPException(status_code=404, detail="数据集不存在")

        csv_path = _get_dataset_csv_path(dataset_id, dataset_config)
        if not csv_path.exists():
            raise HTTPException(status_code=404, detail=f"CSV not found: {csv_path}")

        encoding, header_cols = _read_csv_header(csv_path)
        row_count = _csv_count_rows(csv_path)

        # Parse time column (if configured) for better typing/time range.
        time_range = None
        parse_dates: List[str] = []
        if dataset_config.get("time_column"):
            try:
                resolved_time_col = _resolve_csv_column(header_cols, str(dataset_config["time_column"]))
                parse_dates = [resolved_time_col]
                time_range = _csv_time_range(csv_path, encoding=encoding, time_col=resolved_time_col)
            except Exception:
                # Keep it resilient if the config doesn't match the CSV header.
                time_range = None
                parse_dates = []

        sample_df = pd.read_csv(
            csv_path,
            encoding=encoding,
            nrows=2000,
            parse_dates=parse_dates if parse_dates else False,
            low_memory=False,
        )
        inferred_types = _csv_infer_column_types(sample_df, header_cols)

        column_list: List[Dict[str, Any]] = []
        for name in header_cols:
            col_type = inferred_types.get(name, "text")
            stats = _csv_column_stats(sample_df, name, col_type) if include_stats else None
            column_list.append({"name": name, "type": col_type, "stats": stats})
        
        return {
            "success": True,
            "data": {
                "id": dataset_config["id"],
                "name": dataset_config["name"],
                "description": dataset_config["description"],
                "table_name": dataset_config.get("table_name", dataset_config["id"]),
                "type": dataset_config["type"],
                "row_count": row_count,
                "column_count": len(header_cols),
                "created_at": dataset_config.get("created_at"),
                "updated_at": dataset_config.get("updated_at"),
                "tags": dataset_config.get("tags", []),
                "time_range": time_range,
                "columns": column_list,
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取数据集详情失败: {str(e)}")


@router.get("/{dataset_id}/stats")
def get_dataset_stats(dataset_id: str):
    """获取数据集列统计信息"""
    try:
        config = load_datasets_config()
        dataset_config = next(
            (d for d in config.get("datasets", []) if d["id"] == dataset_id), None
        )
        
        if not dataset_config:
            raise HTTPException(status_code=404, detail="数据集不存在")

        csv_path = _get_dataset_csv_path(dataset_id, dataset_config)
        if not csv_path.exists():
            raise HTTPException(status_code=404, detail=f"CSV not found: {csv_path}")

        encoding, header_cols = _read_csv_header(csv_path)

        # Load the CSV once for stats. (Demo datasets are moderate in size.)
        parse_dates: List[str] = []
        if dataset_config.get("time_column"):
            try:
                resolved_time_col = _resolve_csv_column(header_cols, str(dataset_config["time_column"]))
                parse_dates = [resolved_time_col]
            except Exception:
                parse_dates = []

        df = pd.read_csv(
            csv_path,
            encoding=encoding,
            parse_dates=parse_dates if parse_dates else False,
            low_memory=False,
        )
        inferred_types = _csv_infer_column_types(df.head(2000), header_cols)

        column_stats_list: List[Dict[str, Any]] = []
        for name in header_cols:
            col_type = inferred_types.get(name, "text")
            stats = _csv_column_stats(df, name, col_type)
            column_stats_list.append({"name": name, "type": col_type, "stats": stats})
            
        return {
            "success": True,
            "data": column_stats_list
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取统计信息失败: {str(e)}")


@router.post("/{dataset_id}/rfm")
def run_rfm_analysis(dataset_id: str, req: RFMRunRequest):
    """Run RFM analysis for a dataset.

    Notes:
    - Directly reads the configured dataset CSV (no DB dependency).
    - Uses max(order_date) in the dataset as the reference "Today" to avoid empty windows on historical datasets.
    """
    method = (req.segmentation_method or "quantiles").lower()
    if method not in {"quantiles", "kmeans"}:
        raise HTTPException(status_code=400, detail="segmentation_method must be 'quantiles' or 'kmeans'")

    config = load_datasets_config()
    dataset_config = next((d for d in config.get("datasets", []) if d["id"] == dataset_id), None)
    if not dataset_config:
        raise HTTPException(status_code=404, detail="Dataset not found")

    rfm_cfg = dataset_config.get("rfm")
    if not isinstance(rfm_cfg, dict):
        raise HTTPException(status_code=400, detail="RFM mapping not configured for this dataset (datasets.json -> rfm)")

    user_col = rfm_cfg.get("user_id_column")
    order_col = rfm_cfg.get("order_id_column")
    date_col = rfm_cfg.get("order_date_column")
    amount_cfg = rfm_cfg.get("monetary")

    if not all(isinstance(x, str) and x for x in [user_col, order_col, date_col]):
        raise HTTPException(status_code=400, detail="RFM mapping incomplete (user_id_column/order_id_column/order_date_column)")
    if not isinstance(amount_cfg, dict):
        raise HTTPException(status_code=400, detail="RFM monetary config missing (datasets.json -> rfm.monetary)")

    base_filters = rfm_cfg.get("filters") or []
    if not isinstance(base_filters, list):
        raise HTTPException(status_code=400, detail="RFM filters config must be an array")

    csv_path = _get_dataset_csv_path(dataset_id, dataset_config)
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail=f"CSV not found: {csv_path}")

    encoding, header_cols = _read_csv_header(csv_path)

    # Resolve configured column names against the CSV header (case-insensitive).
    user_col_resolved = _resolve_csv_column(header_cols, str(user_col))
    order_col_resolved = _resolve_csv_column(header_cols, str(order_col))
    date_col_resolved = _resolve_csv_column(header_cols, str(date_col))

    amount_cfg_resolved = dict(amount_cfg)
    if amount_cfg_resolved.get("type") == "column":
        amount_cfg_resolved["column"] = _resolve_csv_column(header_cols, str(amount_cfg_resolved.get("column") or ""))
    elif amount_cfg_resolved.get("type") == "mul":
        amount_cfg_resolved["left"] = _resolve_csv_column(header_cols, str(amount_cfg_resolved.get("left") or ""))
        amount_cfg_resolved["right"] = _resolve_csv_column(header_cols, str(amount_cfg_resolved.get("right") or ""))

    resolved_filters: List[Dict[str, Any]] = []
    for f in base_filters:
        if not isinstance(f, dict):
            raise HTTPException(status_code=400, detail="RFM filters config must be an array of objects")
        col = f.get("column")
        if not isinstance(col, str) or not col:
            raise HTTPException(status_code=400, detail="RFM filters: invalid column")
        resolved_filters.append({**f, "column": _resolve_csv_column(header_cols, col)})

    # Read only required columns to keep memory usage reasonable.
    usecols_set = {user_col_resolved, order_col_resolved, date_col_resolved}
    if amount_cfg_resolved.get("type") == "column":
        usecols_set.add(amount_cfg_resolved["column"])
    elif amount_cfg_resolved.get("type") == "mul":
        usecols_set.add(amount_cfg_resolved["left"])
        usecols_set.add(amount_cfg_resolved["right"])
    for f in resolved_filters:
        usecols_set.add(f["column"])

    df_raw = pd.read_csv(csv_path, encoding=encoding, usecols=sorted(usecols_set), low_memory=False)
    df_raw[date_col_resolved] = pd.to_datetime(df_raw[date_col_resolved], errors="coerce")
    df_raw = df_raw.dropna(subset=[user_col_resolved, order_col_resolved, date_col_resolved])

    if resolved_filters:
        df_raw = _apply_filters_df(df_raw, resolved_filters)

    if df_raw.empty:
        raise HTTPException(status_code=400, detail="No usable data after applying CSV filters")

    # Normalize ids to strings to avoid pandas float representation like "17850.0".
    def _normalize_id_series(s: pd.Series) -> pd.Series:
        num = pd.to_numeric(s, errors="coerce")
        out = s.astype(str)
        mask_int = num.notna() & num.mod(1).eq(0)
        out.loc[mask_int] = num.loc[mask_int].astype("int64").astype(str)
        return out

    df_raw[user_col_resolved] = _normalize_id_series(df_raw[user_col_resolved])
    df_raw[order_col_resolved] = _normalize_id_series(df_raw[order_col_resolved])

    reference_date = pd.to_datetime(df_raw[date_col_resolved].max())
    if pd.isna(reference_date):
        raise HTTPException(status_code=400, detail="Dataset has no valid order_date; cannot compute RFM")

    window_days = int(req.time_window_days)
    current_start = reference_date - pd.Timedelta(days=window_days)
    df_current = df_raw[
        (df_raw[date_col_resolved] >= current_start) & (df_raw[date_col_resolved] <= reference_date)
    ]
    if df_current.empty:
        raise HTTPException(status_code=400, detail="No data found within the analysis time window")

    df = _rfm_aggregate_users(
        df_current,
        user_col=user_col_resolved,
        order_col=order_col_resolved,
        date_col=date_col_resolved,
        amount_cfg=amount_cfg_resolved,
        reference_date=reference_date,
    )
    if df.empty:
        raise HTTPException(status_code=400, detail="No valid user-level data after aggregation")

    scale = int(req.scoring_scale)
    if method == "quantiles":
        df["r_score"] = _rfm_score_quantiles(df["recency_days"], scale=scale, higher_is_better=False)
        df["f_score"] = _rfm_score_quantiles(df["frequency"], scale=scale, higher_is_better=True)
        df["m_score"] = _rfm_score_quantiles(df["monetary"], scale=scale, higher_is_better=True)
    else:
        df["r_score"] = _rfm_score_kmeans(df["recency_days"], scale=scale, higher_is_better=False)
        df["f_score"] = _rfm_score_kmeans(df["frequency"], scale=scale, higher_is_better=True)
        df["m_score"] = _rfm_score_kmeans(df["monetary"], scale=scale, higher_is_better=True)

    df["rfm_score"] = (
        df["r_score"].astype(float) * float(req.weights.r)
        + df["f_score"].astype(float) * float(req.weights.f)
        + df["m_score"].astype(float) * float(req.weights.m)
    )

    # Classic 3x3 matrix needs stable Low/Mid/High buckets.
    df["r_level"] = df["r_score"].apply(lambda v: _rfm_score_level(int(v), scale=scale)).astype("int64")
    df["f_level"] = df["f_score"].apply(lambda v: _rfm_score_level(int(v), scale=scale)).astype("int64")
    df["segment"] = [
        RFM_MATRIX_SEGMENT_BY_LEVEL.get((int(r), int(f)), "Standard")
        for r, f in zip(df["r_level"].tolist(), df["f_level"].tolist())
    ]

    total_users = int(df.shape[0])
    total_revenue = float(df["monetary"].sum())
    total_orders = int(df["frequency"].sum())

    # Aggregate to classic R×F matrix cells.
    cell_stats = (
        df.groupby(["r_level", "f_level"])
        .agg(
            count=("user_id", "count"),
            revenue=("monetary", "sum"),
            avg_recency_days=("recency_days", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_monetary=("monetary", "mean"),
            avg_r_score=("r_score", "mean"),
            avg_f_score=("f_score", "mean"),
            avg_m_score=("m_score", "mean"),
            avg_rfm_score=("rfm_score", "mean"),
        )
        .reset_index()
    )

    def _cell_row(r_level: int, f_level: int) -> Optional[pd.Series]:
        hit = cell_stats[(cell_stats["r_level"] == r_level) & (cell_stats["f_level"] == f_level)]
        if hit.empty:
            return None
        return hit.iloc[0]

    matrix_cells: List[Dict[str, Any]] = []
    segments: List[Dict[str, Any]] = []
    for r_level in (3, 2, 1):  # High -> Low
        for f_level in (3, 2, 1):  # High -> Low
            segment_name = RFM_MATRIX_SEGMENT_BY_LEVEL.get((r_level, f_level), "Standard")
            row = _cell_row(r_level, f_level)

            cnt = int(row["count"]) if row is not None else 0
            rev = float(row["revenue"] or 0.0) if row is not None else 0.0

            payload = {
                "segment": segment_name,
                "count": cnt,
                "share_pct": (cnt / total_users * 100.0) if total_users else 0.0,
                "revenue": rev,
                "revenue_share_pct": (rev / total_revenue * 100.0) if total_revenue else 0.0,
                "avg_recency_days": float(row["avg_recency_days"] or 0.0) if row is not None else 0.0,
                "avg_frequency": float(row["avg_frequency"] or 0.0) if row is not None else 0.0,
                "avg_monetary": float(row["avg_monetary"] or 0.0) if row is not None else 0.0,
                "avg_r_score": float(row["avg_r_score"] or 0.0) if row is not None else 0.0,
                "avg_f_score": float(row["avg_f_score"] or 0.0) if row is not None else 0.0,
                "avg_m_score": float(row["avg_m_score"] or 0.0) if row is not None else 0.0,
                "avg_rfm_score": float(row["avg_rfm_score"] or 0.0) if row is not None else 0.0,
                "color": _rfm_segment_color(segment_name),
            }

            segments.append(payload)
            matrix_cells.append({"r_level": r_level, "f_level": f_level, **payload})

    segments.sort(key=lambda s: (s["count"], s["revenue"]), reverse=True)

    low_end = max(1, int(math.floor(scale * 0.4)))
    high_start = min(scale, int(math.ceil(scale * 0.8)))
    matrix = {
        "rows": [
            {"id": 3, "label": "R High (Recent)"},
            {"id": 2, "label": "R Medium"},
            {"id": 1, "label": "R Low (Stale)"},
        ],
        "cols": [
            {"id": 3, "label": "F High"},
            {"id": 2, "label": "F Medium"},
            {"id": 1, "label": "F Low"},
        ],
        "thresholds": {
            "scale": scale,
            "low_end": low_end,
            "high_start": high_start,
            "rule": "score<=low_end => Low, score>=high_start => High, else Mid",
        },
        "cells": matrix_cells,
    }

    # MoM (previous window of same length). Best-effort; when unavailable returns None.
    mom: Optional[Dict[str, Any]] = None
    try:
        prev_end = current_start
        prev_start = prev_end - pd.Timedelta(days=window_days)
        df_prev = df_raw[(df_raw[date_col_resolved] >= prev_start) & (df_raw[date_col_resolved] <= prev_end)]
        if not df_prev.empty:
            prev_users_df = _rfm_aggregate_users(
                df_prev,
                user_col=user_col_resolved,
                order_col=order_col_resolved,
                date_col=date_col_resolved,
                amount_cfg=amount_cfg_resolved,
                reference_date=pd.to_datetime(prev_end),
            )
            if not prev_users_df.empty:
                prev_users = int(prev_users_df.shape[0])
                prev_rev = float(prev_users_df["monetary"].sum())
                mom = {
                    "total_users": prev_users,
                    "total_revenue": prev_rev,
                    "total_users_change_pct": ((total_users - prev_users) / prev_users * 100.0) if prev_users else None,
                    "total_revenue_change_pct": ((total_revenue - prev_rev) / prev_rev * 100.0) if prev_rev else None,
                }
    except Exception as _e:
        mom = None

    overview = {
        "total_users": total_users,
        "total_orders": total_orders,
        "total_revenue": total_revenue,
    }

    analysis_id = _rfm_make_analysis_id(dataset_id, req)
    _rfm_cache_put(
        analysis_id,
        {
            "created_at": datetime.utcnow(),
            "dataset_id": dataset_id,
            "params": req.model_dump(),
            "df": df[
                [
                    "user_id",
                    "recency_days",
                    "frequency",
                    "monetary",
                    "r_score",
                    "f_score",
                    "m_score",
                    "rfm_score",
                    "r_level",
                    "f_level",
                    "segment",
                ]
            ].copy(),
        },
    )

    insight_md: str
    enable_ai = os.getenv("RFM_ENABLE_AI_INSIGHT")
    # Default: enable when a Volcengine API key is available.
    if enable_ai is None:
        enable_ai = "1" if _get_volcengine_api_key() else "0"

    if str(enable_ai).lower() in {"1", "true", "yes", "on"}:
        try:
            insight_md = _rfm_generate_insight_markdown_ai(
                dataset_id=dataset_id,
                scoring_scale=scale,
                weights=req.weights.model_dump(),
                segmentation_method=method,
                overview=overview,
                segments=segments,
                matrix=matrix,
                mom=mom,
            )
        except Exception as e:
            # Fall back to deterministic summary to keep the analysis usable.
            print(f"⚠️  RFM AI insight generation failed, fallback to template: {e}")
            insight_md = _rfm_generate_insight_markdown(overview=overview, segments=segments, mom=mom)
    else:
        insight_md = _rfm_generate_insight_markdown(overview=overview, segments=segments, mom=mom)

    return {
        "success": True,
        "data": {
            "analysis_id": analysis_id,
            "dataset_id": dataset_id,
            "reference_date": str(reference_date),
            "time_window_days": int(req.time_window_days),
            "scoring_scale": scale,
            "segmentation_method": method,
            "weights": req.weights.model_dump(),
            "overview": overview,
            "segments": segments,
            "matrix": matrix,
            "score_distributions": {
                "r": _rfm_score_distribution(df["r_score"], scale=scale),
                "f": _rfm_score_distribution(df["f_score"], scale=scale),
                "m": _rfm_score_distribution(df["m_score"], scale=scale),
            },
            "mom": mom,
            "insight_markdown": insight_md,
        },
    }


@router.get("/{dataset_id}/rfm/{analysis_id}/segment/{segment_name}")
def get_rfm_segment_detail(
    dataset_id: str,
    analysis_id: str,
    segment_name: str,
    page: int = Query(1, ge=1),
    pageSize: int = Query(20, ge=1, le=200),
):
    """Drill-down: list users in a segment for a previously computed analysis."""
    cached = _rfm_cache_get(analysis_id, dataset_id=dataset_id)
    if not cached:
        raise HTTPException(status_code=404, detail="Analysis result expired; please run analysis again")

    df: pd.DataFrame = cached["df"]
    seg_df = df[df["segment"] == segment_name]

    total = int(seg_df.shape[0])
    if total == 0:
        return {"success": True, "data": {"records": [], "total": 0, "page": page, "pageSize": pageSize}}

    seg_df = seg_df.sort_values(by=["monetary", "frequency", "recency_days"], ascending=[False, False, True])
    start = (page - 1) * pageSize
    end = start + pageSize
    page_df = seg_df.iloc[start:end].copy()

    records: List[Dict[str, Any]] = []
    for _, r in page_df.iterrows():
        user_id = r["user_id"]
        # Pandas may keep numpy scalar types; convert to native Python for JSON serialization.
        if isinstance(user_id, (np.integer, np.floating)):
            user_id = user_id.item()
        records.append(
            {
                "user_id": user_id,
                "recency_days": int(r["recency_days"]),
                "frequency": int(r["frequency"]),
                "monetary": float(r["monetary"]),
                "r_score": int(r["r_score"]),
                "f_score": int(r["f_score"]),
                "m_score": int(r["m_score"]),
                "rfm_score": float(r["rfm_score"]),
                "segment": str(r["segment"]),
            }
        )

    return {
        "success": True,
        "data": {
            "records": records,
            "total": total,
            "page": page,
            "pageSize": pageSize,
        },
    }


@router.post("/{dataset_id}/segmentation")
def run_customer_segmentation(dataset_id: str, req: CustomerSegmentationRequest):
    """
    Customer Segmentation (K-Means).

    feature_mode:
    - "category_mix" (default): clean product descriptions -> category, build category revenue-share features, then cluster.
    - "rfm": legacy clustering on RFM (recency_days/frequency/monetary).
    """
    config = load_datasets_config()
    dataset_config = next((d for d in config.get("datasets", []) if d["id"] == dataset_id), None)
    if not dataset_config:
        raise HTTPException(status_code=404, detail="Dataset not found")

    rfm_cfg = dataset_config.get("rfm")
    if not isinstance(rfm_cfg, dict):
        raise HTTPException(status_code=400, detail="Missing dataset RFM mapping (datasets.json -> rfm)")

    user_col = rfm_cfg.get("user_id_column")
    order_col = rfm_cfg.get("order_id_column")
    date_col = rfm_cfg.get("order_date_column")
    amount_cfg = rfm_cfg.get("monetary")

    if not all(isinstance(x, str) and x for x in [user_col, order_col, date_col]):
        raise HTTPException(
            status_code=400,
            detail="Incomplete RFM mapping (user_id_column/order_id_column/order_date_column)",
        )
    if not isinstance(amount_cfg, dict):
        raise HTTPException(status_code=400, detail="Missing RFM monetary mapping (datasets.json -> rfm.monetary)")

    base_filters = rfm_cfg.get("filters") or []
    if not isinstance(base_filters, list):
        raise HTTPException(status_code=400, detail="RFM filters must be an array")

    csv_path = _get_dataset_csv_path(dataset_id, dataset_config)
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail=f"CSV file not found: {csv_path}")

    encoding, header_cols = _read_csv_header(csv_path)

    user_col_resolved = _resolve_csv_column(header_cols, str(user_col))
    order_col_resolved = _resolve_csv_column(header_cols, str(order_col))
    date_col_resolved = _resolve_csv_column(header_cols, str(date_col))

    amount_cfg_resolved = dict(amount_cfg)
    if amount_cfg_resolved.get("type") == "column":
        amount_cfg_resolved["column"] = _resolve_csv_column(header_cols, str(amount_cfg_resolved.get("column") or ""))
    elif amount_cfg_resolved.get("type") == "mul":
        amount_cfg_resolved["left"] = _resolve_csv_column(header_cols, str(amount_cfg_resolved.get("left") or ""))
        amount_cfg_resolved["right"] = _resolve_csv_column(header_cols, str(amount_cfg_resolved.get("right") or ""))

    resolved_filters: List[Dict[str, Any]] = []
    for f in base_filters:
        if not isinstance(f, dict):
            raise HTTPException(status_code=400, detail="RFM filters must be an object array")
        col = f.get("column")
        if not isinstance(col, str) or not col:
            raise HTTPException(status_code=400, detail="RFM filters: invalid column")
        resolved_filters.append({**f, "column": _resolve_csv_column(header_cols, col)})

    # Read only required columns to keep memory usage reasonable.
    usecols_set = {user_col_resolved, order_col_resolved, date_col_resolved}
    if amount_cfg_resolved.get("type") == "column":
        usecols_set.add(amount_cfg_resolved["column"])
    elif amount_cfg_resolved.get("type") == "mul":
        usecols_set.add(amount_cfg_resolved["left"])
        usecols_set.add(amount_cfg_resolved["right"])
    for f in resolved_filters:
        usecols_set.add(f["column"])

    # Category-mix segmentation requires a description column for keyword-based category cleaning.
    feature_mode = (req.feature_mode or "category_mix").strip().lower()
    desc_col_resolved: Optional[str] = None
    if feature_mode == "category_mix":
        desc_col_resolved = _resolve_csv_column(header_cols, "Description")
        usecols_set.add(desc_col_resolved)

    df_raw = pd.read_csv(csv_path, encoding=encoding, usecols=sorted(usecols_set), low_memory=False)
    df_raw[date_col_resolved] = pd.to_datetime(df_raw[date_col_resolved], errors="coerce")
    df_raw = df_raw.dropna(subset=[user_col_resolved, order_col_resolved, date_col_resolved])

    if resolved_filters:
        df_raw = _apply_filters_df(df_raw, resolved_filters)

    if df_raw.empty:
        raise HTTPException(status_code=400, detail="No usable data after applying filters")

    # Normalize ids to strings to avoid pandas float representation like "17850.0".
    def _normalize_id_series(s: pd.Series) -> pd.Series:
        num = pd.to_numeric(s, errors="coerce")
        out = s.astype(str)
        mask_int = num.notna() & num.mod(1).eq(0)
        out.loc[mask_int] = num.loc[mask_int].astype("int64").astype(str)
        return out

    df_raw[user_col_resolved] = _normalize_id_series(df_raw[user_col_resolved])
    df_raw[order_col_resolved] = _normalize_id_series(df_raw[order_col_resolved])

    reference_date = pd.to_datetime(df_raw[date_col_resolved].max())
    if pd.isna(reference_date):
        raise HTTPException(status_code=400, detail="Dataset has no usable order date; cannot run segmentation")

    window_days = int(req.time_window_days)
    current_start = reference_date - pd.Timedelta(days=window_days)
    df_current = df_raw[
        (df_raw[date_col_resolved] >= current_start) & (df_raw[date_col_resolved] <= reference_date)
    ]
    if df_current.empty:
        raise HTTPException(status_code=400, detail="No data found within the selected time window")

    from shared.utils.customer_segmentation import segment_customers_kmeans

    if feature_mode == "rfm":
        df_users = _rfm_aggregate_users(
            df_current,
            user_col=user_col_resolved,
            order_col=order_col_resolved,
            date_col=date_col_resolved,
            amount_cfg=amount_cfg_resolved,
            reference_date=reference_date,
        )
        if df_users.empty:
            raise HTTPException(status_code=400, detail="No valid users after RFM aggregation")

        selected_features = req.selected_features or ["recency_days", "frequency", "monetary"]
        allowed = {"recency_days", "frequency", "monetary"}
        if not isinstance(selected_features, list) or not selected_features:
            raise HTTPException(status_code=400, detail="selected_features must be a non-empty list")
        if any((not isinstance(f, str)) or f not in allowed for f in selected_features):
            raise HTTPException(status_code=400, detail=f"selected_features must be one of: {sorted(allowed)}")

        feature_definitions = [
            {"name": "recency_days", "description": "Days since the customer's most recent purchase."},
            {"name": "frequency", "description": "Number of unique orders in the analysis window."},
            {"name": "monetary", "description": "Total revenue in the analysis window."},
        ]

        result = segment_customers_kmeans(
            df_users,
            selected_features=selected_features,
            k_range=tuple(req.k_range),
            random_seed=int(req.random_seed),
            outlier_threshold=req.outlier_threshold,
            user_id_column="user_id",
        )
        # Ensure the response stays English.
        result["feature_mode"] = "rfm"
        result["feature_definitions"] = [d for d in feature_definitions if d["name"] in (result.get("model_info") or {}).get("features_used", [])]
    elif feature_mode == "category_mix":
        if not desc_col_resolved:
            raise HTTPException(status_code=400, detail="Missing product description column (Description)")

        # Compute per-line revenue and clean product descriptions to a keyword-based category.
        df_work = df_current.copy()
        df_work["__amount"] = _rfm_compute_amount(df_work, amount_cfg_resolved).clip(lower=0.0)
        df_work["__category"] = _sales_hierarchy_assign_category(df_work[desc_col_resolved])

        # Aggregate per-customer revenue by category.
        cats = [str(r.get("category")) for r in SALES_HIERARCHY_RULES if str(r.get("category") or "").strip()]
        feature_by_cat = {c: _sales_hierarchy_category_feature_key(c) for c in cats}
        cat_by_feature = {v: k for k, v in feature_by_cat.items()}

        pivot = (
            df_work.groupby([user_col_resolved, "__category"], dropna=False)["__amount"]
            .sum()
            .unstack(fill_value=0.0)
        )
        # Ensure stable column order and presence.
        for c in cats:
            if c not in pivot.columns:
                pivot[c] = 0.0
        pivot = pivot[cats]

        total_rev = pivot.sum(axis=1).astype(float)
        pivot = pivot.loc[total_rev > 0.0].copy()
        total_rev = total_rev.loc[pivot.index]
        if pivot.empty:
            raise HTTPException(status_code=400, detail="No customers with positive revenue in the window")

        # Revenue-share feature matrix.
        share = pivot.div(pivot.sum(axis=1), axis=0).fillna(0.0)
        share.columns = [feature_by_cat[c] for c in cats]

        # Order-level metrics (frequency, monetary, seasonal shares) to avoid line-item duplication.
        orders = (
            df_work.groupby([user_col_resolved, order_col_resolved], dropna=False)
            .agg(order_date=(date_col_resolved, "max"), order_revenue=("__amount", "sum"))
            .reset_index()
        )
        orders["order_date"] = pd.to_datetime(orders["order_date"], errors="coerce")
        orders = orders.dropna(subset=["order_date"])
        if orders.empty:
            raise HTTPException(status_code=400, detail="No usable orders after preprocessing")

        user_metrics = (
            orders.groupby(user_col_resolved, dropna=False)
            .agg(frequency=(order_col_resolved, "count"), monetary=("order_revenue", "sum"))
        )

        # Seasonality features: revenue share by season (Winter/Spring/Summer/Autumn).
        m = orders["order_date"].dt.month.fillna(0).astype(int)
        orders["__season"] = np.select(
            [
                m.isin([12, 1, 2]),
                m.isin([3, 4, 5]),
                m.isin([6, 7, 8]),
                m.isin([9, 10, 11]),
            ],
            ["Winter", "Spring", "Summer", "Autumn"],
            default="Unknown",
        )
        season_rev = (
            orders.groupby([user_col_resolved, "__season"], dropna=False)["order_revenue"]
            .sum()
            .unstack(fill_value=0.0)
        )
        seasons = ["Winter", "Spring", "Summer", "Autumn"]
        for s in seasons:
            if s not in season_rev.columns:
                season_rev[s] = 0.0
        season_rev = season_rev[seasons]

        # Align all user-level features to the same set of customers (those with positive revenue).
        active_users = pivot.index
        user_metrics = user_metrics.reindex(active_users).fillna(0.0)
        season_rev = season_rev.reindex(active_users).fillna(0.0)

        season_share = season_rev.div(user_metrics["monetary"].replace(0.0, np.nan), axis=0).fillna(0.0)
        season_feature_by_name = {s: f"season_share_{s.lower()}" for s in seasons}
        season_share.columns = [season_feature_by_name[s] for s in seasons]

        df_users = share.copy()
        df_users["frequency"] = user_metrics["frequency"].astype(float)
        df_users["monetary"] = user_metrics["monetary"].astype(float)
        for col in season_share.columns:
            df_users[col] = season_share[col].astype(float)
        df_users.insert(0, "user_id", df_users.index.astype(str))

        # Build feature definitions for UI.
        feature_definitions: List[Dict[str, Any]] = [
            {
                "name": feature_by_cat[c],
                "description": f"Share of the customer's revenue in category '{c}' (0-1) within the analysis window.",
            }
            for c in cats
        ]
        feature_definitions.extend(
            [
                {"name": "frequency", "description": "Number of unique orders in the analysis window."},
                {"name": "monetary", "description": "Total revenue in the analysis window."},
            ]
        )
        feature_definitions.extend(
            [
                {
                    "name": season_feature_by_name["Winter"],
                    "description": "Share of the customer's revenue that occurs in Winter (Dec-Feb).",
                },
                {
                    "name": season_feature_by_name["Spring"],
                    "description": "Share of the customer's revenue that occurs in Spring (Mar-May).",
                },
                {
                    "name": season_feature_by_name["Summer"],
                    "description": "Share of the customer's revenue that occurs in Summer (Jun-Aug).",
                },
                {
                    "name": season_feature_by_name["Autumn"],
                    "description": "Share of the customer's revenue that occurs in Autumn (Sep-Nov).",
                },
            ]
        )

        # Optional: attach overall category distribution for transparency (cleaning step).
        cat_overview: List[Dict[str, Any]] = []
        overall_by_cat = df_work.groupby("__category")["__amount"].agg(["sum", "count"]).reset_index()
        total_amt = float(overall_by_cat["sum"].sum()) if not overall_by_cat.empty else 0.0
        for _, row in overall_by_cat.iterrows():
            c = str(row["__category"])
            rev = float(row["sum"])
            cnt = int(row["count"])
            cat_overview.append(
                {
                    "category": c,
                    "revenue": rev,
                    "revenue_share_pct": (rev / total_amt * 100.0) if total_amt else 0.0,
                    "line_items": cnt,
                }
            )
        cat_overview.sort(key=lambda x: float(x.get("revenue") or 0.0), reverse=True)

        season_overview: List[Dict[str, Any]] = []
        overall_by_season = orders.groupby("__season")["order_revenue"].agg(["sum", "count"]).reset_index()
        total_season_amt = float(overall_by_season["sum"].sum()) if not overall_by_season.empty else 0.0
        for _, row in overall_by_season.iterrows():
            s = str(row["__season"])
            if s == "Unknown":
                continue
            rev = float(row["sum"])
            cnt = int(row["count"])
            season_overview.append(
                {
                    "season": s,
                    "revenue": rev,
                    "revenue_share_pct": (rev / total_season_amt * 100.0) if total_season_amt else 0.0,
                    "orders": cnt,
                }
            )
        season_overview.sort(key=lambda x: float(x.get("revenue") or 0.0), reverse=True)

        result = segment_customers_kmeans(
            df_users,
            selected_features=[d["name"] for d in feature_definitions],
            k_range=tuple(req.k_range),
            random_seed=int(req.random_seed),
            outlier_threshold=req.outlier_threshold,
            user_id_column="user_id",
        )

        # Replace generic label suggestion with category-driven labels (English).
        if isinstance(result.get("clusters_summary"), list):
            for item in result["clusters_summary"]:
                chars = (item or {}).get("characteristics") or {}
                share_means: List[Tuple[str, float]] = []
                for fname, cat in cat_by_feature.items():
                    m = chars.get(fname, {}).get("mean")
                    if m is None:
                        continue
                    try:
                        share_means.append((cat, float(m)))
                    except Exception:
                        continue
                share_means.sort(key=lambda x: x[1], reverse=True)
                label = "Mixed Basket"
                if share_means:
                    top_cat, top_share = share_means[0]
                    if top_share >= 0.55:
                        label = f"{top_cat} Focused"
                    elif len(share_means) >= 2:
                        label = f"Mixed ({top_cat} + {share_means[1][0]})"
                    else:
                        label = f"{top_cat} Leaning"

                season_keys = [season_feature_by_name[s] for s in seasons]
                season_means: List[Tuple[str, float]] = []
                for sk in season_keys:
                    sm = chars.get(sk, {}).get("mean")
                    if sm is None:
                        continue
                    try:
                        season_means.append((sk, float(sm)))
                    except Exception:
                        continue
                season_means.sort(key=lambda x: x[1], reverse=True)
                if season_means:
                    top_key, top_share = season_means[0]
                    if top_share >= 0.6:
                        season_name = top_key.replace("season_share_", "").replace("_", " ").title()
                        label = f"{label} · {season_name} Heavy"
                item["label_suggestion"] = label

        feature_defs_used = [
            d
            for d in feature_definitions
            if d.get("name") in (result.get("model_info") or {}).get("features_used", [])
        ]

        # AI Insight (optional).
        insight_md: Optional[str] = None
        enable_ai_env = os.getenv("SEGMENTATION_ENABLE_AI_INSIGHT")
        if req.enable_ai_insight is None:
            if enable_ai_env is None:
                enable_ai = bool(_get_volcengine_api_key())
            else:
                enable_ai = str(enable_ai_env).lower() in {"1", "true", "yes", "on"}
        else:
            enable_ai = bool(req.enable_ai_insight)

        try:
            if enable_ai:
                insight_md = _segmentation_generate_insight_markdown_ai(
                    dataset_id=dataset_id,
                    feature_mode="category_mix",
                    feature_definitions=feature_defs_used,
                    clusters_summary=result.get("clusters_summary") or [],
                )
            else:
                insight_md = _segmentation_generate_insight_markdown(
                    feature_definitions=feature_defs_used,
                    clusters_summary=result.get("clusters_summary") or [],
                )
        except Exception as e:
            print(f"⚠️  Segmentation AI insight generation failed; fallback to template: {e}")
            insight_md = _segmentation_generate_insight_markdown(
                feature_definitions=feature_defs_used,
                clusters_summary=result.get("clusters_summary") or [],
            )

        result["feature_mode"] = "category_mix"
        result["feature_definitions"] = feature_defs_used
        result["category_overview"] = cat_overview
        result["season_overview"] = season_overview
        result["insight_markdown"] = insight_md
    else:
        raise HTTPException(status_code=400, detail="feature_mode must be 'category_mix' or 'rfm'")

    if result.get("status") != "success":
        raise HTTPException(status_code=400, detail=str(result.get("message") or "Segmentation failed"))

    # Attach dataset-scoped metadata for UI display.
    result["dataset_id"] = dataset_id
    result["reference_date"] = str(reference_date)
    result["time_window_days"] = window_days
    return result


# -----------------------------
# Sales Forecasting (Time Series)
# -----------------------------

SALES_HIERARCHY_RULES: List[Dict[str, Any]] = [
    {
        "category": "Seasonal",
        "keywords": ["CHRISTMAS", "XMAS", "HALLOWEEN", "EASTER", "VALENTINE"],
        "sub_categories": [
            {"name": "Christmas", "keywords": ["CHRISTMAS", "XMAS"]},
            {"name": "Halloween", "keywords": ["HALLOWEEN"]},
            {"name": "Easter", "keywords": ["EASTER"]},
            {"name": "Valentine", "keywords": ["VALENTINE"]},
            {"name": "Other", "keywords": []},
        ],
    },
    {
        "category": "Bags",
        "keywords": ["BAG", "TOTE"],
        "sub_categories": [
            {"name": "Lunch Bags", "keywords": ["LUNCH"]},
            {"name": "Jumbo Bags", "keywords": ["JUMBO"]},
            {"name": "Gift Bags", "keywords": ["GIFT"]},
            {"name": "Other", "keywords": []},
        ],
    },
    {
        "category": "Kitchenware",
        "keywords": ["MUG", "CUP", "PLATE", "BOWL", "PANTRY", "KITCHEN", "TEA", "COFFEE", "CAKE", "LUNCH"],
        "sub_categories": [
            {"name": "Mugs & Cups", "keywords": ["MUG", "CUP"]},
            {"name": "Lunchware", "keywords": ["LUNCH"]},
            {"name": "Pantry", "keywords": ["PANTRY"]},
            {"name": "Cakeware", "keywords": ["CAKE"]},
            {"name": "Other", "keywords": []},
        ],
    },
    {
        "category": "Stationery",
        "keywords": ["CARD", "PAPER", "NOTE", "TAG", "WRAP", "STICKER"],
        "sub_categories": [
            {"name": "Cards", "keywords": ["CARD"]},
            {"name": "Wrapping", "keywords": ["WRAP"]},
            {"name": "Tags", "keywords": ["TAG"]},
            {"name": "Paper & Notes", "keywords": ["PAPER", "NOTE"]},
            {"name": "Other", "keywords": []},
        ],
    },
    {
        "category": "Storage",
        "keywords": ["BOX", "TIN", "BASKET", "JAR", "BOTTLE", "CASE", "CASES"],
        "sub_categories": [
            {"name": "Boxes", "keywords": ["BOX"]},
            {"name": "Tins", "keywords": ["TIN"]},
            {"name": "Cases", "keywords": ["CASE", "CASES"]},
            {"name": "Bottles & Jars", "keywords": ["BOTTLE", "JAR"]},
            {"name": "Other", "keywords": []},
        ],
    },
    {
        "category": "Home Decor",
        "keywords": ["HEART", "HANGING", "T-LIGHT", "CANDLE", "HOLDER", "SIGN", "DECORATION", "LIGHT", "METAL", "WOODEN", "GLASS", "DOORMAT"],
        "sub_categories": [
            {"name": "Lights", "keywords": ["LIGHT"]},
            {"name": "Candles", "keywords": ["CANDLE", "T-LIGHT"]},
            {"name": "Hanging Decor", "keywords": ["HANGING"]},
            {"name": "Signs", "keywords": ["SIGN"]},
            {"name": "Other", "keywords": []},
        ],
    },
    {
        "category": "Toys",
        "keywords": ["DOLLY", "SPACEBOY"],
        "sub_categories": [
            {"name": "Dolly", "keywords": ["DOLLY"]},
            {"name": "Spaceboy", "keywords": ["SPACEBOY"]},
            {"name": "Other", "keywords": []},
        ],
    },
    {"category": "Other", "keywords": [], "sub_categories": [{"name": "Other", "keywords": []}]},
]


def _sales_hierarchy_norm(s: str) -> str:
    return (s or "").strip().lower()


def _sales_hierarchy_get_category_rule(category: str) -> Optional[Dict[str, Any]]:
    cat = _sales_hierarchy_norm(category)
    for r in SALES_HIERARCHY_RULES:
        if _sales_hierarchy_norm(str(r.get("category", ""))) == cat:
            return r
    return None


def _sales_hierarchy_build_keyword_regex(keywords: List[str]) -> str:
    if not keywords:
        return ""

    parts: List[str] = []
    for kw in keywords:
        kwu = str(kw or "").upper().strip()
        if not kwu:
            continue
        # Use word boundaries for short alnum keywords to reduce false positives.
        if kwu.isalnum() and len(kwu) <= 4:
            parts.append(rf"\b{re.escape(kwu)}\b")
        else:
            parts.append(re.escape(kwu))
    return "|".join(parts)


def _sales_hierarchy_category_feature_key(category: str) -> str:
    """
    Convert a human-readable category name into a stable feature key.

    Example: "Home Decor" -> "cat_share_home_decor"
    """
    base = re.sub(r"[^a-z0-9]+", "_", _sales_hierarchy_norm(category)).strip("_")
    if not base:
        base = "other"
    return f"cat_share_{base}"


_SALES_HIERARCHY_CATEGORY_PATTERNS: Optional[List[Tuple[str, str]]] = None


def _sales_hierarchy_category_patterns() -> List[Tuple[str, str]]:
    """Build (category, regex_pattern) list once for fast classification."""
    global _SALES_HIERARCHY_CATEGORY_PATTERNS
    if isinstance(_SALES_HIERARCHY_CATEGORY_PATTERNS, list):
        return _SALES_HIERARCHY_CATEGORY_PATTERNS

    patterns: List[Tuple[str, str]] = []
    for r in SALES_HIERARCHY_RULES:
        cat = str(r.get("category") or "").strip()
        if not cat:
            continue
        if _sales_hierarchy_norm(cat) == "other":
            continue
        pat = _sales_hierarchy_build_keyword_regex(list(r.get("keywords") or []))
        if pat:
            patterns.append((cat, pat))

    _SALES_HIERARCHY_CATEGORY_PATTERNS = patterns
    return patterns


def _sales_hierarchy_assign_category(desc: pd.Series) -> pd.Series:
    """Assign a top-level category to each description using keyword rules."""
    desc_upper = desc.astype(str).str.upper()
    out = pd.Series("Other", index=desc_upper.index, dtype="object")

    remaining = pd.Series(True, index=desc_upper.index)
    for cat, pat in _sales_hierarchy_category_patterns():
        if not bool(remaining.any()):
            break
        m = remaining & desc_upper.str.contains(pat, regex=True, na=False)
        if bool(m.any()):
            out.loc[m] = cat
            remaining.loc[m] = False

    return out


def _sales_hierarchy_assign_sub_category(desc: pd.Series, *, category_rule: Dict[str, Any]) -> pd.Series:
    """Assign a sub-category within a category using the same first-match-wins strategy."""
    desc_upper = desc.astype(str).str.upper()
    out = pd.Series("Other", index=desc_upper.index, dtype="object")

    sub_rules = list(category_rule.get("sub_categories") or [])
    remaining = pd.Series(True, index=desc_upper.index)
    for sr in sub_rules:
        name = str(sr.get("name") or "").strip()
        if not name:
            continue
        if _sales_hierarchy_norm(name) == "other":
            continue
        pat = _sales_hierarchy_build_keyword_regex(list(sr.get("keywords") or []))
        if not pat:
            continue
        if not bool(remaining.any()):
            break
        m = remaining & desc_upper.str.contains(pat, regex=True, na=False)
        if bool(m.any()):
            out.loc[m] = name
            remaining.loc[m] = False

    return out


def _sales_hierarchy_masks_for_chunk(
    *,
    desc_upper: pd.Series,
    category: str,
    sub_category: Optional[str],
) -> pd.Series:
    """Return a boolean mask for rows that belong to (category, sub_category) based on keyword rules."""
    rule = _sales_hierarchy_get_category_rule(category)
    if not rule:
        # Unknown category -> match nothing.
        return pd.Series(False, index=desc_upper.index)

    canonical_cat = str(rule.get("category") or "").strip() or str(category).strip()
    assigned_cat = _sales_hierarchy_assign_category(desc_upper)
    cat_mask = assigned_cat.astype(str).str.lower() == _sales_hierarchy_norm(canonical_cat)

    if not sub_category:
        return cat_mask

    sub_rules = list(rule.get("sub_categories") or [])
    sub_norm = _sales_hierarchy_norm(str(sub_category or ""))
    sub_canonical = next((str(sr.get("name") or "").strip() for sr in sub_rules if _sales_hierarchy_norm(str(sr.get("name") or "")) == sub_norm), "")
    if not sub_canonical:
        return pd.Series(False, index=desc_upper.index)

    assigned_sub = _sales_hierarchy_assign_sub_category(desc_upper, category_rule=rule)
    sub_mask = assigned_sub.astype(str).str.lower() == _sales_hierarchy_norm(sub_canonical)
    return cat_mask & sub_mask


@router.get("/{dataset_id}/sales-hierarchy")
def get_sales_hierarchy(dataset_id: str):
    """Return keyword-based product hierarchy options for Sales Forecast."""
    config = load_datasets_config()
    dataset_config = next((d for d in config.get("datasets", []) if d["id"] == dataset_id), None)
    if not dataset_config:
        raise HTTPException(status_code=404, detail="Dataset not found")

    categories = []
    for r in SALES_HIERARCHY_RULES:
        categories.append(
            {
                "category": r.get("category"),
                "sub_categories": [sr.get("name") for sr in (r.get("sub_categories") or [])],
            }
        )

    return {
        "success": True,
        "data": {
            "dataset_id": dataset_id,
            "levels": ["category", "sub_category", "sku"],
            "categories": categories,
        },
    }


@router.get("/{dataset_id}/sales-hierarchy/skus")
def list_sales_hierarchy_skus(
    dataset_id: str,
    category: str = Query(..., min_length=1),
    sub_category: Optional[str] = Query(None),
    limit: int = Query(200, ge=1, le=2000),
):
    """List top SKUs (StockCode) for a given (category, sub_category) by total revenue."""
    config = load_datasets_config()
    dataset_config = next((d for d in config.get("datasets", []) if d["id"] == dataset_id), None)
    if not dataset_config:
        raise HTTPException(status_code=404, detail="Dataset not found")

    csv_path = _get_dataset_csv_path(dataset_id, dataset_config)
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail=f"CSV file not found: {csv_path}")

    encoding, header_cols = _read_csv_header(csv_path)
    date_col_req = dataset_config.get("time_column") or "InvoiceDate"
    entity_col_req = "StockCode"
    desc_col_req = "Description"
    qty_col_req = "Quantity"
    price_col_req = "UnitPrice"

    entity_col = _resolve_csv_column(header_cols, str(entity_col_req))
    desc_col = _resolve_csv_column(header_cols, str(desc_col_req))
    qty_col = _resolve_csv_column(header_cols, str(qty_col_req))
    price_col = _resolve_csv_column(header_cols, str(price_col_req))
    _ = _resolve_csv_column(header_cols, str(date_col_req))  # validate existence (not used)

    # Accumulate revenue by SKU (StockCode).
    sums: Dict[str, float] = {}
    desc_by_sku: Dict[str, str] = {}

    usecols = sorted({entity_col, desc_col, qty_col, price_col})
    try:
        reader = pd.read_csv(csv_path, encoding=encoding, usecols=usecols, low_memory=False, chunksize=200_000)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {e}")

    for chunk in reader:
        if chunk.empty:
            continue

        desc_upper = chunk[desc_col].astype(str).str.upper()
        mask = _sales_hierarchy_masks_for_chunk(desc_upper=desc_upper, category=category, sub_category=sub_category)
        if not bool(mask.any()):
            continue

        sub = chunk.loc[mask].copy()
        if sub.empty:
            continue

        sku = sub[entity_col].astype(str).str.strip()
        qty = pd.to_numeric(sub[qty_col], errors="coerce").fillna(0.0).astype(float)
        price = pd.to_numeric(sub[price_col], errors="coerce").fillna(0.0).astype(float)
        revenue = (qty * price).clip(lower=0.0)

        grp = revenue.groupby(sku).sum()
        for k, v in grp.items():
            kk = str(k).strip()
            if not kk:
                continue
            sums[kk] = sums.get(kk, 0.0) + float(v)

        # Keep a representative description for display (first non-empty seen).
        for kk, dd in zip(sku.tolist(), sub[desc_col].astype(str).tolist()):
            key = str(kk).strip()
            if key and key not in desc_by_sku and dd:
                desc_by_sku[key] = str(dd)

    items = [
        {"sku": k, "description": desc_by_sku.get(k), "total_revenue": float(v)}
        for k, v in sums.items()
    ]
    items.sort(key=lambda x: float(x.get("total_revenue") or 0.0), reverse=True)
    items = items[: int(limit)]

    return {"success": True, "data": items, "total": len(items)}


class SalesHistoryPoint(BaseModel):
    date: str
    sales: float
    is_holiday: int = 0


class SalesForecastScenario(BaseModel):
    """
    Lightweight scenario simulation knobs.

    This does NOT retrain the ML model. It applies business adjustments on top of the baseline forecast.
    """

    enabled: bool = True

    # Price change as a percentage (e.g. -0.10 means -10%).
    price_change_pct: float = Field(0.0, ge=-0.99, le=0.99)
    # Price elasticity of demand (usually negative). %ΔQ = elasticity * %ΔP.
    price_elasticity: float = Field(-1.2)

    # Total marketing budget for the horizon (currency units). Added as incremental revenue using ROI.
    marketing_budget: float = Field(0.0, ge=0.0)
    marketing_roi: float = Field(3.0, ge=0.0)

    # Promotion factor applied to the scenario line (baseline uses request.promotion_factor).
    promotion_factor: float = Field(1.0, gt=0.0)

    # Overall market growth assumption over the horizon (e.g. 0.02 means +2%).
    market_growth_pct: float = Field(0.0, ge=-0.99, le=10.0)


class SalesForecastRequest(BaseModel):
    model_id: str = Field(..., min_length=1)
    # Backward-compatible SKU identifier. Prefer the hierarchy fields below.
    target_entity_id: Optional[str] = Field(None, description="Target entity id (e.g. SKU / StockCode)")
    forecast_horizon: int = Field(7, ge=1, le=365)

    # When historical_context is not provided, the service loads it from the dataset CSV.
    context_window_days: int = Field(60, ge=7, le=3650)
    historical_context: Optional[List[SalesHistoryPoint]] = None

    # Dataset mapping overrides (defaults are resolved from datasets.json + common names).
    target_entity_column: Optional[str] = None
    date_column: Optional[str] = None
    description_column: Optional[str] = None
    quantity_column: Optional[str] = None
    unit_price_column: Optional[str] = None

    # Scope selection (keyword-based product hierarchy).
    scope_level: str = Field("sku", description="category | sub_category | sku")
    category: Optional[str] = None
    sub_category: Optional[str] = None
    sku_id: Optional[str] = None

    # Business knobs
    sales_metric: str = Field("revenue", description="quantity | revenue")
    promotion_factor: float = Field(1.0, gt=0)
    holiday_country: Optional[str] = Field("GB", description="holidays package country code, e.g. GB/US/CN")
    rounding: str = Field("none", description="round | floor | none")
    clip_negative: bool = Field(True)

    # Optional scenario simulation (baseline vs simulated forecast).
    scenario: Optional[SalesForecastScenario] = None


def _sales_forecast_cache_key(dataset_id: str, metric: str) -> str:
    return f"{dataset_id}:{metric}"


def _sales_forecast_is_holiday(ts: pd.Timestamp, country_code: Optional[str]) -> int:
    code = (country_code or "").strip().upper()
    if not code:
        return 0
    try:
        import holidays  # type: ignore
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Missing dependency 'holidays' - cannot compute holiday feature: {e}")

    cache_key = f"{code}:{int(ts.year)}"
    with SALES_FORECAST_HOLIDAY_LOCK:
        cached = SALES_FORECAST_HOLIDAY_CACHE.get(cache_key)
        if isinstance(cached, set):
            return int(ts.date() in cached)

    try:
        cal_obj = holidays.country_holidays(code, years=[int(ts.year)])
        cal_set = set(cal_obj.keys())
    except Exception:
        cal_set = set()

    with SALES_FORECAST_HOLIDAY_LOCK:
        SALES_FORECAST_HOLIDAY_CACHE[cache_key] = cal_set
    return int(ts.date() in cal_set)


def _sales_forecast_load_joblib_model(model_path: str) -> Any:
    with SALES_FORECAST_MODEL_CACHE_LOCK:
        cached = SALES_FORECAST_MODEL_CACHE.get(model_path)
        if cached:
            return cached["model"]

    try:
        model = joblib.load(model_path)
    except ModuleNotFoundError as e:
        # Common demo pitfall: the API is started from a venv without LightGBM installed,
        # so the pickled estimator can't be unpickled.
        if getattr(e, "name", None) == "lightgbm":
            raise HTTPException(
                status_code=500,
                detail=(
                    "Missing dependency 'lightgbm'. This model was trained with LightGBM, "
                    "so the API must run in an environment that has LightGBM installed. "
                    "Tip: start the API via 'prediction_app/start_api.sh' (it prefers '.venv_py312')."
                ),
            )
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")

    with SALES_FORECAST_MODEL_CACHE_LOCK:
        SALES_FORECAST_MODEL_CACHE[model_path] = {"created_at": datetime.now(), "model": model}
        # simple FIFO eviction
        if len(SALES_FORECAST_MODEL_CACHE) > SALES_FORECAST_MODEL_CACHE_MAX_ITEMS:
            oldest_key = min(
                SALES_FORECAST_MODEL_CACHE.keys(),
                key=lambda k: SALES_FORECAST_MODEL_CACHE[k]["created_at"],
            )
            SALES_FORECAST_MODEL_CACHE.pop(oldest_key, None)

    return model


def _sales_forecast_resolve_model(model_id: str) -> tuple[Optional[Any], str]:
    """Resolve model_id to a loadable joblib model (or builtin baseline)."""
    mid = (model_id or "").strip()
    if not mid:
        raise HTTPException(status_code=400, detail="model_id is required")

    builtin_ids = {"baseline_moving_average", "baseline"}
    if mid in builtin_ids:
        return None, "baseline_moving_average"

    # Try deployed model registry first: model_name[:version] or model_name@version.
    name = mid
    version: Optional[str] = None
    if ":" in mid:
        name, version = mid.split(":", 1)
    elif "@" in mid:
        name, version = mid.split("@", 1)
    name = (name or "").strip()
    version = (version or "").strip() or None

    try:
        from api.deployment import ModelDeploymentManager
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load deployment manager: {e}")

    mgr = ModelDeploymentManager()
    info = mgr.get_model_info(name)
    if isinstance(info, dict) and "current_version" in info:
        use_ver = version or str(info.get("current_version"))
        ver_info = info.get(use_ver) if isinstance(info.get(use_ver), dict) else None
        model_path = ver_info.get("path") if isinstance(ver_info, dict) else None
        if isinstance(model_path, str) and model_path and Path(model_path).exists():
            return _sales_forecast_load_joblib_model(model_path), f"{name}:{use_ver}"

    # Repo-level model assets directory: models/{name}/{version}/model.pkl
    project_root_local = Path(__file__).parent.parent
    models_root = project_root_local.parent / "models"
    name_dir = models_root / name
    if name_dir.exists() and name_dir.is_dir():
        if version:
            asset_path = name_dir / version / "model.pkl"
            if asset_path.exists():
                return _sales_forecast_load_joblib_model(str(asset_path)), f"{name}:{version}"
        else:
            candidates = sorted(name_dir.glob("*/model.pkl"), key=lambda p: p.parent.name)
            if candidates:
                asset_path = candidates[-1]
                use_ver = asset_path.parent.name
                return _sales_forecast_load_joblib_model(str(asset_path)), f"{name}:{use_ver}"

    # Fallback: local training models directory
    training_path = project_root_local / "training" / "models" / f"{mid}_model.pkl"
    if training_path.exists():
        return _sales_forecast_load_joblib_model(str(training_path)), mid

    raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")


def _sales_forecast_unwrap_estimator(model_obj: Any) -> Any:
    """
    Prefer objects that already implement predict() (e.g. wrappers that apply target transforms).

    Fallback to unwrapping `obj.model` when the outer object does not implement predict().
    """
    if hasattr(model_obj, "predict"):
        return model_obj
    if hasattr(model_obj, "model") and hasattr(getattr(model_obj, "model"), "predict"):
        return getattr(model_obj, "model")
    return model_obj


def _sales_forecast_predict_one(model_obj: Any, features: Dict[str, Any]) -> float:
    estimator = _sales_forecast_unwrap_estimator(model_obj)
    if not hasattr(estimator, "predict"):
        raise HTTPException(status_code=400, detail="Model object does not support predict()")

    X = pd.DataFrame([features])
    # Align feature order when the estimator was trained with pandas DataFrame.
    if hasattr(estimator, "feature_names_in_"):
        names = [str(n) for n in getattr(estimator, "feature_names_in_")]
        for n in names:
            if n not in X.columns:
                X[n] = 0
        X = X[names]

    try:
        y = estimator.predict(X)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {e}")

    if isinstance(y, (list, tuple)) and len(y) > 0:
        return float(y[0])
    if hasattr(y, "__len__") and len(y) > 0:  # numpy array
        return float(y[0])
    return float(y)


def _sales_forecast_daily_series_from_csv(
    *,
    csv_path: Path,
    encoding: str,
    date_col: str,
    entity_col: str,
    target_entity_id: str,
    metric: str,
    qty_col: str,
    price_col: Optional[str],
    chunksize: int = 200_000,
) -> pd.Series:
    target = str(target_entity_id).strip()
    daily_sum: Dict[pd.Timestamp, float] = {}

    usecols = {date_col, entity_col, qty_col}
    if metric == "revenue":
        if not price_col:
            raise HTTPException(status_code=400, detail="sales_metric=revenue requires unit_price_column")
        usecols.add(price_col)

    try:
        reader = pd.read_csv(
            csv_path,
            encoding=encoding,
            usecols=sorted(usecols),
            low_memory=False,
            chunksize=chunksize,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {e}")

    for chunk in reader:
        if chunk.empty:
            continue

        # Normalize entity column for matching.
        ent = chunk[entity_col].astype(str).str.strip()
        mask = ent == target
        if not bool(mask.any()):
            continue

        sub = chunk.loc[mask].copy()
        sub[date_col] = pd.to_datetime(sub[date_col], errors="coerce")
        sub = sub.dropna(subset=[date_col])
        if sub.empty:
            continue

        day = sub[date_col].dt.floor("D")

        qty = pd.to_numeric(sub[qty_col], errors="coerce").fillna(0.0).astype(float)
        if metric == "revenue":
            price = pd.to_numeric(sub[price_col], errors="coerce").fillna(0.0).astype(float)  # type: ignore[index]
            val = qty * price
        else:
            val = qty

        grp = val.groupby(day).sum()
        for d, v in grp.items():
            daily_sum[pd.Timestamp(d)] = daily_sum.get(pd.Timestamp(d), 0.0) + float(v)

    if not daily_sum:
        return pd.Series(dtype="float64")

    s = pd.Series(daily_sum).sort_index()
    # Ensure continuous daily index (fill missing with 0).
    full_idx = pd.date_range(start=s.index.min(), end=s.index.max(), freq="D")
    return s.reindex(full_idx, fill_value=0.0).astype(float)


def _sales_forecast_daily_series_from_csv_by_hierarchy(
    *,
    csv_path: Path,
    encoding: str,
    date_col: str,
    desc_col: str,
    category: str,
    sub_category: Optional[str],
    metric: str,
    qty_col: str,
    price_col: Optional[str],
    chunksize: int = 200_000,
) -> pd.Series:
    """Aggregate daily sales for a keyword-based (category, sub_category) scope."""
    daily_sum: Dict[pd.Timestamp, float] = {}

    usecols = {date_col, desc_col, qty_col}
    if metric == "revenue":
        if not price_col:
            raise HTTPException(status_code=400, detail="sales_metric=revenue requires unit_price_column")
        usecols.add(price_col)

    try:
        reader = pd.read_csv(
            csv_path,
            encoding=encoding,
            usecols=sorted(usecols),
            low_memory=False,
            chunksize=chunksize,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {e}")

    for chunk in reader:
        if chunk.empty:
            continue

        desc_upper = chunk[desc_col].astype(str).str.upper()
        mask = _sales_hierarchy_masks_for_chunk(desc_upper=desc_upper, category=category, sub_category=sub_category)
        if not bool(mask.any()):
            continue

        sub = chunk.loc[mask].copy()
        if sub.empty:
            continue

        sub[date_col] = pd.to_datetime(sub[date_col], errors="coerce")
        sub = sub.dropna(subset=[date_col])
        if sub.empty:
            continue

        day = sub[date_col].dt.floor("D")
        qty = pd.to_numeric(sub[qty_col], errors="coerce").fillna(0.0).astype(float)
        if metric == "revenue":
            price = pd.to_numeric(sub[price_col], errors="coerce").fillna(0.0).astype(float)  # type: ignore[index]
            val = qty * price
        else:
            val = qty

        # Sales can't be negative; clip returns to 0 (consistent with post-processing).
        val = val.clip(lower=0.0)

        grp = val.groupby(day).sum()
        for d, v in grp.items():
            daily_sum[pd.Timestamp(d)] = daily_sum.get(pd.Timestamp(d), 0.0) + float(v)

    if not daily_sum:
        return pd.Series(dtype="float64")

    s = pd.Series(daily_sum).sort_index()
    full_idx = pd.date_range(start=s.index.min(), end=s.index.max(), freq="D")
    return s.reindex(full_idx, fill_value=0.0).astype(float)


def _sales_forecast_global_daily_mean_from_csv(
    *,
    cache_key: str,
    csv_path: Path,
    encoding: str,
    date_col: str,
    metric: str,
    qty_col: str,
    price_col: Optional[str],
    chunksize: int = 200_000,
) -> float:
    with SALES_FORECAST_FALLBACK_MEAN_LOCK:
        cached = SALES_FORECAST_FALLBACK_MEAN_CACHE.get(cache_key)
        if isinstance(cached, (int, float)):
            return float(cached)

    daily_sum: Dict[pd.Timestamp, float] = {}
    usecols = {date_col, qty_col}
    if metric == "revenue":
        if not price_col:
            raise HTTPException(status_code=400, detail="sales_metric=revenue requires unit_price_column")
        usecols.add(price_col)

    try:
        reader = pd.read_csv(
            csv_path,
            encoding=encoding,
            usecols=sorted(usecols),
            low_memory=False,
            chunksize=chunksize,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {e}")

    for chunk in reader:
        if chunk.empty:
            continue
        chunk[date_col] = pd.to_datetime(chunk[date_col], errors="coerce")
        chunk = chunk.dropna(subset=[date_col])
        if chunk.empty:
            continue

        day = chunk[date_col].dt.floor("D")
        qty = pd.to_numeric(chunk[qty_col], errors="coerce").fillna(0.0).astype(float)
        if metric == "revenue":
            price = pd.to_numeric(chunk[price_col], errors="coerce").fillna(0.0).astype(float)  # type: ignore[index]
            val = qty * price
        else:
            val = qty

        grp = val.groupby(day).sum()
        for d, v in grp.items():
            daily_sum[pd.Timestamp(d)] = daily_sum.get(pd.Timestamp(d), 0.0) + float(v)

    if not daily_sum:
        with SALES_FORECAST_FALLBACK_MEAN_LOCK:
            SALES_FORECAST_FALLBACK_MEAN_CACHE[cache_key] = 0.0
        return 0.0

    s = pd.Series(daily_sum).sort_index()
    # Use mean of daily totals over the observed date range (missing days treated as 0).
    full_idx = pd.date_range(start=s.index.min(), end=s.index.max(), freq="D")
    mean = float(s.reindex(full_idx, fill_value=0.0).mean())

    with SALES_FORECAST_FALLBACK_MEAN_LOCK:
        SALES_FORECAST_FALLBACK_MEAN_CACHE[cache_key] = mean
    return mean


def _sales_forecast_compute_features(
    *,
    ts: pd.Timestamp,
    sales_history: List[float],
    holiday_country: Optional[str],
) -> Dict[str, Any]:
    # Calendar features
    feats: Dict[str, Any] = {
        "month": int(ts.month),
        "day_of_week": int(ts.dayofweek),
        "is_weekend": int(ts.dayofweek >= 5),
        "day_of_year": int(ts.dayofyear),
        "is_holiday": _sales_forecast_is_holiday(ts, holiday_country),
    }

    lags = [1, 7, 14, 28]
    rolls = [7, 14, 28]
    n = len(sales_history)
    for lag in lags:
        feats[f"lag_{lag}"] = float(sales_history[-lag]) if n >= lag else 0.0

    arr = np.asarray(sales_history, dtype=float) if sales_history else np.asarray([], dtype=float)
    for w in rolls:
        window = arr[-w:] if arr.size >= 1 else arr
        feats[f"rolling_mean_{w}"] = float(window.mean()) if window.size else 0.0
        feats[f"rolling_std_{w}"] = float(window.std(ddof=0)) if window.size else 0.0

    # Bounded momentum feature to avoid extreme values when lag_7 is ~0 (common in sparse series).
    lag7 = float(feats["lag_7"])
    lag1 = float(feats["lag_1"])
    feats["mom_7"] = float((lag1 - lag7) / (abs(lag1) + abs(lag7) + 1e-6))
    return feats


def _sales_forecast_post_process(
    *,
    value: float,
    promotion_factor: float,
    clip_negative: bool,
    rounding: str,
) -> Any:
    v = float(value) * float(promotion_factor)
    if clip_negative and v < 0:
        v = 0.0
    r = (rounding or "round").lower()
    if r == "none":
        return float(v)
    if r == "floor":
        return int(math.floor(v))
    return int(round(v))


@router.post("/{dataset_id}/sales-forecast")
def run_sales_forecast(dataset_id: str, req: SalesForecastRequest):
    """
    Sales Forecasting Service (dataset-scoped)

    - Select dataset via path param (dataset_id)
    - Select model via request.model_id
    - Loads recent history from dataset CSV when historical_context is not provided
    """
    try:
        metric = (req.sales_metric or "revenue").lower()
        if metric not in {"quantity", "revenue"}:
            raise HTTPException(status_code=400, detail="sales_metric must be 'quantity' or 'revenue'")

        # 1) Resolve model (or baseline).
        model_obj, model_version = _sales_forecast_resolve_model(req.model_id)

        # 2) Load dataset config + history context.
        config = load_datasets_config()
        dataset_config = next((d for d in config.get("datasets", []) if d["id"] == dataset_id), None)
        if not dataset_config:
            raise HTTPException(status_code=404, detail="Dataset not found")

        scenario = req.scenario if (req.scenario and bool(req.scenario.enabled)) else None
        scope_level = (req.scope_level or "sku").strip().lower()

        history_points: List[Dict[str, Any]] = []
        reference_date: Optional[pd.Timestamp] = None
        scope_meta: Dict[str, Any] = {"level": scope_level}

        # For top-down allocation (sub_category), the model forecast is built on category history.
        model_history_sales: List[float] = []
        target_history_sales: List[float] = []
        allocation_share: float = 1.0

        if req.historical_context:
            df_hist = pd.DataFrame([p.model_dump() for p in req.historical_context])
            if df_hist.empty:
                raise HTTPException(status_code=400, detail="historical_context must not be empty")
            df_hist["date"] = pd.to_datetime(df_hist["date"], errors="coerce").dt.floor("D")
            df_hist = df_hist.dropna(subset=["date"])
            if df_hist.empty:
                raise HTTPException(status_code=400, detail="historical_context has no parseable 'date' values")
            df_hist["sales"] = pd.to_numeric(df_hist["sales"], errors="coerce").fillna(0.0).astype(float)
            df_hist["is_holiday"] = pd.to_numeric(df_hist.get("is_holiday"), errors="coerce").fillna(0).astype(int)
            df_hist = (
                df_hist.groupby("date", as_index=False)
                .agg(sales=("sales", "sum"), is_holiday=("is_holiday", "max"))
                .sort_values("date")
            )
            full_idx = pd.date_range(start=df_hist["date"].min(), end=df_hist["date"].max(), freq="D")
            df_hist = (
                df_hist.set_index("date")
                .reindex(full_idx)
                .fillna({"sales": 0.0, "is_holiday": 0})
                .reset_index()
                .rename(columns={"index": "date"})
            )
            reference_date = pd.Timestamp(df_hist["date"].max())
            start = reference_date - pd.Timedelta(days=int(req.context_window_days) - 1)
            df_hist = df_hist[df_hist["date"] >= start]
            for _, r in df_hist.iterrows():
                history_points.append(
                    {
                        "date": r["date"].strftime("%Y-%m-%d"),
                        "sales": float(max(0.0, float(r["sales"]))),
                        "is_holiday": int(r.get("is_holiday", 0)),
                    }
                )
            target_history_sales = [float(p["sales"]) for p in history_points]
            model_history_sales = list(target_history_sales)
            scope_meta.update({"level": "custom", "note": "historical_context provided"})
        else:
            csv_path = _get_dataset_csv_path(dataset_id, dataset_config)
            if not csv_path.exists():
                raise HTTPException(status_code=404, detail=f"CSV file not found: {csv_path}")

            encoding, header_cols = _read_csv_header(csv_path)
            date_col_req = req.date_column or dataset_config.get("time_column") or "InvoiceDate"
            entity_col_req = req.target_entity_column or "StockCode"
            desc_col_req = req.description_column or "Description"
            qty_col_req = req.quantity_column or "Quantity"
            price_col_req = req.unit_price_column or "UnitPrice"

            date_col = _resolve_csv_column(header_cols, str(date_col_req))
            entity_col = _resolve_csv_column(header_cols, str(entity_col_req))
            desc_col = _resolve_csv_column(header_cols, str(desc_col_req))
            qty_col = _resolve_csv_column(header_cols, str(qty_col_req))
            price_col = _resolve_csv_column(header_cols, str(price_col_req)) if metric == "revenue" else None

            if scope_level == "category":
                category = (req.category or "").strip()
                if not category:
                    raise HTTPException(status_code=400, detail="category is required when scope_level='category'")

                s_target = _sales_forecast_daily_series_from_csv_by_hierarchy(
                    csv_path=csv_path,
                    encoding=encoding,
                    date_col=date_col,
                    desc_col=desc_col,
                    category=category,
                    sub_category=None,
                    metric=metric,
                    qty_col=qty_col,
                    price_col=price_col,
                )
                s_model = s_target
                scope_meta.update({"level": "category", "category": category})

            elif scope_level in {"sub_category", "subcategory", "sub-category"}:
                category = (req.category or "").strip()
                sub_category = (req.sub_category or "").strip()
                if not category:
                    raise HTTPException(status_code=400, detail="category is required when scope_level='sub_category'")
                if not sub_category:
                    raise HTTPException(status_code=400, detail="sub_category is required when scope_level='sub_category'")

                s_cat = _sales_forecast_daily_series_from_csv_by_hierarchy(
                    csv_path=csv_path,
                    encoding=encoding,
                    date_col=date_col,
                    desc_col=desc_col,
                    category=category,
                    sub_category=None,
                    metric=metric,
                    qty_col=qty_col,
                    price_col=price_col,
                )

                s_sub_raw = _sales_forecast_daily_series_from_csv_by_hierarchy(
                    csv_path=csv_path,
                    encoding=encoding,
                    date_col=date_col,
                    desc_col=desc_col,
                    category=category,
                    sub_category=sub_category,
                    metric=metric,
                    qty_col=qty_col,
                    price_col=price_col,
                )

                if s_cat.empty:
                    s_model = s_cat
                    s_target = s_sub_raw
                else:
                    # Align target to category index so reference_date is consistent.
                    s_target = s_sub_raw.reindex(s_cat.index, fill_value=0.0).astype(float) if not s_sub_raw.empty else pd.Series(0.0, index=s_cat.index)
                    s_model = s_cat.astype(float)

                scope_meta.update({"level": "sub_category", "category": category, "sub_category": sub_category, "allocation": "top_down"})

            elif scope_level == "sku":
                sku_id = (req.sku_id or req.target_entity_id or "").strip()
                if not sku_id:
                    raise HTTPException(status_code=400, detail="sku_id (or target_entity_id) is required when scope_level='sku'")

                s_target = _sales_forecast_daily_series_from_csv(
                    csv_path=csv_path,
                    encoding=encoding,
                    date_col=date_col,
                    entity_col=entity_col,
                    target_entity_id=sku_id,
                    metric=metric,
                    qty_col=qty_col,
                    price_col=price_col,
                )
                s_model = s_target
                scope_meta.update({"level": "sku", "sku_id": sku_id})
            else:
                raise HTTPException(status_code=400, detail="scope_level must be one of: category | sub_category | sku")

            if s_model.empty:
                # Cold start: use dataset-level mean daily sales for the requested metric.
                mean_cache_key = _sales_forecast_cache_key(dataset_id, metric)
                fallback_mean = _sales_forecast_global_daily_mean_from_csv(
                    cache_key=mean_cache_key,
                    csv_path=csv_path,
                    encoding=encoding,
                    date_col=date_col,
                    metric=metric,
                    qty_col=qty_col,
                    price_col=price_col,
                )
                reference_date = None
                history_points = []
                model_history_sales = [float(fallback_mean)] * max(7, int(req.context_window_days))
                target_history_sales = list(model_history_sales)
            else:
                # Use last available day as "today" for forecasting.
                reference_date = pd.Timestamp(s_model.index.max())

                # Clip negative daily totals (returns etc.) because "sales" can't be negative.
                s_model = s_model.clip(lower=0.0)
                s_target = (s_target if "s_target" in locals() else s_model).clip(lower=0.0)  # type: ignore[has-type]

                start = reference_date - pd.Timedelta(days=int(req.context_window_days) - 1)
                s_model_ctx = s_model.loc[s_model.index >= start]
                s_target_ctx = s_target.loc[s_target.index >= start]

                model_history_sales = [float(v) for v in s_model_ctx.values.tolist()]
                target_history_sales = [float(v) for v in s_target_ctx.values.tolist()]

                # Allocation share for top-down (constant share computed from the same window).
                if scope_level in {"sub_category", "subcategory", "sub-category"}:
                    cat_total = float(np.asarray(s_model_ctx.values, dtype=float).sum()) if len(s_model_ctx) else 0.0
                    tgt_total = float(np.asarray(s_target_ctx.values, dtype=float).sum()) if len(s_target_ctx) else 0.0
                    if cat_total <= 1e-6:
                        rule = _sales_hierarchy_get_category_rule(str(scope_meta.get("category") or "")) or {}
                        subs = [sr.get("name") for sr in (rule.get("sub_categories") or [])]
                        n_children = int(len([x for x in subs if x])) or 1
                        allocation_share = 1.0 / float(n_children)
                    else:
                        allocation_share = max(0.0, min(1.0, tgt_total / cat_total))
                    scope_meta["share"] = allocation_share

                # Build history_points for UI using the selected scope (target).
                for d, v in s_target_ctx.items():
                    history_points.append(
                        {
                            "date": pd.Timestamp(d).strftime("%Y-%m-%d"),
                            "sales": float(v),
                            "is_holiday": _sales_forecast_is_holiday(pd.Timestamp(d), req.holiday_country),
                        }
                    )

        # 3) Forecast with recursive feature generation.
        horizon = int(req.forecast_horizon)
        model_series = list(model_history_sales)

        # Confidence interval width from recent volatility of the target series (simple heuristic).
        hist_tail = np.asarray(target_history_sales[-30:], dtype=float) if target_history_sales else np.asarray([], dtype=float)
        sigma_raw = float(hist_tail.std(ddof=0)) if hist_tail.size else 0.0
        z = 1.96

        if reference_date is None:
            # If we have no reference_date (cold start), start "tomorrow" from today().
            reference_date = pd.Timestamp(datetime.now().date())

        baseline_rows: List[Dict[str, Any]] = []
        baseline_pre: List[float] = []

        scenario_pre: List[float] = []
        scenario_scales: List[float] = []
        scenario_dates: List[str] = []

        # Scenario multipliers (kept simple & transparent).
        price_factor = 1.0
        growth_pct = 0.0
        scenario_promo = 1.0
        marketing_add_total = 0.0
        if scenario:
            dp = float(scenario.price_change_pct)
            e = float(scenario.price_elasticity)
            q_change = e * dp
            price_factor = max(0.0, (1.0 + dp) * (1.0 + q_change))
            growth_pct = float(scenario.market_growth_pct)
            scenario_promo = float(scenario.promotion_factor)
            marketing_add_total = float(scenario.marketing_budget) * float(scenario.marketing_roi)

        for step in range(1, horizon + 1):
            ts = reference_date + pd.Timedelta(days=step)
            feats = _sales_forecast_compute_features(ts=ts, sales_history=model_series, holiday_country=req.holiday_country)

            if model_obj is None:
                # Baseline: 7-day moving average.
                raw_pred_model = float(feats.get("rolling_mean_7", 0.0))
            else:
                raw_pred_model = _sales_forecast_predict_one(model_obj, feats)

            # Never allow negative sales into the recursive history.
            raw_pred_model = float(raw_pred_model)
            raw_pred_model_hist = max(0.0, raw_pred_model) if req.clip_negative else raw_pred_model
            model_series.append(float(raw_pred_model_hist))

            # Top-down allocation: selected scope may be a sub-category allocated from the category total.
            raw_pred_target = float(raw_pred_model)
            if scope_level in {"sub_category", "subcategory", "sub-category"}:
                raw_pred_target = raw_pred_target * float(allocation_share)

            # Baseline applies request.promotion_factor (legacy knob).
            baseline_scale = float(req.promotion_factor)
            baseline_value_pre = float(raw_pred_target) * baseline_scale
            baseline_pre.append(baseline_value_pre)

            baseline_value = _sales_forecast_post_process(
                value=raw_pred_target,
                promotion_factor=baseline_scale,
                clip_negative=req.clip_negative,
                rounding=req.rounding,
            )

            # CI scales with the same baseline multiplier.
            sigma_baseline = sigma_raw * baseline_scale
            ci_lower_pre = baseline_value_pre - z * sigma_baseline
            ci_upper_pre = baseline_value_pre + z * sigma_baseline
            if req.clip_negative:
                ci_lower_pre = max(0.0, ci_lower_pre)
                ci_upper_pre = max(0.0, ci_upper_pre)

            ci_lower = _sales_forecast_post_process(value=ci_lower_pre, promotion_factor=1.0, clip_negative=True, rounding=req.rounding)
            ci_upper = _sales_forecast_post_process(value=ci_upper_pre, promotion_factor=1.0, clip_negative=True, rounding=req.rounding)

            date_str = ts.strftime("%Y-%m-%d")
            baseline_rows.append(
                {
                    "date": date_str,
                    "predicted_sales": baseline_value,
                    "confidence_interval": {"lower": ci_lower, "upper": ci_upper},
                }
            )

            if scenario:
                # Apply growth linearly over the horizon (simple & explainable).
                growth_factor = 1.0 + (growth_pct * (float(step) / float(horizon))) if horizon > 0 else 1.0
                scenario_scale = scenario_promo * price_factor * growth_factor
                scenario_scales.append(float(scenario_scale))
                scenario_pre.append(float(raw_pred_target) * float(scenario_scale))
                scenario_dates.append(date_str)

        # 4) Trend summary (simple).
        last7 = np.asarray(target_history_sales[-7:], dtype=float) if target_history_sales else np.asarray([], dtype=float)
        last7_avg = float(last7.mean()) if last7.size else 0.0
        fut = np.asarray([r["predicted_sales"] for r in baseline_rows], dtype=float)
        fut_avg = float(fut.mean()) if fut.size else 0.0

        trend_summary = "stable"
        if last7_avg <= 1e-6:
            trend_summary = "expected_growth" if fut_avg > 0 else "stable"
        else:
            ratio = fut_avg / last7_avg
            if ratio >= 1.05:
                trend_summary = "expected_growth"
            elif ratio <= 0.95:
                trend_summary = "expected_decline"

        out: Dict[str, Any] = {
            "status": "success",
            "meta": {
                "model_version": model_version,
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
            "scope": scope_meta,
            "forecast": baseline_rows,
            "trend_summary": trend_summary,
            # Helpful for UI visualization / debugging (optional fields).
            "history": history_points[-30:],
        }

        if scenario:
            # Allocate marketing budget as incremental revenue over the horizon using baseline weights.
            extras = [0.0] * horizon
            if marketing_add_total > 0 and horizon > 0:
                weights = [max(0.0, float(v)) for v in baseline_pre]
                wsum = float(sum(weights))
                if wsum <= 1e-6:
                    extras = [float(marketing_add_total) / float(horizon)] * horizon
                else:
                    extras = [float(marketing_add_total) * (w / wsum) for w in weights]

            sim_rows: List[Dict[str, Any]] = []
            for i in range(horizon):
                base_pre = float(scenario_pre[i])
                add = float(extras[i])
                value_pre = base_pre + add
                value_final = _sales_forecast_post_process(
                    value=value_pre,
                    promotion_factor=1.0,
                    clip_negative=req.clip_negative,
                    rounding=req.rounding,
                )

                sigma_s = sigma_raw * float(scenario_scales[i])
                ci_lower_pre = (base_pre - z * sigma_s) + add
                ci_upper_pre = (base_pre + z * sigma_s) + add
                if req.clip_negative:
                    ci_lower_pre = max(0.0, ci_lower_pre)
                    ci_upper_pre = max(0.0, ci_upper_pre)

                sim_rows.append(
                    {
                        "date": scenario_dates[i],
                        "predicted_sales": value_final,
                        "confidence_interval": {
                            "lower": _sales_forecast_post_process(value=ci_lower_pre, promotion_factor=1.0, clip_negative=True, rounding=req.rounding),
                            "upper": _sales_forecast_post_process(value=ci_upper_pre, promotion_factor=1.0, clip_negative=True, rounding=req.rounding),
                        },
                    }
                )

            out["scenario"] = scenario.model_dump()
            out["simulation_forecast"] = sim_rows

        return out
    except HTTPException:
        raise
    except Exception as e:
        # Surface the root cause to the client for easier debugging in demo environments.
        raise HTTPException(status_code=500, detail=f"Sales forecast failed: {e}")


# -----------------------------
# Stock Allocation (Demo)
# -----------------------------


class StockAllocationRunRequest(BaseModel):
    """
    Stock allocation simulation request.

    This is intentionally lightweight and explainable for demo purposes.
    """

    product_name: str = Field(..., description="Product Name as shown in the dataset (store_sales_data.csv)")

    objective: str = Field("profit", description="profit | fairness | strategic")

    # Constraints
    enable_vip_new_york: bool = Field(True, description="Hard constraint: New York must meet a minimum fill rate")
    vip_new_york_min_fill: float = Field(0.5, ge=0.0, le=1.0, description="Minimum fill rate for New York when enabled")

    enable_min_shipment: bool = Field(True, description="Hard constraint: allocations must be 0 or >= min_shipment_units")
    min_shipment_units: int = Field(10, ge=1, le=1_000_000)

    # Soft constraint (used by the fairness objective)
    min_fill_rate: float = Field(0.3, ge=0.0, le=1.0, description="Minimum fill rate for all cities under fairness objective")

    # Simulation knobs
    demand_shock_pct: float = Field(0.0, ge=-50.0, le=50.0, description="Demand shock, percentage")
    inventory_shrink_pct: float = Field(0.0, ge=-20.0, le=0.0, description="Inventory shrink, percentage (negative)")

    # Horizon / replenishment (fixed defaults for the demo)
    horizon_days: int = Field(7, ge=1, le=30)
    inbound_days: int = Field(3, ge=0, le=30)
    inbound_units: int = Field(2000, ge=0, le=10_000_000)

    # Optional overrides
    target_total_demand_units: int = Field(10000, ge=100, le=10_000_000, description="Scale demand to a demo-friendly total")
    base_inventory_units: Optional[int] = Field(None, ge=0, le=10_000_000, description="If omitted, defaults to 50% of target_total_demand_units")
    max_cities: int = Field(15, ge=3, le=50)

    # Forecast model selection (repo-level model asset).
    model_id: str = Field("store_sales_forecast:v1.0.0", description="Model asset id (models/{name}/{version}/model.pkl)")
    holiday_country: str = Field("US", description="holidays country code")
    context_window_days: int = Field(365, ge=30, le=3650)

    # Insight (LLM)
    include_ai_explanation: bool = Field(
        True,
        description="When true, generate an AI explanation (Markdown) for why the allocation looks like this.",
    )


def _stock_alloc_parse_store_date(series: pd.Series) -> pd.Series:
    # The sample dataset uses day-first dates (e.g., 08/11/2017 = 8 Nov 2017).
    dt = pd.to_datetime(series, errors="coerce", dayfirst=True)
    # Fallback: try US-style month-first when parsing fails.
    if dt.isna().mean() > 0.5:
        dt2 = pd.to_datetime(series, errors="coerce", dayfirst=False)
        if dt2.notna().sum() > dt.notna().sum():
            dt = dt2
    return dt.dt.floor("D")


def _stock_alloc_city_alias(city: str) -> str:
    c = (city or "").strip()
    if c.lower() in {"new york"}:
        return "New York City"
    return c


def _stock_alloc_city_latlon(city: str) -> tuple[float, float]:
    # Minimal US city map for the demo. Unknown cities are placed deterministically.
    known = {
        "San Francisco": (37.7749, -122.4194),
        "New York City": (40.7128, -74.0060),
        "Chicago": (41.8781, -87.6298),
        "Los Angeles": (34.0522, -118.2437),
        "Seattle": (47.6062, -122.3321),
        "Dallas": (32.7767, -96.7970),
        "Houston": (29.7604, -95.3698),
        "Phoenix": (33.4484, -112.0740),
        "Philadelphia": (39.9526, -75.1652),
        "San Diego": (32.7157, -117.1611),
        "San Jose": (37.3382, -121.8863),
        "Austin": (30.2672, -97.7431),
        "Jacksonville": (30.3322, -81.6557),
        "Columbus": (39.9612, -82.9988),
        "Fort Worth": (32.7555, -97.3308),
        "Charlotte": (35.2271, -80.8431),
        "Detroit": (42.3314, -83.0458),
        "El Paso": (31.7619, -106.4850),
        "Memphis": (35.1495, -90.0490),
        "Boston": (42.3601, -71.0589),
        "Denver": (39.7392, -104.9903),
        "Washington": (38.9072, -77.0369),
        "Nashville": (36.1627, -86.7816),
        "Baltimore": (39.2904, -76.6122),
        "Oklahoma City": (35.4676, -97.5164),
        "Portland": (45.5152, -122.6784),
        "Las Vegas": (36.1699, -115.1398),
        "Louisville": (38.2527, -85.7585),
        "Milwaukee": (43.0389, -87.9065),
        "Albuquerque": (35.0844, -106.6504),
        "Tucson": (32.2226, -110.9747),
        "Fresno": (36.7378, -119.7871),
        "Sacramento": (38.5816, -121.4944),
        "Mesa": (33.4152, -111.8315),
        "Atlanta": (33.7490, -84.3880),
        "Miami": (25.7617, -80.1918),
        "Minneapolis": (44.9778, -93.2650),
        "Cleveland": (41.4993, -81.6944),
        "Kansas City": (39.0997, -94.5786),
        "Indianapolis": (39.7684, -86.1581),
    }
    c = _stock_alloc_city_alias(city)
    if c in known:
        return known[c]

    # Deterministic pseudo-location within US bounds.
    # Lat: [25, 49], Lon: [-124, -67]
    h = hashlib.sha1(c.encode("utf-8")).hexdigest()
    a = int(h[:8], 16)
    b = int(h[8:16], 16)
    lat = 25.0 + (a % 2400) / 2400.0 * (49.0 - 25.0)
    lon = -124.0 + (b % 2800) / 2800.0 * (-67.0 - (-124.0))
    return lat, lon


def _stock_alloc_generate_explanation_markdown_fallback(
    *,
    objective: str,
    constraints: Dict[str, Any],
    simulation: Dict[str, Any],
    totals: Dict[str, Any],
    kpis: Dict[str, Any],
    top_cities: List[Dict[str, Any]],
) -> str:
    """Deterministic fallback explanation (English, Markdown)."""
    total_demand = int(totals.get("total_demand_units") or 0)
    supply_total = int(totals.get("supply_total_units") or 0)
    fill_rate = float(kpis.get("fill_rate") or 0.0)
    profit = float(kpis.get("profit") or 0.0)
    lost_sales = float(kpis.get("lost_sales") or 0.0)
    risk = int(kpis.get("risk_store_count") or 0)

    vip_on = bool(constraints.get("enable_vip_new_york"))
    vip_min = float(constraints.get("vip_new_york_min_fill") or 0.0)
    min_ship_on = bool(constraints.get("enable_min_shipment"))
    min_ship_units = int(constraints.get("min_shipment_units") or 0)
    min_fill = float(constraints.get("min_fill_rate") or 0.0)

    demand_shock = float(simulation.get("demand_shock_pct") or 0.0)
    shrink = float(simulation.get("inventory_shrink_pct") or 0.0)

    lines: List[str] = []
    lines.append("### Executive Summary")
    lines.append(
        f"- Simulated demand: {total_demand:,} units; supply considered: {supply_total:,} units; global fill rate: {fill_rate*100:.0f}%."
    )
    lines.append(f"- Estimated profit: ${profit:,.0f}; lost sales proxy: ${lost_sales:,.0f}; risk stores (<20% fill): {risk}.")
    lines.append("")

    lines.append("### Why This Allocation Looks Like This")
    if objective == "profit":
        lines.append("- Objective = Profit Max: allocate inventory to higher-margin cities first to maximize profit per unit.")
    elif objective == "fairness":
        lines.append(
            f"- Objective = Fairness: enforce a minimum fill rate baseline for all cities (current: {min_fill*100:.0f}%), then distribute remaining supply."
        )
    else:
        lines.append("- Objective = Strategic: prioritize core cities (e.g., San Francisco) after hard constraints, then allocate by margin.")

    if vip_on:
        lines.append(f"- VIP constraint: New York City must receive at least {vip_min*100:.0f}% of its forecast demand (hard constraint).")
    if min_ship_on:
        lines.append(f"- Minimum shipment constraint: allocations are either 0 or >= {min_ship_units:,} units; this can zero-out tiny leftovers.")

    if abs(demand_shock) > 1e-9:
        lines.append(f"- Demand shock applied: {demand_shock:+.0f}% (stress test).")
    if shrink < -1e-9:
        lines.append(f"- Inventory shrink applied: {shrink:.0f}% (unexpected warehouse loss).")
    lines.append("")

    lines.append("### Recommended Actions")
    if fill_rate < 0.6:
        lines.append("- Consider increasing available inventory (or expediting inbound) to lift the global fill rate.")
    if risk > 0:
        lines.append("- Review the riskiest cities (<20% fill) and decide whether to add fairness constraints or relax profit focus.")
    if objective == "profit":
        lines.append("- If customer experience is impacted, switch to Fairness or add city-level minimum fill constraints for key accounts.")
    if objective == "fairness":
        lines.append("- Tune the minimum fill slider to balance customer coverage vs. profit (higher floor -> lower profit).")
    if min_ship_on:
        lines.append("- If too many cities get 0 allocation, lower the minimum shipment size to reduce drop-offs.")
    if not any(l.startswith("-") for l in lines[-3:]):
        lines.append("- Re-run with alternative objectives to compare KPIs and trade-offs.")

    if top_cities:
        lines.append("")
        lines.append("#### Quick View (Top Cities)")
        for c in top_cities[:8]:
            city = str(c.get("city") or "")
            d = int(c.get("forecast_demand") or 0)
            a = int(c.get("allocated") or 0)
            fr = float(c.get("fill_rate") or 0.0)
            tier = str(c.get("margin_tier") or "")
            lines.append(f"- {city}: demand {d:,}, allocated {a:,}, fill {fr*100:.0f}%, margin {tier}")

    return "\n".join(lines).rstrip() + "\n"


def _stock_alloc_generate_explanation_markdown_ai(*, context: Dict[str, Any]) -> str:
    """AI explanation generator for stock allocation (English, Markdown)."""
    model = os.getenv("VOLCENGINE_MODEL") or os.getenv("STOCK_ALLOC_INSIGHT_MODEL") or "kimi-k2-250905"
    timeout_seconds = int(os.getenv("STOCK_ALLOC_AI_TIMEOUT_SECONDS") or "30")
    max_tokens = int(os.getenv("STOCK_ALLOC_AI_MAX_TOKENS") or "900")

    system = (
        "You are a supply chain optimization analyst.\n"
        "Explain why a simulated stock allocation plan looks the way it does.\n"
        "Output MUST be Markdown and MUST contain exactly these three sections (use '### ' headings):\n"
        "1) ### Executive Summary\n"
        "2) ### Why This Allocation Looks Like This\n"
        "3) ### Recommended Actions\n"
        "Rules:\n"
        "- Write in English.\n"
        "- Reference the provided numbers where possible (fill rate, profit, lost sales, risk stores).\n"
        "- Explicitly mention objective, hard constraints, and what-if shocks.\n"
        "- Be concise and executive-friendly.\n"
        "- Do NOT invent data that is not in the JSON context.\n"
    )

    user = "Stock allocation context (JSON):\n" + json.dumps(context, ensure_ascii=False, separators=(",", ":"))
    return _call_volcengine_chat_completion(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.2,
        max_tokens=max_tokens,
        timeout_seconds=timeout_seconds,
    )


def _stock_alloc_margin_for_region(region: str) -> tuple[str, float]:
    r = (region or "").strip().lower()
    if r == "west":
        return "High", 0.50
    if r == "east":
        return "Low", 0.15
    if r == "central":
        return "Med", 0.30
    if r == "south":
        return "Med", 0.30
    return "Med", 0.30


def _stock_alloc_feature_row(history: List[float], ts: pd.Timestamp, holiday_country: str) -> Dict[str, float]:
    # Feature schema must match the training script `train_store_sales_forecast_lgbm.py`.
    lags = (1, 7, 14, 28)
    rolls = (7, 14, 28)

    def _lag(k: int) -> float:
        if k <= 0:
            return 0.0
        if len(history) < k:
            return 0.0
        return float(history[-k])

    def _roll_mean(w: int) -> float:
        if w <= 0:
            return 0.0
        window = history[-w:] if len(history) >= w else history
        if not window:
            return 0.0
        return float(np.mean(window))

    def _roll_std(w: int) -> float:
        if w <= 0:
            return 0.0
        window = history[-w:] if len(history) >= w else history
        if not window:
            return 0.0
        return float(np.std(window))

    lag1 = _lag(1)
    lag7 = _lag(7)
    mom7 = (lag1 - lag7) / (abs(lag1) + abs(lag7) + 1e-6)

    return {
        "month": float(int(ts.month)),
        "day_of_week": float(int(ts.dayofweek)),
        "is_weekend": float(int(ts.dayofweek >= 5)),
        "day_of_year": float(int(ts.dayofyear)),
        "is_holiday": float(_sales_forecast_is_holiday(ts, holiday_country)),
        **{f"lag_{k}": float(_lag(k)) for k in lags},
        **{f"rolling_mean_{w}": float(_roll_mean(w)) for w in rolls},
        **{f"rolling_std_{w}": float(_roll_std(w)) for w in rolls},
        "mom_7": float(mom7),
    }


def _stock_alloc_forecast_daily_units(
    *,
    model: Any,
    history: List[float],
    reference_date: pd.Timestamp,
    horizon_days: int,
    holiday_country: str,
) -> List[float]:
    # Iterative day-ahead simulation, same as the sales_forecast module style.
    feature_cols = [
        "month",
        "day_of_week",
        "is_weekend",
        "day_of_year",
        "is_holiday",
        "lag_1",
        "lag_7",
        "lag_14",
        "lag_28",
        "rolling_mean_7",
        "rolling_mean_14",
        "rolling_mean_28",
        "rolling_std_7",
        "rolling_std_14",
        "rolling_std_28",
        "mom_7",
    ]

    hist = [float(max(0.0, v)) for v in history]
    preds: List[float] = []
    for step in range(1, int(horizon_days) + 1):
        ts = (pd.Timestamp(reference_date) + pd.Timedelta(days=int(step))).floor("D")
        row = _stock_alloc_feature_row(hist, ts, holiday_country)
        X = pd.DataFrame([[row.get(c, 0.0) for c in feature_cols]], columns=feature_cols)
        yhat = float(model.predict(X)[0])
        yhat = float(max(0.0, yhat))
        preds.append(yhat)
        hist.append(yhat)
    return preds


def _stock_alloc_allocate(
    *,
    demands: Dict[str, int],
    margin_rate: Dict[str, float],
    objective: str,
    supply_units: int,
    enable_min_shipment: bool,
    min_shipment_units: int,
    enable_vip_new_york: bool,
    vip_new_york_min_fill: float,
    min_fill_rate: float,
    strategic_cities: Optional[List[str]] = None,
) -> Dict[str, int]:
    obj = (objective or "profit").strip().lower()
    if obj not in {"profit", "fairness", "strategic"}:
        raise HTTPException(status_code=400, detail="objective must be 'profit', 'fairness', or 'strategic'")

    remaining = int(max(0, supply_units))
    alloc: Dict[str, int] = {c: 0 for c in demands.keys()}

    def _min_ship_ok(v: int) -> bool:
        if not enable_min_shipment:
            return True
        return v == 0 or v >= int(min_shipment_units)

    # Build required mins.
    required: Dict[str, int] = {c: 0 for c in demands.keys()}
    if enable_vip_new_york:
        for c in demands.keys():
            if _stock_alloc_city_alias(c).lower() == "new york city":
                required[c] = max(required[c], int(math.ceil(float(demands[c]) * float(vip_new_york_min_fill))))

    if obj == "fairness":
        for c in demands.keys():
            required[c] = max(required[c], int(math.ceil(float(demands[c]) * float(min_fill_rate))))

    # Apply min-shipment constraint to required mins.
    for c, req in list(required.items()):
        req = int(min(req, demands[c]))
        if enable_min_shipment and 0 < req < int(min_shipment_units):
            req = 0
        required[c] = req

    required_total = int(sum(required.values()))
    if required_total > remaining:
        # Not enough supply to meet mins: allocate VIP first, then proportionally by demand.
        vip_city: Optional[str] = None
        if enable_vip_new_york:
            for c in demands.keys():
                if _stock_alloc_city_alias(c).lower() == "new york city":
                    vip_city = c
                    break

        if vip_city and required.get(vip_city, 0) > 0:
            take = int(min(remaining, required[vip_city]))
            alloc[vip_city] = take
            remaining -= take

        total_demand = float(sum(demands.values()) or 1.0)
        if remaining > 0:
            # Proportional to demand; keep deterministic ordering.
            for c in sorted(demands.keys()):
                if remaining <= 0:
                    break
                if c == vip_city:
                    continue
                share = float(demands[c]) / total_demand
                take = int(math.floor(float(remaining) * share))
                take = int(min(take, demands[c]))
                if enable_min_shipment and 0 < take < int(min_shipment_units):
                    take = 0
                if take > 0:
                    alloc[c] = take
        return alloc

    # 1) Allocate required mins.
    if required_total > 0:
        # VIP first for better story alignment.
        ordered = sorted(required.keys(), key=lambda c: (0 if _stock_alloc_city_alias(c).lower() == "new york city" else 1, c))
        for c in ordered:
            req = required[c]
            if req <= 0:
                continue
            take = int(min(req, remaining))
            if take <= 0:
                continue
            alloc[c] = take
            remaining -= take

    # 2) Allocate remaining by objective.
    def _remaining_demand(c: str) -> int:
        return int(max(0, demands[c] - alloc.get(c, 0)))

    # Strategic: fill core cities first (after mins), then behave like profit.
    if obj == "strategic":
        cores = [c for c in (strategic_cities or ["San Francisco"]) if c in demands]
        for c in cores:
            if remaining <= 0:
                break
            need = _remaining_demand(c)
            if need <= 0:
                continue
            take = int(min(need, remaining))
            if enable_min_shipment and 0 < take < int(min_shipment_units):
                take = 0
            alloc[c] += take
            remaining -= take

    # Profit-like distribution for the rest.
    ordered_cities = sorted(
        demands.keys(),
        key=lambda c: (float(margin_rate.get(c, 0.0)), float(demands.get(c, 0))),
        reverse=True,
    )
    for c in ordered_cities:
        if remaining <= 0:
            break
        need = _remaining_demand(c)
        if need <= 0:
            continue
        take = int(min(need, remaining))
        if enable_min_shipment and 0 < take < int(min_shipment_units):
            take = 0
        alloc[c] += take
        remaining -= take

    # Ensure min shipment: zero out any tiny allocations (best-effort).
    if enable_min_shipment:
        for c in list(alloc.keys()):
            if 0 < int(alloc[c]) < int(min_shipment_units):
                alloc[c] = 0

    return alloc


@router.get("/{dataset_id}/stock-allocation/products")
def stock_allocation_list_products(
    dataset_id: str,
    limit: int = Query(50, ge=1, le=200),
):
    """List top products for the allocation demo (from CSV; no DB dependency)."""
    config = load_datasets_config()
    dataset_config = next((d for d in config.get("datasets", []) if d.get("id") == dataset_id), None)
    if not dataset_config:
        raise HTTPException(status_code=404, detail="Dataset not found")

    csv_path = _get_dataset_csv_path(dataset_id, dataset_config)
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail=f"CSV file not found: {csv_path}")

    encoding, header_cols = _read_csv_header(csv_path)
    date_col = _resolve_csv_column(header_cols, str(dataset_config.get("time_column") or "Order Date"))
    city_col = _resolve_csv_column(header_cols, "City")
    prod_col = _resolve_csv_column(header_cols, "Product Name")
    cat_col = _resolve_csv_column(header_cols, "Category")
    subcat_col = _resolve_csv_column(header_cols, "Sub-Category")
    sales_col = _resolve_csv_column(header_cols, "Sales")

    df = pd.read_csv(csv_path, encoding=encoding, usecols=[date_col, city_col, prod_col, cat_col, subcat_col, sales_col], low_memory=False)
    df[date_col] = _stock_alloc_parse_store_date(df[date_col])
    df = df.dropna(subset=[date_col, prod_col, city_col])
    df[sales_col] = pd.to_numeric(df[sales_col], errors="coerce").fillna(0.0).astype(float)

    # Aggregate by product name (top by revenue).
    grp = (
        df.groupby([prod_col, cat_col, subcat_col], dropna=False)
        .agg(total_orders=(prod_col, "size"), total_revenue=(sales_col, "sum"))
        .reset_index()
        .sort_values("total_revenue", ascending=False)
    )

    items: List[Dict[str, Any]] = []
    for _, r in grp.head(int(limit)).iterrows():
        items.append(
            {
                "product_name": str(r[prod_col]),
                "category": str(r.get(cat_col) or ""),
                "sub_category": str(r.get(subcat_col) or ""),
                "total_orders": int(r.get("total_orders") or 0),
                "total_revenue": float(r.get("total_revenue") or 0.0),
            }
        )

    # Ensure the demo product exists in the list when present in the dataset.
    demo_prod = "Staple envelope"
    if demo_prod not in [it["product_name"] for it in items]:
        hit = grp[grp[prod_col].astype(str) == demo_prod]
        if not hit.empty:
            rr = hit.iloc[0]
            items.insert(
                0,
                {
                    "product_name": demo_prod,
                    "category": str(rr.get(cat_col) or ""),
                    "sub_category": str(rr.get(subcat_col) or ""),
                    "total_orders": int(rr.get("total_orders") or 0),
                    "total_revenue": float(rr.get("total_revenue") or 0.0),
                },
            )

    return {"success": True, "data": items, "total": len(items)}


@router.post("/{dataset_id}/stock-allocation/run")
def stock_allocation_run(dataset_id: str, req: StockAllocationRunRequest):
    """Run the stock allocation simulation (CSV + LightGBM + heuristic optimization)."""
    config = load_datasets_config()
    dataset_config = next((d for d in config.get("datasets", []) if d.get("id") == dataset_id), None)
    if not dataset_config:
        raise HTTPException(status_code=404, detail="Dataset not found")

    csv_path = _get_dataset_csv_path(dataset_id, dataset_config)
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail=f"CSV file not found: {csv_path}")

    # Load model (repo-level asset in models/).
    model, model_version = _sales_forecast_resolve_model(req.model_id)
    if model is None:
        raise HTTPException(status_code=400, detail="Stock allocation requires a LightGBM model_id (baseline is not supported here).")

    encoding, header_cols = _read_csv_header(csv_path)
    date_col = _resolve_csv_column(header_cols, str(dataset_config.get("time_column") or "Order Date"))
    city_col = _resolve_csv_column(header_cols, "City")
    region_col = _resolve_csv_column(header_cols, "Region")
    prod_col = _resolve_csv_column(header_cols, "Product Name")
    cat_col = _resolve_csv_column(header_cols, "Category")
    subcat_col = _resolve_csv_column(header_cols, "Sub-Category")
    sales_col = _resolve_csv_column(header_cols, "Sales")

    df = pd.read_csv(
        csv_path,
        encoding=encoding,
        usecols=[date_col, city_col, region_col, prod_col, cat_col, subcat_col, sales_col],
        low_memory=False,
    )
    df[date_col] = _stock_alloc_parse_store_date(df[date_col])
    df = df.dropna(subset=[date_col, city_col, prod_col])
    df[sales_col] = pd.to_numeric(df[sales_col], errors="coerce").fillna(0.0).astype(float)
    df[city_col] = df[city_col].astype(str).map(_stock_alloc_city_alias)

    product_name = (req.product_name or "").strip()
    if not product_name:
        raise HTTPException(status_code=400, detail="product_name is required")

    df_prod = df[df[prod_col].astype(str) == product_name].copy()
    if df_prod.empty:
        raise HTTPException(status_code=404, detail=f"Product not found in dataset: {product_name}")

    category = str(df_prod[cat_col].iloc[0] if cat_col in df_prod.columns else "")
    sub_category = str(df_prod[subcat_col].iloc[0] if subcat_col in df_prod.columns else "")

    # Forecast at sub-category level (more stable), then downscale to product via recent share.
    df_sub = df[df[subcat_col].astype(str) == sub_category].copy() if sub_category else df_prod.copy()
    if df_sub.empty:
        df_sub = df_prod.copy()

    reference_date = pd.Timestamp(df_sub[date_col].max()).floor("D")
    ctx_days = int(req.context_window_days)
    ctx_start = reference_date - pd.Timedelta(days=ctx_days - 1)

    df_sub_ctx = df_sub[df_sub[date_col] >= ctx_start]
    df_prod_ctx = df_prod[df_prod[date_col] >= ctx_start]
    sub_units = int(df_sub_ctx.shape[0])
    prod_units = int(df_prod_ctx.shape[0])
    product_share = float(prod_units) / float(sub_units) if sub_units > 0 else 1.0
    product_share = float(max(0.01, min(0.50, product_share)))

    # Build dense daily history for the sub-category.
    daily_sub = (
        df_sub_ctx.groupby(date_col, as_index=False)
        .size()
        .rename(columns={"size": "y"})
        .sort_values(date_col)
    )
    if daily_sub.empty:
        raise HTTPException(status_code=400, detail="No usable history in the context window for this sub-category")

    full_idx = pd.date_range(start=pd.Timestamp(daily_sub[date_col].min()), end=reference_date, freq="D")
    s_sub = daily_sub.set_index(date_col)["y"].reindex(full_idx, fill_value=0.0).astype(float)
    history_sub = [float(v) for v in s_sub.values.tolist()]

    horizon = int(req.horizon_days)
    forecast_sub = _stock_alloc_forecast_daily_units(
        model=model,
        history=history_sub,
        reference_date=reference_date,
        horizon_days=horizon,
        holiday_country=req.holiday_country,
    )

    # Product-level forecast derived from sub-category.
    forecast_prod_daily = [float(v) * float(product_share) for v in forecast_sub]
    shock = 1.0 + float(req.demand_shock_pct) / 100.0
    forecast_prod_daily = [float(max(0.0, v * shock)) for v in forecast_prod_daily]
    raw_total_demand = float(sum(forecast_prod_daily))
    if raw_total_demand <= 1e-9:
        # Keep demo usable even for ultra-sparse products.
        raw_total_demand = 1.0
        forecast_prod_daily = [1.0 / float(horizon)] * horizon

    # Scale to a demo-friendly total demand (units).
    target_total = int(req.target_total_demand_units)
    scale = float(target_total) / float(raw_total_demand)
    daily_total_demand = [int(max(0, round(v * scale))) for v in forecast_prod_daily]
    total_demand = int(sum(daily_total_demand))
    if total_demand <= 0:
        daily_total_demand = [0] * horizon
        total_demand = 0

    # City demand shares: prefer product-specific if enough signal, else use sub-category.
    df_city_base = df_prod_ctx if int(df_prod_ctx.shape[0]) >= 50 else df_sub_ctx
    city_units = (
        df_city_base.groupby([city_col, region_col], dropna=False)
        .size()
        .rename("units")
        .reset_index()
        .sort_values("units", ascending=False)
    )
    if city_units.empty:
        raise HTTPException(status_code=400, detail="No city data available for allocation")

    # Pick top cities (plus mandatory demo cities when present).
    mandatory = {"San Francisco", "New York City", "Chicago"}
    picked: List[Dict[str, Any]] = []
    seen = set()

    for _, r in city_units.iterrows():
        c = str(r[city_col])
        if c in seen:
            continue
        picked.append({"city": c, "region": str(r.get(region_col) or ""), "units": int(r.get("units") or 0)})
        seen.add(c)
        if len(picked) >= int(req.max_cities):
            break

    for m in sorted(mandatory):
        if m in seen:
            continue
        hit = city_units[city_units[city_col].astype(str) == m]
        if not hit.empty:
            rr = hit.iloc[0]
            picked.append({"city": m, "region": str(rr.get(region_col) or ""), "units": int(rr.get("units") or 0)})
            seen.add(m)

    picked = picked[: int(req.max_cities)]
    total_units_for_share = float(sum(int(p.get("units") or 0) for p in picked) or 1.0)
    shares = {p["city"]: float(p.get("units") or 0) / total_units_for_share for p in picked}

    # City-level demand (scaled).
    demand_by_city: Dict[str, int] = {c: int(max(0, round(float(total_demand) * float(shares.get(c, 0.0))))) for c in shares.keys()}
    # Fix rounding drift by adjusting the top city.
    drift = int(total_demand - sum(demand_by_city.values()))
    if drift != 0:
        top_city = max(demand_by_city.keys(), key=lambda c: demand_by_city[c])
        demand_by_city[top_city] = int(max(0, demand_by_city[top_city] + drift))

    # Supply assumptions (scaled units).
    base_inventory = int(req.base_inventory_units) if req.base_inventory_units is not None else int(round(float(total_demand) * 0.5))
    base_inventory = int(max(0, round(float(base_inventory) * (1.0 + float(req.inventory_shrink_pct) / 100.0))))
    inbound_units = int(req.inbound_units)
    inbound_days = int(req.inbound_days)
    supply_total = int(base_inventory + (inbound_units if 0 < inbound_days <= horizon else 0))

    # Margin by region (profit per unit); keeps KPI magnitudes readable.
    margin_rate: Dict[str, float] = {}
    margin_tier: Dict[str, str] = {}
    region_by_city: Dict[str, str] = {}
    for p in picked:
        c = p["city"]
        reg = p.get("region") or ""
        tier, rate = _stock_alloc_margin_for_region(str(reg))
        margin_rate[c] = float(rate)
        margin_tier[c] = str(tier)
        region_by_city[c] = str(reg)

    # Allocate.
    alloc_by_city = _stock_alloc_allocate(
        demands=demand_by_city,
        margin_rate=margin_rate,
        objective=req.objective,
        supply_units=supply_total,
        enable_min_shipment=bool(req.enable_min_shipment),
        min_shipment_units=int(req.min_shipment_units),
        enable_vip_new_york=bool(req.enable_vip_new_york),
        vip_new_york_min_fill=float(req.vip_new_york_min_fill),
        min_fill_rate=float(req.min_fill_rate),
        strategic_cities=["San Francisco"],
    )

    # Baseline profit-max for delta (same hard constraints, no fairness mins).
    alloc_profit_max = _stock_alloc_allocate(
        demands=demand_by_city,
        margin_rate=margin_rate,
        objective="profit",
        supply_units=supply_total,
        enable_min_shipment=bool(req.enable_min_shipment),
        min_shipment_units=int(req.min_shipment_units),
        enable_vip_new_york=bool(req.enable_vip_new_york),
        vip_new_york_min_fill=float(req.vip_new_york_min_fill),
        min_fill_rate=0.0,
        strategic_cities=["San Francisco"],
    )

    total_alloc = int(sum(int(v) for v in alloc_by_city.values()))
    fill_rate_global = float(total_alloc) / float(total_demand) if total_demand > 0 else 0.0
    lost_units = int(max(0, total_demand - total_alloc))

    profit = float(sum(float(alloc_by_city.get(c, 0)) * float(margin_rate.get(c, 0.0)) for c in demand_by_city.keys()))
    profit_max = float(sum(float(alloc_profit_max.get(c, 0)) * float(margin_rate.get(c, 0.0)) for c in demand_by_city.keys()))
    profit_delta = float(profit - profit_max)

    risk_store_count = 0
    for c, d in demand_by_city.items():
        if d <= 0:
            continue
        fr = float(alloc_by_city.get(c, 0)) / float(d)
        if fr < 0.2:
            risk_store_count += 1

    # Trend: inventory level vs forecast daily demand.
    inv = int(base_inventory)
    trend_points: List[Dict[str, Any]] = []
    out_of_stock_date: Optional[str] = None
    for i in range(horizon):
        day = reference_date + pd.Timedelta(days=i + 1)
        if inbound_days > 0 and (i + 1) == inbound_days:
            inv += inbound_units
        demand_i = int(daily_total_demand[i])
        inv = int(max(0, inv - demand_i))
        if out_of_stock_date is None and inv == 0 and demand_i > 0:
            out_of_stock_date = day.strftime("%Y-%m-%d")
        trend_points.append(
            {
                "date": day.strftime("%Y-%m-%d"),
                "forecast_demand": int(demand_i),
                "planned_inventory": int(inv),
            }
        )

    # City response.
    city_rows: List[Dict[str, Any]] = []
    for c in sorted(demand_by_city.keys(), key=lambda x: demand_by_city[x], reverse=True):
        d = int(demand_by_city.get(c, 0))
        a = int(alloc_by_city.get(c, 0))
        fr = float(a) / float(d) if d > 0 else 0.0
        status = "Sufficient" if fr >= 0.8 else ("Adjusted" if fr > 0 else "Dropped")
        lat, lon = _stock_alloc_city_latlon(c)
        city_rows.append(
            {
                "city": c,
                "region": region_by_city.get(c, ""),
                "lat": float(lat),
                "lon": float(lon),
                "margin_tier": margin_tier.get(c, "Med"),
                "margin_rate": float(margin_rate.get(c, 0.0)),
                "forecast_demand": d,
                "allocated": a,
                "fill_rate": float(fr),
                "status": status,
            }
        )

    gap_pct = (1.0 - (float(base_inventory) / float(total_demand))) * 100.0 if total_demand > 0 else 0.0

    # AI explanation (best-effort; never fail the simulation because of LLM issues).
    constraints_ctx = {
        "enable_vip_new_york": bool(req.enable_vip_new_york),
        "vip_new_york_min_fill": float(req.vip_new_york_min_fill),
        "enable_min_shipment": bool(req.enable_min_shipment),
        "min_shipment_units": int(req.min_shipment_units),
        "min_fill_rate": float(req.min_fill_rate),
    }
    simulation_ctx = {
        "demand_shock_pct": float(req.demand_shock_pct),
        "inventory_shrink_pct": float(req.inventory_shrink_pct),
    }
    totals_ctx = {
        "horizon_days": int(horizon),
        "total_demand_units": int(total_demand),
        "supply_total_units": int(supply_total),
        "available_inventory_units": int(base_inventory),
        "inbound_units": int(inbound_units),
        "inbound_days": int(inbound_days),
        "gap_pct": float(gap_pct),
        "out_of_stock_date": out_of_stock_date,
    }
    kpis_ctx = {
        "profit": float(profit),
        "profit_delta_vs_profit_max": float(profit_delta),
        "fill_rate": float(fill_rate_global),
        "lost_sales": float(lost_units),  # $1 per unit for demo readability
        "risk_store_count": int(risk_store_count),
    }
    top_cities_ctx = [
        {
            "city": r.get("city"),
            "region": r.get("region"),
            "margin_tier": r.get("margin_tier"),
            "margin_rate": r.get("margin_rate"),
            "forecast_demand": r.get("forecast_demand"),
            "allocated": r.get("allocated"),
            "fill_rate": r.get("fill_rate"),
            "status": r.get("status"),
        }
        for r in (city_rows[:10] if city_rows else [])
    ]

    ai_explanation_markdown: str
    ai_explanation_source: str
    ai_explanation_error: Optional[str] = None

    if bool(req.include_ai_explanation):
        try:
            ai_context = {
                "dataset_id": dataset_id,
                "reference_date": reference_date.strftime("%Y-%m-%d"),
                "product": {
                    "name": product_name,
                    "category": category,
                    "sub_category": sub_category,
                    "forecast_scope": "sub_category",
                    "product_share_within_sub_category": float(product_share),
                },
                "objective": str(req.objective),
                "constraints": constraints_ctx,
                "simulation": simulation_ctx,
                "totals": totals_ctx,
                "kpis": kpis_ctx,
                "top_cities": top_cities_ctx,
                "notes": {
                    "margin_tier_proxy": "Margin tier is a demo proxy derived from Region (West=High, East=Low, Central/South=Med).",
                    "profit_definition": "profit = sum(allocated_units * margin_rate), where margin_rate is a proxy per region.",
                    "lost_sales_definition": "lost_sales is reported as $1 per unfilled unit for readability in the demo.",
                },
            }
            ai_explanation_markdown = _stock_alloc_generate_explanation_markdown_ai(context=ai_context)
            ai_explanation_source = "ai"
        except Exception as e:
            ai_explanation_error = str(e)
            ai_explanation_markdown = _stock_alloc_generate_explanation_markdown_fallback(
                objective=str(req.objective),
                constraints=constraints_ctx,
                simulation=simulation_ctx,
                totals=totals_ctx,
                kpis=kpis_ctx,
                top_cities=top_cities_ctx,
            )
            ai_explanation_source = "fallback"
    else:
        ai_explanation_markdown = _stock_alloc_generate_explanation_markdown_fallback(
            objective=str(req.objective),
            constraints=constraints_ctx,
            simulation=simulation_ctx,
            totals=totals_ctx,
            kpis=kpis_ctx,
            top_cities=top_cities_ctx,
        )
        ai_explanation_source = "fallback"

    return {
        "success": True,
        "data": {
            "dataset_id": dataset_id,
            "model_version": model_version,
            "product": {
                "name": product_name,
                "category": category,
                "sub_category": sub_category,
                "forecast_scope": "sub_category",
                "product_share_within_sub_category": float(product_share),
            },
            "params": req.model_dump(),
            "reference_date": reference_date.strftime("%Y-%m-%d"),
            "supply": {
                "available_inventory": int(base_inventory),
                "inbound": {"days": int(inbound_days), "quantity": int(inbound_units)},
                "gap_pct": float(gap_pct),
                "supply_total": int(supply_total),
            },
            "kpis": {
                "profit": float(profit),
                "profit_delta_vs_profit_max": float(profit_delta),
                "fill_rate": float(fill_rate_global),
                "lost_sales": float(lost_units),  # $1 per unit for demo readability
                "risk_store_count": int(risk_store_count),
            },
            "ai_explanation_markdown": ai_explanation_markdown,
            "ai_explanation_source": ai_explanation_source,
            "ai_explanation_error": ai_explanation_error,
            "visuals": {
                "daily_forecast_total": [{"date": p["date"], "demand": p["forecast_demand"]} for p in trend_points],
                "trend": {"points": trend_points, "out_of_stock_date": out_of_stock_date},
            },
            "cities": city_rows,
        },
    }
