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
from typing import Optional, Dict, Any, List, Tuple
from urllib import request as urlrequest
from urllib import error as urlerror
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
    """Dataset-scoped customer clustering based on RFM user features."""

    time_window_days: int = Field(365, ge=1, le=3650)
    selected_features: List[str] = Field(default_factory=lambda: ["recency_days", "frequency", "monetary"])
    k_range: Tuple[int, int] = Field((3, 6))
    random_seed: int = Field(42)
    outlier_threshold: Optional[float] = Field(3.0)


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

    base_url = (os.getenv("VOLCENGINE_BASE_URL") or "https://ark.cn-beijing.volces.com/api/v3").rstrip("/")
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


@router.get("")
def get_datasets_list():
    """获取数据集列表"""
    try:
        config = load_datasets_config()
        datasets = []
        
        for dataset_config in config.get("datasets", []):
            try:
                row_count = get_table_row_count(dataset_config["table_name"])
                datasets.append({
                    "id": dataset_config["id"],
                    "name": dataset_config["name"],
                    "description": dataset_config["description"],
                    "table_name": dataset_config["table_name"],
                    "type": dataset_config["type"],
                    "row_count": row_count,
                    "created_at": dataset_config.get("created_at"),
                    "updated_at": dataset_config.get("updated_at"),
                    "tags": dataset_config.get("tags", []),
                })
            except Exception as e:
                print(f"⚠️  获取数据集 {dataset_config['id']} 统计信息失败: {e}")
                datasets.append({
                    "id": dataset_config["id"],
                    "name": dataset_config["name"],
                    "description": dataset_config["description"],
                    "table_name": dataset_config["table_name"],
                    "type": dataset_config["type"],
                    "row_count": 0,
                    "created_at": dataset_config.get("created_at"),
                    "updated_at": dataset_config.get("updated_at"),
                    "tags": dataset_config.get("tags", []),
                })
        
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
        
        # 获取总记录数
        row_count = get_table_row_count(dataset_config["table_name"])
        
        # 获取分页数据
        offset = (page - 1) * pageSize
        conn = get_db_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    f'SELECT * FROM "{dataset_config["table_name"]}" LIMIT %s OFFSET %s',
                    (pageSize, offset),
                )
                rows = [dict(row) for row in cur.fetchall()]
        finally:
            conn.close()
        
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
        
        # 获取表的基本信息
        row_count = get_table_row_count(dataset_config["table_name"])
        columns = get_table_columns(dataset_config["table_name"])
        
        # 获取时间范围
        time_range = None
        if dataset_config.get("time_column"):
            time_range = get_time_range(
                dataset_config["table_name"], dataset_config["time_column"]
            )
        
        # 构建列信息
        column_list = []
        for col in columns:
            stats = None
            if include_stats:
                stats = get_column_stats(
                    dataset_config["table_name"], col["column_name"], col["data_type"]
                )
            
            column_list.append({
                "name": col["column_name"],
                "type": col["data_type"],
                "stats": stats,
            })
        
        return {
            "success": True,
            "data": {
                "id": dataset_config["id"],
                "name": dataset_config["name"],
                "description": dataset_config["description"],
                "table_name": dataset_config["table_name"],
                "type": dataset_config["type"],
                "row_count": row_count,
                "column_count": len(columns),
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
            
        columns = get_table_columns(dataset_config["table_name"])
        
        # 获取所有列的统计信息
        column_stats_list = []
        for col in columns:
            stats = get_column_stats(
                dataset_config["table_name"], col["column_name"], col["data_type"]
            )
            column_stats_list.append({
                "name": col["column_name"],
                "type": col["data_type"],
                "stats": stats,
            })
            
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
    """Customer Segmentation (K-Means clustering) based on dataset RFM user features."""
    config = load_datasets_config()
    dataset_config = next((d for d in config.get("datasets", []) if d["id"] == dataset_id), None)
    if not dataset_config:
        raise HTTPException(status_code=404, detail="数据集不存在")

    rfm_cfg = dataset_config.get("rfm")
    if not isinstance(rfm_cfg, dict):
        raise HTTPException(status_code=400, detail="该数据集未配置 RFM 字段映射（datasets.json -> rfm）")

    user_col = rfm_cfg.get("user_id_column")
    order_col = rfm_cfg.get("order_id_column")
    date_col = rfm_cfg.get("order_date_column")
    amount_cfg = rfm_cfg.get("monetary")

    if not all(isinstance(x, str) and x for x in [user_col, order_col, date_col]):
        raise HTTPException(status_code=400, detail="RFM 字段映射不完整（user_id_column/order_id_column/order_date_column）")
    if not isinstance(amount_cfg, dict):
        raise HTTPException(status_code=400, detail="RFM monetary 配置缺失（datasets.json -> rfm.monetary）")

    base_filters = rfm_cfg.get("filters") or []
    if not isinstance(base_filters, list):
        raise HTTPException(status_code=400, detail="RFM filters 配置必须为数组")

    csv_path = _get_dataset_csv_path(dataset_id, dataset_config)
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail=f"CSV 文件不存在: {csv_path}")

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
            raise HTTPException(status_code=400, detail="RFM filters 配置必须为对象数组")
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
        raise HTTPException(status_code=400, detail="CSV 在过滤条件后无可用数据")

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
        raise HTTPException(status_code=400, detail="数据集缺少可用的订单时间，无法进行聚类分析")

    window_days = int(req.time_window_days)
    current_start = reference_date - pd.Timedelta(days=window_days)
    df_current = df_raw[
        (df_raw[date_col_resolved] >= current_start) & (df_raw[date_col_resolved] <= reference_date)
    ]
    if df_current.empty:
        raise HTTPException(status_code=400, detail="在当前分析窗口内未找到可用于聚类的数据")

    df_users = _rfm_aggregate_users(
        df_current,
        user_col=user_col_resolved,
        order_col=order_col_resolved,
        date_col=date_col_resolved,
        amount_cfg=amount_cfg_resolved,
        reference_date=reference_date,
    )
    if df_users.empty:
        raise HTTPException(status_code=400, detail="RFM 聚合后无有效用户数据，无法进行聚类")

    selected_features = req.selected_features or ["recency_days", "frequency", "monetary"]
    allowed = {"recency_days", "frequency", "monetary"}
    if not isinstance(selected_features, list) or not selected_features:
        raise HTTPException(status_code=400, detail="selected_features 必须为非空数组")
    if any((not isinstance(f, str)) or f not in allowed for f in selected_features):
        raise HTTPException(status_code=400, detail=f"selected_features 仅支持: {sorted(allowed)}")

    from shared.utils.customer_segmentation import segment_customers_kmeans

    result = segment_customers_kmeans(
        df_users,
        selected_features=selected_features,
        k_range=tuple(req.k_range),
        random_seed=int(req.random_seed),
        outlier_threshold=req.outlier_threshold,
        user_id_column="user_id",
    )
    if result.get("status") != "success":
        raise HTTPException(status_code=400, detail=str(result.get("message") or "聚类分析失败"))

    # Attach dataset-scoped metadata for UI display.
    result["dataset_id"] = dataset_id
    result["reference_date"] = str(reference_date)
    result["time_window_days"] = window_days
    return result


# -----------------------------
# Sales Forecasting (Time Series)
# -----------------------------

class SalesHistoryPoint(BaseModel):
    date: str
    sales: float
    is_holiday: int = 0


class SalesForecastRequest(BaseModel):
    model_id: str = Field(..., min_length=1)
    target_entity_id: str = Field(..., min_length=1, description="Target entity id (e.g. SKU / StockCode)")
    forecast_horizon: int = Field(7, ge=1, le=365)

    # When historical_context is not provided, the service loads it from the dataset CSV.
    context_window_days: int = Field(60, ge=7, le=3650)
    historical_context: Optional[List[SalesHistoryPoint]] = None

    # Dataset mapping overrides (defaults are resolved from datasets.json + common names).
    target_entity_column: Optional[str] = None
    date_column: Optional[str] = None
    quantity_column: Optional[str] = None
    unit_price_column: Optional[str] = None

    # Business knobs
    sales_metric: str = Field("revenue", description="quantity | revenue")
    promotion_factor: float = Field(1.0, gt=0)
    holiday_country: Optional[str] = Field("GB", description="holidays package country code, e.g. GB/US/CN")
    rounding: str = Field("none", description="round | floor | none")
    clip_negative: bool = Field(True)


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
    """If a wrapper model is saved, unwrap to underlying estimator when possible."""
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

    # Simple momentum feature (avoid divide-by-zero).
    lag7 = feats["lag_7"]
    lag1 = feats["lag_1"]
    feats["mom_7"] = float((lag1 - lag7) / (abs(lag7) + 1e-6))
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
        metric = (req.sales_metric or "quantity").lower()
        if metric not in {"quantity", "revenue"}:
            raise HTTPException(status_code=400, detail="sales_metric must be 'quantity' or 'revenue'")

        # 1) Resolve model (or baseline).
        model_obj, model_version = _sales_forecast_resolve_model(req.model_id)

        # 2) Load dataset config + history context.
        config = load_datasets_config()
        dataset_config = next((d for d in config.get("datasets", []) if d["id"] == dataset_id), None)
        if not dataset_config:
            raise HTTPException(status_code=404, detail="Dataset not found")

        history_points: List[Dict[str, Any]] = []
        reference_date: Optional[pd.Timestamp] = None

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
        else:
            csv_path = _get_dataset_csv_path(dataset_id, dataset_config)
            if not csv_path.exists():
                raise HTTPException(status_code=404, detail=f"CSV file not found: {csv_path}")

            encoding, header_cols = _read_csv_header(csv_path)
            date_col_req = req.date_column or dataset_config.get("time_column") or "InvoiceDate"
            entity_col_req = req.target_entity_column or "StockCode"
            qty_col_req = req.quantity_column or "Quantity"
            price_col_req = req.unit_price_column or "UnitPrice"

            date_col = _resolve_csv_column(header_cols, str(date_col_req))
            entity_col = _resolve_csv_column(header_cols, str(entity_col_req))
            qty_col = _resolve_csv_column(header_cols, str(qty_col_req))
            price_col = _resolve_csv_column(header_cols, str(price_col_req)) if metric == "revenue" else None

            s = _sales_forecast_daily_series_from_csv(
                csv_path=csv_path,
                encoding=encoding,
                date_col=date_col,
                entity_col=entity_col,
                target_entity_id=req.target_entity_id,
                metric=metric,
                qty_col=qty_col,
                price_col=price_col,
            )

            if s.empty:
                # cold start: use dataset-level mean daily sales
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
                history_sales = [float(fallback_mean)] * max(7, int(req.context_window_days))
            else:
                # Use last available day as "today" for forecasting.
                reference_date = pd.Timestamp(s.index.max())

                # Clip negative daily totals (returns etc.) because "sales" can't be negative.
                s = s.clip(lower=0.0)

                start = reference_date - pd.Timedelta(days=int(req.context_window_days) - 1)
                s_ctx = s.loc[s.index >= start]
                for d, v in s_ctx.items():
                    history_points.append(
                        {
                            "date": pd.Timestamp(d).strftime("%Y-%m-%d"),
                            "sales": float(v),
                            "is_holiday": _sales_forecast_is_holiday(pd.Timestamp(d), req.holiday_country),
                        }
                    )
                history_sales = [float(v) for v in s_ctx.values.tolist()]

        if "history_sales" not in locals():
            history_sales = [float(p["sales"]) for p in history_points]

        # 3) Forecast with recursive feature generation.
        horizon = int(req.forecast_horizon)
        sales_series = list(history_sales)
        forecast_rows: List[Dict[str, Any]] = []

        # Confidence interval width from recent volatility (simple heuristic).
        hist_tail = np.asarray(sales_series[-30:], dtype=float) if sales_series else np.asarray([], dtype=float)
        sigma = float(hist_tail.std(ddof=0)) if hist_tail.size else 0.0
        z = 1.96

        if reference_date is None:
            # If we have no reference_date (cold start), start "tomorrow" from today().
            reference_date = pd.Timestamp(datetime.now().date())

        for step in range(1, horizon + 1):
            ts = reference_date + pd.Timedelta(days=step)
            feats = _sales_forecast_compute_features(ts=ts, sales_history=sales_series, holiday_country=req.holiday_country)

            if model_obj is None:
                # Baseline: 7-day moving average.
                raw_pred = float(feats.get("rolling_mean_7", 0.0))
            else:
                raw_pred = _sales_forecast_predict_one(model_obj, feats)

            pred = _sales_forecast_post_process(
                value=raw_pred,
                promotion_factor=req.promotion_factor,
                clip_negative=req.clip_negative,
                rounding=req.rounding,
            )

            sales_series.append(float(pred))

            ci_lower = max(0.0, pred - z * sigma)
            ci_upper = max(0.0, pred + z * sigma)

            forecast_rows.append(
                {
                    "date": ts.strftime("%Y-%m-%d"),
                    "predicted_sales": pred,
                    "confidence_interval": {
                        "lower": _sales_forecast_post_process(value=ci_lower, promotion_factor=1.0, clip_negative=True, rounding=req.rounding),
                        "upper": _sales_forecast_post_process(value=ci_upper, promotion_factor=1.0, clip_negative=True, rounding=req.rounding),
                    },
                }
            )

        # 4) Trend summary (simple).
        last7 = np.asarray(history_sales[-7:], dtype=float) if history_sales else np.asarray([], dtype=float)
        last7_avg = float(last7.mean()) if last7.size else 0.0
        fut = np.asarray([r["predicted_sales"] for r in forecast_rows], dtype=float)
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

        return {
            "status": "success",
            "meta": {
                "model_version": model_version,
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
            "forecast": forecast_rows,
            "trend_summary": trend_summary,
            # Helpful for UI visualization / debugging (optional fields).
            "history": history_points[-30:],
        }
    except HTTPException:
        raise
    except Exception as e:
        # Surface the root cause to the client for easier debugging in demo environments.
        raise HTTPException(status_code=500, detail=f"Sales forecast failed: {e}")
