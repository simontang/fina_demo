"""
Customer segmentation (K-Means) utilities.

This module uses scikit-learn for K-Means + StandardScaler + silhouette scoring,
and wraps it with practical data-cleaning helpers for customer clustering workflows.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


class CustomerSegmentationError(Exception):
    """Raised when inputs are invalid or clustering cannot be performed."""


@dataclass(frozen=True)
class _ScaleParams:
    mean: np.ndarray
    std: np.ndarray


def _as_dataframe(source_data: Union[pd.DataFrame, str, Path]) -> pd.DataFrame:
    if isinstance(source_data, pd.DataFrame):
        return source_data.copy()
    path = Path(source_data)
    if not path.exists():
        raise CustomerSegmentationError(f"source_data path not found: {path}")
    try:
        return pd.read_csv(path)
    except Exception as e:  # pragma: no cover (depends on pandas IO errors)
        raise CustomerSegmentationError(f"failed to read CSV: {path}: {e}") from e


def _validate_selected_features(df: pd.DataFrame, selected_features: List[str]) -> List[str]:
    if not isinstance(selected_features, list) or not selected_features:
        raise CustomerSegmentationError("selected_features must be a non-empty list of column names")
    features: List[str] = []
    for f in selected_features:
        if not isinstance(f, str) or not f.strip():
            raise CustomerSegmentationError("selected_features contains an invalid column name")
        features.append(f)
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise CustomerSegmentationError(f"selected_features not found in data: {missing}")
    return features


def _drop_zero_variance_features(df_feat: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    # Treat "almost constant" as constant to avoid numerical issues.
    std = df_feat.std(axis=0, ddof=0)
    to_drop = std[std <= 0].index.tolist()
    if to_drop:
        df_feat = df_feat.drop(columns=to_drop)
    return df_feat, to_drop


def _fill_missing_with_mean(df_feat: pd.DataFrame) -> pd.DataFrame:
    means = df_feat.mean(axis=0, skipna=True)
    # If a column is entirely NaN, mean will be NaN -> not recoverable with mean-impute.
    bad_cols = means[means.isna()].index.tolist()
    if bad_cols:
        raise CustomerSegmentationError(f"columns are all-NaN after numeric coercion: {bad_cols}")
    return df_feat.fillna(means.to_dict())


def _remove_outliers_zscore(df_feat: pd.DataFrame, threshold: float) -> Tuple[pd.Series, int]:
    if threshold <= 0:
        return pd.Series(True, index=df_feat.index), 0
    means = df_feat.mean(axis=0)
    stds = df_feat.std(axis=0, ddof=0).replace(0, np.nan)
    z = (df_feat - means) / stds
    keep_mask = ~(z.abs() > float(threshold)).any(axis=1)
    removed = int((~keep_mask).sum())
    return keep_mask, removed


def _log_transform_right_skewed(
    df_feat: pd.DataFrame,
    *,
    skew_threshold: float = 1.0,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    info: Dict[str, Dict[str, float]] = {}
    df_out = df_feat.copy()

    for col in df_feat.columns:
        s = float(df_feat[col].skew())
        if not np.isfinite(s) or s <= skew_threshold:
            continue

        min_v = float(df_feat[col].min())
        shift = float(-min_v) if min_v < 0 else 0.0

        # Ensure domain for log1p is valid (x + shift > -1).
        x = df_feat[col].astype(float) + shift
        if float(x.min()) <= -1.0:
            # Extremely negative values: skip rather than silently producing NaNs.
            continue

        df_out[col] = np.log1p(x)
        info[col] = {"skew": s, "shift": shift}

    return df_out, info


def _standardize(df_feat: pd.DataFrame) -> Tuple[np.ndarray, _ScaleParams]:
    X = df_feat.to_numpy(dtype=float, copy=True)
    mean = X.mean(axis=0)
    std = X.std(axis=0, ddof=0)
    std_safe = np.where(std == 0, 1.0, std)
    X = (X - mean) / std_safe
    return X, _ScaleParams(mean=mean, std=std_safe)


def _pairwise_euclidean_distances(X: np.ndarray) -> np.ndarray:
    # dist(i, j) = sqrt(||xi||^2 + ||xj||^2 - 2*xi·xj)
    x2 = np.sum(X * X, axis=1)
    d2 = x2[:, None] + x2[None, :] - 2.0 * (X @ X.T)
    d2 = np.maximum(d2, 0.0)
    return np.sqrt(d2, dtype=float)


def _silhouette_score_from_distance_matrix(dist: np.ndarray, labels: np.ndarray) -> float:
    labels = labels.astype(int, copy=False)
    m = int(labels.shape[0])
    if m == 0:
        return float("nan")

    unique = np.unique(labels)
    if unique.size < 2:
        return float("nan")

    # Cluster sizes in the sample.
    counts = {int(c): int((labels == c).sum()) for c in unique.tolist()}
    singletons = np.array([counts[int(l)] == 1 for l in labels], dtype=bool)

    a = np.zeros(m, dtype=float)
    b = np.full(m, np.inf, dtype=float)

    for c in unique:
        idx_c = np.where(labels == c)[0]
        size_c = int(idx_c.size)
        if size_c == 0:
            continue

        if size_c > 1:
            dist_cc = dist[np.ix_(idx_c, idx_c)]
            a[idx_c] = np.sum(dist_cc, axis=1) / float(size_c - 1)
        else:
            a[idx_c] = 0.0

        mean_to_c = np.mean(dist[:, idx_c], axis=1)
        not_c = labels != c
        b[not_c] = np.minimum(b[not_c], mean_to_c[not_c])

    denom = np.maximum(a, b)
    s = np.zeros(m, dtype=float)
    ok = np.isfinite(denom) & (denom > 0)
    s[ok] = (b[ok] - a[ok]) / denom[ok]

    # By convention, silhouette for singleton clusters is 0.
    s[singletons] = 0.0
    return float(np.nanmean(s))


def _kmeans_plus_plus_init(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = int(X.shape[0])
    if k <= 0 or k > n:
        raise CustomerSegmentationError("k must be in [1, n_samples]")

    centers = np.empty((k, X.shape[1]), dtype=float)
    first = int(rng.integers(0, n))
    centers[0] = X[first]

    # Track squared distance to closest chosen center.
    closest_d2 = np.sum((X - centers[0]) ** 2, axis=1)
    for i in range(1, k):
        total = float(np.sum(closest_d2))
        if not np.isfinite(total) or total <= 0:
            # Fallback: data collapsed; pick random centers.
            centers[i:] = X[rng.choice(n, size=(k - i), replace=False)]
            break

        probs = closest_d2 / total
        idx = int(rng.choice(n, p=probs))
        centers[i] = X[idx]

        d2_new = np.sum((X - centers[i]) ** 2, axis=1)
        closest_d2 = np.minimum(closest_d2, d2_new)

    return centers


def _kmeans_fit(
    X: np.ndarray,
    k: int,
    *,
    rng: np.random.Generator,
    max_iter: int = 300,
    tol: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray, float, int, bool, int]:
    n, p = X.shape
    centers = _kmeans_plus_plus_init(X, k=k, rng=rng)

    converged = False
    empty_reseeds = 0

    for it in range(1, int(max_iter) + 1):
        x2 = np.sum(X * X, axis=1)  # (n,)
        c2 = np.sum(centers * centers, axis=1)  # (k,)
        d2 = x2[:, None] + c2[None, :] - 2.0 * (X @ centers.T)  # (n, k)
        d2 = np.maximum(d2, 0.0)

        labels = np.argmin(d2, axis=1).astype(int)

        counts = np.bincount(labels, minlength=k).astype(int)
        sums = np.zeros((k, p), dtype=float)
        np.add.at(sums, labels, X)

        new_centers = centers.copy()
        nonempty = counts > 0
        new_centers[nonempty] = sums[nonempty] / counts[nonempty][:, None]

        empty = np.where(~nonempty)[0]
        if empty.size:
            empty_reseeds += int(empty.size)
            for j in empty.tolist():
                new_centers[j] = X[int(rng.integers(0, n))]

        shift = np.max(np.linalg.norm(new_centers - centers, axis=1))
        centers = new_centers
        if float(shift) <= float(tol):
            converged = True
            break

    # Final inertia
    x2 = np.sum(X * X, axis=1)
    c2 = np.sum(centers * centers, axis=1)
    d2 = x2[:, None] + c2[None, :] - 2.0 * (X @ centers.T)
    d2 = np.maximum(d2, 0.0)
    labels = np.argmin(d2, axis=1).astype(int)
    inertia = float(np.sum(d2[np.arange(n), labels]))

    return labels, centers, inertia, it, converged, empty_reseeds


def _format_pct(ratio: float) -> str:
    pct = float(ratio) * 100.0
    s = f"{pct:.1f}".rstrip("0").rstrip(".")
    return f"{s}%"


def _cluster_label_suggestion(
    cluster_means: Dict[int, Dict[str, float]],
    *,
    recency_feature_names: Iterable[str],
) -> Dict[int, str]:
    """Suggest simple business labels using mean R/F/M-style heuristics."""
    if not cluster_means:
        return {}

    # Pick one "recency-like" feature if present, otherwise skip recency logic.
    rec_feature = next((f for f in recency_feature_names if all(f in m for m in cluster_means.values())), None)

    ids = sorted(cluster_means.keys())
    features = sorted({k for m in cluster_means.values() for k in m.keys()})

    # Rank-based scoring to avoid unit/scale domination (e.g. Monetary dwarfing Frequency).
    score = {cid: 0.0 for cid in ids}
    for f in features:
        vals = {cid: float(cluster_means[cid].get(f, 0.0)) for cid in ids}
        if rec_feature and f == rec_feature:
            ordered = sorted(ids, key=lambda cid: vals[cid])  # lower is better
        else:
            ordered = sorted(ids, key=lambda cid: vals[cid], reverse=True)  # higher is better
        for rank, cid in enumerate(ordered):
            # Best gets the largest points.
            score[cid] += float(len(ordered) - rank)

    best = max(ids, key=lambda i: score[i])
    worst = min(ids, key=lambda i: score[i])
    labels = {cid: "潜力发展户" for cid in ids}
    labels[best] = "VIP 客户"
    labels[worst] = "流失风险客户"
    return labels


def segment_customers_kmeans(
    source_data: Union[pd.DataFrame, str, Path],
    selected_features: List[str],
    *,
    k_range: Tuple[int, int] = (3, 6),
    random_seed: int = 42,
    outlier_threshold: Optional[float] = 3.0,
    user_id_column: str = "user_id",
) -> Dict[str, Any]:
    """Run K-Means customer segmentation with automatic K selection.

    Args:
        source_data: Input customer dataset (DataFrame) or CSV path.
        selected_features: Feature columns used for clustering (e.g. R/F/M).
        k_range: (min_k, max_k) inclusive.
        random_seed: Random seed for reproducibility.
        outlier_threshold: If set, remove rows whose any feature Z-score exceeds this threshold.

    Returns:
        A JSON-serializable dict following the spec in the prompt.
    """
    try:
        # Prefer scikit-learn implementation for correctness/performance; keep a clear error when missing.
        try:
            from sklearn.cluster import KMeans
            from sklearn.exceptions import ConvergenceWarning
            from sklearn.metrics import silhouette_score
            from sklearn.preprocessing import StandardScaler
        except Exception as e:
            raise CustomerSegmentationError(
                "scikit-learn 未安装或不可用；请先安装 scikit-learn 后再运行聚类"
            ) from e

        df = _as_dataframe(source_data)
        features = _validate_selected_features(df, selected_features)

        # Coerce features to numeric; invalid parses become NaN and are mean-imputed.
        df_feat = df[features].apply(pd.to_numeric, errors="coerce")
        df_feat = _fill_missing_with_mean(df_feat)

        dropped_const: List[str] = []
        df_feat, dropped_const = _drop_zero_variance_features(df_feat)
        if df_feat.shape[1] == 0:
            raise CustomerSegmentationError("all selected_features have zero variance after cleaning")

        # Keep df aligned with current feature frame.
        df = df.loc[df_feat.index].copy()
        # Persist cleaned numeric features for later reporting/preview.
        for c in df_feat.columns:
            df[c] = df_feat[c].astype(float)

        removed_outliers = 0
        if outlier_threshold is not None:
            keep_mask, removed_outliers = _remove_outliers_zscore(df_feat, float(outlier_threshold))
            df_feat = df_feat.loc[keep_mask].copy()
            df = df.loc[keep_mask].copy()

        min_k, max_k = int(k_range[0]), int(k_range[1])
        if min_k < 2 or max_k < 2 or min_k > max_k:
            raise CustomerSegmentationError("k_range must be a tuple(min_k, max_k) with min_k>=2 and min_k<=max_k")
        n_samples = int(df_feat.shape[0])
        if n_samples < max_k:
            raise CustomerSegmentationError("数据量不足以进行聚类：样本数小于 k_range 最大值")

        # Log-transform highly right-skewed features (Kaggle-style).
        df_transformed, log_info = _log_transform_right_skewed(df_feat)

        # Standardize using scikit-learn to match typical ML workflows.
        X = df_transformed.to_numpy(dtype=float, copy=True)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        k_values = list(range(min_k, max_k + 1))
        elbow_curve_data: List[Dict[str, float]] = []
        silhouette_by_k: Dict[int, float] = {}
        labels_by_k: Dict[int, np.ndarray] = {}

        warnings: List[str] = []

        # Silhouette is O(m^2); keep bounded sample size for large datasets.
        max_silhouette_samples = 2000
        sample_size: Optional[int] = None if n_samples <= max_silhouette_samples else max_silhouette_samples
        if sample_size is not None:
            warnings.append(f"silhouette computed on a random sample of {sample_size} / {n_samples} rows for performance")

        for k in k_values:
            # KMeans supports k == n_samples; silhouette_score does not.
            if k >= n_samples:
                warnings.append(f"k={k}: silhouette undefined for k >= n_samples; treating silhouette as -1")

            import warnings as py_warnings

            with py_warnings.catch_warnings(record=True) as caught:
                py_warnings.simplefilter("always")
                model = KMeans(
                    n_clusters=int(k),
                    random_state=int(random_seed),
                    n_init=10,
                    max_iter=300,
                )
                labels = model.fit_predict(X_scaled).astype(int)

            for w in caught:
                if isinstance(w.message, ConvergenceWarning):
                    warnings.append(f"k={k}: k-means may not have fully converged ({w.message})")

            inertia = float(getattr(model, "inertia_", float("nan")))
            labels_by_k[k] = labels
            elbow_curve_data.append({"k": int(k), "sse": inertia})

            sil = -1.0
            if k < n_samples:
                try:
                    sil = float(
                        silhouette_score(
                            X_scaled,
                            labels,
                            metric="euclidean",
                            sample_size=sample_size,
                            random_state=int(random_seed),
                        )
                    )
                except Exception as e:
                    warnings.append(f"k={k}: failed to compute silhouette score ({e}); treating as -1")
                    sil = -1.0

            silhouette_by_k[k] = float(sil)

        # Prefer smaller k on ties for stability.
        best_k = max(k_values, key=lambda kk: (silhouette_by_k.get(kk, -1.0), -kk))
        best_silhouette = float(silhouette_by_k.get(best_k, -1.0))
        best_labels_full = labels_by_k[best_k]

        # Attach labels to cleaned (original-scale) data for reporting.
        final_features = df_feat.columns.tolist()
        df_out = df.copy()
        df_out["cluster_label"] = best_labels_full.astype(int)

        clusters_summary: List[Dict[str, Any]] = []
        cluster_means_for_label: Dict[int, Dict[str, float]] = {}
        total = int(df_out.shape[0])
        for cid in range(int(best_k)):
            cdf = df_out[df_out["cluster_label"] == cid]
            size = int(cdf.shape[0])
            if size == 0:
                continue

            characteristics: Dict[str, Dict[str, float]] = {}
            mean_map: Dict[str, float] = {}
            for f in final_features:
                v = pd.to_numeric(cdf[f], errors="coerce")
                mean_v = float(v.mean())
                std_v = float(v.std(ddof=0))
                characteristics[f] = {"mean": mean_v, "std": std_v}
                mean_map[f] = mean_v

            cluster_means_for_label[cid] = mean_map
            clusters_summary.append(
                {
                    "cluster_id": int(cid),
                    "size": size,
                    "percentage": _format_pct(size / total if total else 0.0),
                    "characteristics": characteristics,
                }
            )

        # Add label suggestions (optional but useful for business interpretation).
        suggestions = _cluster_label_suggestion(
            cluster_means_for_label,
            recency_feature_names=("Recency", "recency", "recency_days", "R", "r"),
        )
        for item in clusters_summary:
            cid = int(item["cluster_id"])
            if cid in suggestions:
                item["label_suggestion"] = suggestions[cid]

        # Preview: small sample of labeled rows (keep only id + label + features if possible).
        preview_n = 5
        cols: List[str] = []
        if user_id_column and user_id_column in df_out.columns:
            cols.append(user_id_column)
        cols.extend(["cluster_label", *final_features])
        preview_df = df_out[cols].head(preview_n)
        data_preview = preview_df.to_dict(orient="records")

        model_info = {
            "best_k": int(best_k),
            "best_silhouette_score": best_silhouette,
            "features_used": final_features,
        }

        resp: Dict[str, Any] = {
            "status": "success",
            "model_info": model_info,
            "clusters_summary": clusters_summary,
            "data_preview": data_preview,
            "elbow_curve_data": elbow_curve_data,
        }

        if dropped_const:
            warnings.append(f"dropped zero-variance features: {dropped_const}")
        if removed_outliers:
            warnings.append(f"removed outliers (z-score > {outlier_threshold}): {removed_outliers}")
        if log_info:
            resp["log_transform"] = log_info
        if warnings:
            # Deduplicate while preserving order.
            seen = set()
            resp["warnings"] = [w for w in warnings if not (w in seen or seen.add(w))]

        return resp
    except CustomerSegmentationError as e:
        return {"status": "error", "message": str(e)}
