from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score


def _safe_scores(
    X: pd.DataFrame,
    labels: np.ndarray,
    ignore_noise_for_dbscan: bool = False,
) -> tuple[float | None, float | None]:
    """
    Safely calculate Silhouette and Davies-Bouldin scores.

    For DBSCAN, noise points (-1) can optionally be excluded before evaluation.
    """
    X_eval = X
    labels_eval = labels

    if ignore_noise_for_dbscan:
        mask = labels != -1
        X_eval = X.loc[mask]
        labels_eval = labels[mask]

    unique_labels = set(labels_eval.tolist())

    if len(unique_labels) < 2 or len(labels_eval) <= len(unique_labels):
        return None, None

    silhouette = silhouette_score(X_eval, labels_eval)
    davies_bouldin = davies_bouldin_score(X_eval, labels_eval)

    return float(silhouette), float(davies_bouldin)


def find_optimal_k(
    X: pd.DataFrame,
    min_k: int = 2,
    max_k: int = 8,
    random_state: int = 42,
) -> pd.DataFrame:
    """Evaluate K-Means over a range of K values."""
    max_allowed = min(max_k, len(X) - 1)

    records = []

    for k in range(min_k, max_allowed + 1):
        model = KMeans(
            n_clusters=k,
            random_state=random_state,
            n_init=20,
        )

        labels = model.fit_predict(X)
        silhouette, db_index = _safe_scores(X, labels)

        records.append(
            {
                "k": k,
                "inertia": float(model.inertia_),
                "silhouette_score": silhouette,
                "davies_bouldin_index": db_index,
            }
        )

    return pd.DataFrame(records)


def select_best_k(k_results: pd.DataFrame) -> int:
    """
    Select K using the highest Silhouette Score.

    This is simple and reproducible for a portfolio/demo project.
    """
    valid = k_results.dropna(subset=["silhouette_score"])

    if valid.empty:
        raise ValueError("Unable to determine a valid K.")

    best_row = valid.sort_values(
        ["silhouette_score", "davies_bouldin_index"],
        ascending=[False, True],
    ).iloc[0]

    return int(best_row["k"])


def run_kmeans(
    X: pd.DataFrame,
    n_clusters: int,
    random_state: int = 42,
) -> tuple[np.ndarray, dict]:
    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=20,
    )

    labels = model.fit_predict(X)
    silhouette, db_index = _safe_scores(X, labels)

    metrics = {
        "model": "K-Means",
        "clusters": len(set(labels.tolist())),
        "silhouette_score": silhouette,
        "davies_bouldin_index": db_index,
    }

    return labels, metrics


def run_hierarchical(
    X: pd.DataFrame,
    n_clusters: int,
) -> tuple[np.ndarray, dict]:
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage="ward",
    )

    labels = model.fit_predict(X)
    silhouette, db_index = _safe_scores(X, labels)

    metrics = {
        "model": "Hierarchical",
        "clusters": len(set(labels.tolist())),
        "silhouette_score": silhouette,
        "davies_bouldin_index": db_index,
    }

    return labels, metrics


def run_dbscan(
    X: pd.DataFrame,
    eps: float = 3.0,
    min_samples: int = 2,
) -> tuple[np.ndarray, dict]:
    """
    Run DBSCAN.

    eps is intentionally configurable because density-based clustering is
    sensitive to feature dimensionality and scale.
    """
    model = DBSCAN(
        eps=eps,
        min_samples=min_samples,
    )

    labels = model.fit_predict(X)

    non_noise_clusters = len(set(labels.tolist()) - {-1})
    noise_points = int((labels == -1).sum())

    silhouette, db_index = _safe_scores(
        X,
        labels,
        ignore_noise_for_dbscan=True,
    )

    metrics = {
        "model": "DBSCAN",
        "clusters": non_noise_clusters,
        "noise_points": noise_points,
        "silhouette_score": silhouette,
        "davies_bouldin_index": db_index,
    }

    return labels, metrics
