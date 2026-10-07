from pathlib import Path

import pandas as pd

from data_loader import load_gdp_data, validate_dataset
from preprocessing import (
    compute_growth_rates,
    winsorize_growth_rates,
    build_country_feature_matrix,
)
from clustering import (
    find_optimal_k,
    select_best_k,
    run_kmeans,
    run_hierarchical,
    run_dbscan,
)
from visualization import (
    save_growth_boxplot,
    save_correlation_heatmap,
    save_k_selection_plot,
    save_pca_clusters,
    save_dendrogram,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "Europe_GDP.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
IMAGE_DIR = PROJECT_ROOT / "images"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load and validate
    df = load_gdp_data(DATA_PATH)
    validation = validate_dataset(df)

    print("\nDataset validation")
    for key, value in validation.items():
        print(f"{key}: {value}")

    # 2. Feature engineering
    growth = compute_growth_rates(df)

    # 3. Outlier treatment
    growth_winsorized = winsorize_growth_rates(growth)

    # 4. Country-level feature matrix
    _, X = build_country_feature_matrix(growth_winsorized)

    # Save intermediate features for transparency/reproducibility.
    growth.to_csv(OUTPUT_DIR / "gdp_growth_rates.csv")
    growth_winsorized.to_csv(
        OUTPUT_DIR / "gdp_growth_rates_winsorized.csv"
    )
    X.to_csv(OUTPUT_DIR / "country_features_scaled.csv")

    # 5. Exploratory visualizations
    save_growth_boxplot(
        growth_winsorized,
        IMAGE_DIR / "growth_boxplot.png",
    )

    save_correlation_heatmap(
        growth_winsorized,
        IMAGE_DIR / "correlation_heatmap.png",
    )

    # 6. K-Means model selection
    k_results = find_optimal_k(X)
    best_k = select_best_k(k_results)

    print(f"\nSelected K: {best_k}")

    k_results.to_csv(
        OUTPUT_DIR / "kmeans_model_selection.csv",
        index=False,
    )

    save_k_selection_plot(
        k_results,
        IMAGE_DIR / "kmeans_model_selection.png",
    )

    # 7. Train clustering models
    kmeans_labels, kmeans_metrics = run_kmeans(
        X,
        n_clusters=best_k,
    )

    hierarchical_labels, hierarchical_metrics = run_hierarchical(
        X,
        n_clusters=best_k,
    )

    dbscan_labels, dbscan_metrics = run_dbscan(
        X,
        eps=3.0,
        min_samples=2,
    )

    # 8. Save cluster assignments
    assignments = pd.DataFrame(
        {
            "Country": X.index,
            "KMeans_Cluster": kmeans_labels,
            "Hierarchical_Cluster": hierarchical_labels,
            "DBSCAN_Cluster": dbscan_labels,
        }
    )

    assignments.to_csv(
        OUTPUT_DIR / "cluster_assignments.csv",
        index=False,
    )

    # 9. Compare models
    metrics = pd.DataFrame(
        [
            kmeans_metrics,
            hierarchical_metrics,
            dbscan_metrics,
        ]
    )

    metrics.to_csv(
        OUTPUT_DIR / "model_comparison.csv",
        index=False,
    )

    print("\nModel comparison")
    print(metrics.to_string(index=False))

    print("\nCluster assignments")
    print(assignments.to_string(index=False))

    # 10. PCA visualizations
    save_pca_clusters(
        X,
        kmeans_labels,
        "Economic Regimes - K-Means Clustering",
        IMAGE_DIR / "kmeans_pca_clusters.png",
    )

    save_pca_clusters(
        X,
        hierarchical_labels,
        "Economic Regimes - Hierarchical Clustering",
        IMAGE_DIR / "hierarchical_pca_clusters.png",
    )

    save_pca_clusters(
        X,
        dbscan_labels,
        "Economic Regimes - DBSCAN",
        IMAGE_DIR / "dbscan_pca_clusters.png",
    )

    save_dendrogram(
        X,
        IMAGE_DIR / "hierarchical_dendrogram.png",
    )

    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    print(f"Images saved to: {IMAGE_DIR}")


if __name__ == "__main__":
    main()
