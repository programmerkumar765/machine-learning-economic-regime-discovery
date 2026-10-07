from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA


def save_growth_boxplot(
    growth_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 8))
    plt.boxplot(
        growth_df.values,
        vert=False,
        labels=growth_df.columns,
    )
    plt.title("GDP Growth Rate Distribution by Country")
    plt.xlabel("Annual GDP Growth Rate (%)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_correlation_heatmap(
    growth_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    correlation = growth_df.corr()

    plt.figure(figsize=(14, 10))
    sns.heatmap(
        correlation,
        cmap="coolwarm",
        center=0,
        linewidths=0.25,
    )
    plt.title("Correlation of GDP Growth Patterns")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_k_selection_plot(
    k_results: pd.DataFrame,
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(
        k_results["k"],
        k_results["inertia"],
        marker="o",
        label="Inertia",
    )
    ax1.set_xlabel("Number of Clusters (K)")
    ax1.set_ylabel("Inertia")

    ax2 = ax1.twinx()
    ax2.plot(
        k_results["k"],
        k_results["silhouette_score"],
        marker="s",
        linestyle="--",
        label="Silhouette Score",
    )
    ax2.set_ylabel("Silhouette Score")

    plt.title("K-Means Model Selection")
    fig.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_pca_clusters(
    X: pd.DataFrame,
    labels,
    title: str,
    output_path: str | Path,
) -> pd.DataFrame:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pca = PCA(n_components=2)
    components = pca.fit_transform(X)

    pca_df = pd.DataFrame(
        {
            "Country": X.index,
            "PCA1": components[:, 0],
            "PCA2": components[:, 1],
            "Cluster": labels,
        }
    )

    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=pca_df,
        x="PCA1",
        y="PCA2",
        hue="Cluster",
        palette="viridis",
        s=100,
    )

    for _, row in pca_df.iterrows():
        plt.annotate(
            row["Country"],
            (row["PCA1"], row["PCA2"]),
            fontsize=8,
            xytext=(4, 3),
            textcoords="offset points",
        )

    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return pca_df


def save_dendrogram(
    X: pd.DataFrame,
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    linkage_matrix = linkage(X.values, method="ward")

    plt.figure(figsize=(12, 7))
    dendrogram(
        linkage_matrix,
        labels=X.index.tolist(),
        leaf_rotation=90,
    )
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Country")
    plt.ylabel("Ward Distance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
