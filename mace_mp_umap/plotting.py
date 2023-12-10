import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd


def get_colors_for_training_data(training_data_df, test_data_df):
    if training_data_df["num_neighbours"].isnull().all():
        return ["C0"] * len(training_data_df), None
    min_val = min(
        training_data_df["num_neighbours"].min(), test_data_df["num_neighbours"].min()
    )
    max_val = max(
        training_data_df["num_neighbours"].max(), test_data_df["num_neighbours"].max()
    )
    norm = mcolors.Normalize(vmin=min_val, vmax=max_val)
    cmap = plt.get_cmap("viridis")
    colors = cmap(norm(training_data_df["num_neighbours"]))
    return colors, norm


def plot_dimensionality_reduction(
    training_data_df: pd.DataFrame, test_data_df: pd.DataFrame, num_layers: int
) -> plt.Figure:
    fig, axes = plt.subplots(num_layers, 2, figsize=(12, 8 * num_layers))
    colors, norm = get_colors_for_training_data(training_data_df, test_data_df)
    if norm is not None:
        cbar_ax = fig.add_axes([0.2, 0.95, 0.6, 0.02])
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
        sm.set_array([])
        fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
        cbar_ax.set_title("num_neighbours", size=10, loc="left")

    for i in range(num_layers):
        tag = f"layer_{i}"
        ax = axes[i]
        ax[0].scatter(
            training_data_df[f"{tag}_umap_1"],
            training_data_df[f"{tag}_umap_2"],
            s=30,
            alpha=0.8,
            c=colors,
            rasterized=True,
        )
        ax[1].scatter(
            training_data_df[f"{tag}_pca_1"],
            training_data_df[f"{tag}_pca_2"],
            s=30,
            alpha=0.8,
            c=colors,
            rasterized=True,
        )
        ax[0].set_title(f"UMAP {tag}")
        ax[1].set_title(f"PCA {tag}")

    if test_data_df is not None:
        for i in range(num_layers):
            tag = f"layer_{i}"
            ax = axes[i]
            ax[0].scatter(
                test_data_df[f"{tag}_umap_1"],
                test_data_df[f"{tag}_umap_2"],
                s=30,
                alpha=0.8,
                c="none",
                edgecolors="red",
                linewidths=2,
                rasterized=True,
            )
            ax[1].scatter(
                test_data_df[f"{tag}_pca_1"],
                test_data_df[f"{tag}_pca_2"],
                s=30,
                alpha=0.8,
                c="none",
                edgecolors="red",
                linewidths=2,
                rasterized=True,
            )

    return fig
