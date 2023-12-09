def plot_dimensionality_reduction(training_data_df, test_data_df, num_layers):
    fig, axes = plt.subplots(num_layers, 2, figsize=(12, 8 * num_layers))
    training_min = training_data_df["num_neighbours"].min()
    training_max = training_data_df["num_neighbours"].max()
    test_min = np.inf if test_data_df is None else test_data_df["num_neighbours"].min()
    test_max = -np.inf if test_data_df is None else test_data_df["num_neighbours"].max()
    min_val = min(training_min, test_min)
    max_val = max(training_max, test_max)

    cbar_ax = fig.add_axes([0.2, 0.95, 0.6, 0.02])
    sm = plt.cm.ScalarMappable(
        cmap="viridis", norm=plt.Normalize(vmin=min_val, vmax=max_val)
    )
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
            c=training_data_df["num_neighbours"],
            cmap="viridis",
            vmin=min_val,
            vmax=max_val,
            rasterized=True,
        )
        ax[1].scatter(
            training_data_df[f"{tag}_pca_1"],
            training_data_df[f"{tag}_pca_2"],
            s=30,
            alpha=0.8,
            c=training_data_df["num_neighbours"],
            cmap="viridis",
            vmin=min_val,
            vmax=max_val,
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


def find_closest_training_points(training_df, test_df):
    structure_groups = test_df.groupby("structure_index")
    training_descriptors = np.vstack(
        training_df["descriptor"]
    )  # num_many_atoms x num_features
    training_norms = np.linalg.norm(training_descriptors, axis=1)
    mp_ids = training_df["mp_id"].values
    unique_mp_ids = np.unique(mp_ids)
    results = []
    for structure_index, structure_df in tqdm(structure_groups):
        structure_descriptors = np.vstack(
            structure_df["descriptor"]
        )  # num_atoms x num_features
        structure_distances = np.dot(
            structure_descriptors, training_descriptors.T
        )  # num_atoms x num_training_atoms
        structure_distances /= np.linalg.norm(structure_descriptors, axis=1)[:, None]
        structure_distances /= training_norms[None, :]
        elements = np.unique(structure_df["element"].values)
        for mp_id in unique_mp_ids:
            mp_id_mask = mp_ids == mp_id
            mp_id_distances = structure_distances[:, mp_id_mask]
            mp_id_distances = np.max(mp_id_distances, axis=1)
            per_element_average_distances = [
                np.mean(mp_id_distances[structure_df["element"] == x]) for x in elements
            ]
            per_element_results = dict(zip(elements, per_element_average_distances))
            results.append(
                {
                    "structure_index": structure_index,
                    "mp_id": mp_id,
                    "average_distance": np.mean(mp_id_distances),
                    "element_stratified_average_distance": np.mean(
                        per_element_average_distances
                    ),
                }
                | per_element_results
            )  # type: ignore
    return pd.DataFrame(results).sort_values(
        by=["structure_index", "element_stratified_average_distance"],
        ascending=[True, False],
    )


def create_chemiscope_input_file(train_atoms, test_atoms, reducers, run_name):
    all_atoms = train_atoms + test_atoms
    descriptors = np.vstack([mol.arrays["mace_descriptors"] for mol in all_atoms])
    pca = reducers[1][1].transform(descriptors)
    umap_emb = reducers[1][0].transform(descriptors)
    tag = ["train"] * sum([len(x) for x in train_atoms]) + ["test"] * sum(
        [len(x) for x in test_atoms]
    )
    symbols = np.hstack([mol.symbols for mol in all_atoms])
    groups_and_periods = np.array([get_group_and_period(x) for x in symbols])
    groups = groups_and_periods[:, 0]
    periods = groups_and_periods[:, 1]
    num_neighbours = np.hstack([mol.arrays["num_neighbours"] for mol in all_atoms])
    properties = {
        "PCA": {
            "target": "atom",
            "values": pca,
            "description": "PCA of per-atom representation of the structures",
        },
        "UMAP": {
            "target": "atom",
            "values": umap_emb,
            "description": "UMAP of per-atom representation of the structures",
        },
        "train_test": {
            "target": "atom",
            "values": tag,
            "description": "Whether the structure is in the training or test set",
        },
        "num_neighbours": {
            "target": "atom",
            "values": num_neighbours,
            "description": "Number of neighbours within cutoff",
        },
        "element": {
            "target": "atom",
            "values": symbols,
            "description": "Element of the atom",
        },
        "group": {
            "target": "atom",
            "values": groups,
            "description": "Group of the atom",
        },
        "period": {
            "target": "atom",
            "values": periods,
            "description": "Period of the atom",
        },
    }
    chemiscope.write_input(
        path=f"chemiscope{run_name}.json.gz",
        frames=all_atoms,
        properties=properties,
        # This is required to display properties with `target: "atom"`
        environments=chemiscope.all_atomic_environments(all_atoms),
    )
