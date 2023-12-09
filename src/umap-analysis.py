import typing as t
from collections import defaultdict
from enum import Enum

import ase
import ase.io as aio
import chemiscope
import matplotlib.pyplot as plt
import mendeleev
import numpy as np
import pandas as pd
import typer
import umap
from ase.neighborlist import neighbor_list
from mace.calculators import mace_mp
from sklearn.decomposition import PCA
from tqdm import tqdm
from typing_extensions import Annotated

ELEMENT_DICT = {}


def apply_dimensionality_reduction(df, tag, feature_slice, umap_reducer, pca_reducer):
    descriptors = np.vstack(df["descriptor"])[:, feature_slice]
    embeddings = umap_reducer.transform(descriptors)
    embeddings_pca = pca_reducer.transform(descriptors)
    df[f"{tag}_umap_1"] = embeddings[:, 0]
    df[f"{tag}_umap_2"] = embeddings[:, 1]
    df[f"{tag}_pca_1"] = embeddings_pca[:, 0]
    df[f"{tag}_pca_2"] = embeddings_pca[:, 1]


def calculate_descriptors(atoms: t.List[ase.Atoms | ase.Atom], calc, cutoffs) -> None:
    for mol in tqdm(atoms):
        descriptors = calc.get_descriptors(mol, invariants_only=True)
        mol.arrays["mace_descriptors"] = descriptors
        cut = np.array([cutoffs[x] for x in mol.symbols])
        num_neighbours = calculate_local_atom_density(mol, cut)
        mol.arrays["num_neighbours"] = num_neighbours


def calculate_local_atom_density(atoms: ase.Atoms, cutoff=1.8) -> np.ndarray:
    """
    Calculate local atom number density using ase.neighborlist.neighbor_list
    """
    edge_array = neighbor_list("ij", atoms, cutoff=cutoff, self_interaction=True)
    edge_array = np.stack(edge_array, axis=1)

    neighbour_numbers = np.bincount(edge_array[:, 0])
    return neighbour_numbers


def convert_to_dataframe(atoms: t.List[ase.Atoms | ase.Atom]) -> pd.DataFrame:
    data = []
    for mol_idx, mol in enumerate(tqdm(atoms)):
        descs = mol.arrays["mace_descriptors"]
        num_neighbours = mol.arrays["num_neighbours"]
        elements = mol.symbols
        for idx, d in enumerate(descs):
            data.append(
                {
                    "mp_id": mol.info["id"] if "id" in mol.info else None,
                    "structure_index": mol_idx,
                    "atom_index": idx,
                    "element": elements[idx],
                    "descriptor": d,
                    "num_neighbours": num_neighbours[idx],
                }
            )
    return pd.DataFrame(data)


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


def filter_atoms(atoms, element_subset, filtering_type):
    if filtering_type == "none":
        return True
    elif filtering_type == "exclusive":
        atom_symbols = np.unique(atoms.symbols)
        return all(
            [x in element_subset for x in atom_symbols]
        )  # atoms must *only* contain elements in the subset
    elif filtering_type == "inclusive":
        atom_symbols = np.unique(atoms.symbols)
        return all(
            [x in atom_symbols for x in element_subset]
        )  # atoms must *at least* contain elements in the subset


def fit_dimensionality_reduction(df, tag, feature_slice, random_state=42):
    umap_reducer, pca_reducer = get_reducers(random_state)
    descriptors = np.vstack(df["descriptor"])
    print(f"Before slice {descriptors.shape}")
    descriptors = np.vstack(df["descriptor"])[:, feature_slice]
    print(f"After slice {descriptors.shape}")
    embeddings = umap_reducer.fit_transform(descriptors)
    embeddings_pca = pca_reducer.fit_transform(descriptors)
    df[f"{tag}_umap_1"] = embeddings[:, 0]
    df[f"{tag}_umap_2"] = embeddings[:, 1]
    df[f"{tag}_pca_1"] = embeddings_pca[:, 0]
    df[f"{tag}_pca_2"] = embeddings_pca[:, 1]
    return umap_reducer, pca_reducer


def get_cleaned_dataframe(path, calc, element_subset, element_cutoffs, filtering_type):
    data = aio.read(path, index=":", format="extxyz")
    print(f"Loaded {len(data)} structures")
    filtered_data = list(
        filter(lambda x: filter_atoms(x, element_subset, filtering_type), tqdm(data))
    )
    print(f"Filtered to {len(filtered_data)} structures")
    calculate_descriptors(filtered_data, calc, element_cutoffs)
    df = convert_to_dataframe(filtered_data)
    df = remove_non_unique_environments(df)
    return filtered_data, df


def get_group_and_period(symbol):
    if symbol not in ELEMENT_DICT:
        element = mendeleev.element(symbol)
        group = element.group_id
        group = -1 if group is None else group
        ELEMENT_DICT[symbol] = (group, element.period)
    return ELEMENT_DICT[symbol]


def get_reducers(random_state=42):
    umap_projection = umap.UMAP(
        n_components=2,
        n_neighbors=50,
        min_dist=0.1,
        metric="euclidean",
        random_state=random_state,
    )
    pca_projection = PCA(n_components=2)
    return umap_projection, pca_projection


def get_layer_specific_feature_slices(calc):
    num_layers = calc.models[0].num_interactions
    irreps_out = calc.models[0].products[0].linear.__dict__["irreps_out"]
    l_max = irreps_out.lmax
    features_per_layer = irreps_out.dim // (l_max + 1) ** 2
    slices = [slice(0, (i + 1) * features_per_layer) for i in range(num_layers)]
    return slices


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


def remove_non_unique_environments(df, decimals=4):
    descriptors = np.vstack(df["descriptor"])
    _, indices = np.unique(np.round(descriptors, decimals), return_index=True, axis=0)
    return df.iloc[indices].copy().sort_values(by="mp_id")


class FilterType(str, Enum):
    exclusive = "exclusive"
    inclusive = "inclusive"
    none = "none"


def main(
    training_data_path: str = "./universal_train_corr.xyz",
    test_data_path: str | None = None,
    run_name: str | None = None,
    element_subset: t.List[str] = typer.Option(
        ["C", "O"], help="List of elements to include in the subset."
    ),
    element_cutoffs: t.List[float] = typer.Option(
        [3.0, 3.0], help="List of cutoff values for each element in the subset."
    ),
    filtering: Annotated[
        FilterType,
        typer.Option(
            case_sensitive=False,
            help="Whether to filter out structures that contain elements not in the subset or to include them.",
        ),
    ] = FilterType.exclusive,
):
    calc = mace_mp(
        model="medium",
        device="cuda",
        default_dtype="float64",
    )
    print("element_subset", element_subset)
    cutoffs = dict(zip(element_subset, element_cutoffs))

    # If filtering is inclusive, make all cutoffs the same as the first one
    if filtering == FilterType.inclusive:
        cutoffs = defaultdict(lambda: element_cutoffs[0], cutoffs)
    elif filtering == FilterType.exclusive:
        assert len(element_subset) == len(
            element_cutoffs
        ), "Must provide a cutoff for each element in the subset"

    train_atoms, training_data_df = get_cleaned_dataframe(
        training_data_path, calc, element_subset, cutoffs, filtering_type=filtering
    )
    if test_data_path is not None:
        test_atoms, test_data_df = get_cleaned_dataframe(
            test_data_path, calc, element_subset, cutoffs, filtering_type="none"
        )
    else:
        test_atoms = []
        test_data_df = None

    slices = get_layer_specific_feature_slices(calc)
    reducers = []
    for i, sli in enumerate(slices):
        tag = f"layer_{i}"
        umap_reducer, pca_reducer = fit_dimensionality_reduction(
            training_data_df, tag, sli
        )
        reducers.append((umap_reducer, pca_reducer))
        if test_data_df is not None:
            apply_dimensionality_reduction(
                test_data_df, tag, sli, umap_reducer, pca_reducer
            )

    figure = plot_dimensionality_reduction(training_data_df, test_data_df, len(slices))
    if run_name is None:
        tag = ""
    else:
        tag = f"_{run_name}"
    figure.savefig(f"dim_reduction{tag}.pdf", dpi=300)

    if test_data_df is not None:
        closest_training_points = find_closest_training_points(
            training_data_df, test_data_df
        )
        closest_training_points.to_csv(f"closest_mp_ids{tag}.csv", index=False)

    create_chemiscope_input_file(train_atoms, test_atoms, reducers, tag)


app = typer.Typer()
app.command()(main)

if __name__ == "__main__":
    app()
