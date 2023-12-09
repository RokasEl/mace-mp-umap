import ase
import numpy as np
import pandas as pd
from ase.neighborlist import neighbor_list
from tqdm import tqdm


def calculate_local_atom_density(
    atoms: ase.Atoms | ase.Atom, cutoff: float | np.ndarray = 1.8
) -> np.ndarray:
    """
    Calculate local atom number density using ase.neighborlist.neighbor_list
    """
    edge_array = neighbor_list("ij", atoms, cutoff=cutoff, self_interaction=True)
    edge_array = np.stack(edge_array, axis=1)

    neighbour_numbers = np.bincount(edge_array[:, 0])
    return neighbour_numbers


def find_closest_training_points(training_df, test_df):
    structure_groups = test_df.groupby("structure_index")
    training_descriptors = np.vstack(
        training_df["descriptor"]
    )  # num_many_atoms x num_features
    training_norms = np.linalg.norm(training_descriptors, axis=1)
    mp_ids = training_df["mp_id"].values
    unique_mp_ids = np.unique(mp_ids)
    results = []
    print("Finding closest training points")
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
