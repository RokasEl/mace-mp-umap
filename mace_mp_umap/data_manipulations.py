import typing as t

import ase
import ase.io as aio
import numpy as np
import pandas as pd
from mace.calculators.mace import MACECalculator
from tqdm import tqdm

from .analysis import calculate_local_atom_density


def convert_to_dataframe(atoms: t.List[ase.Atoms | ase.Atom]) -> pd.DataFrame:
    data = []
    for mol_idx, mol in enumerate(tqdm(atoms)):
        descs = mol.arrays["mace_descriptors"]
        num_neighbours = (
            mol.arrays["num_neighbours"] if "num_neighbours" in mol.arrays else None
        )
        elements = mol.symbols
        for idx, d in enumerate(descs):
            data.append(
                {
                    "mp_id": mol.info["mp_id"] if "mp_id" in mol.info else None,
                    "structure_index": mol_idx,
                    "atom_index": idx,
                    "element": elements[idx],
                    "descriptor": d,
                    "num_neighbours": num_neighbours[idx]
                    if num_neighbours is not None
                    else None,
                }
            )
    return pd.DataFrame(data)


def calculate_descriptors(
    atoms: t.List[ase.Atoms | ase.Atom], calc: MACECalculator, cutoffs: None | dict
) -> None:
    print("Calculating descriptors")
    for mol in tqdm(atoms):
        descriptors = calc.get_descriptors(mol, invariants_only=True)
        mol.arrays["mace_descriptors"] = descriptors
        if cutoffs is not None:
            cut = np.array([cutoffs[x] for x in mol.symbols])
            num_neighbours = calculate_local_atom_density(mol, cut)
            mol.arrays["num_neighbours"] = num_neighbours


def get_cleaned_dataframe(
    path: str,
    calc: MACECalculator,
    element_subset: list[str],
    element_cutoffs: dict,
    filtering_type: str,
):
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


def filter_atoms(
    atoms: ase.Atoms, element_subset: list[str], filtering_type: str
) -> bool:
    """
    Filters atoms based on the provided filtering type and element subset.

    Parameters:
    atoms (ase.Atoms): The atoms object to filter.
    element_subset (list): The list of elements to consider during filtering.
    filtering_type (str): The type of filtering to apply. Can be 'none', 'exclusive', or 'inclusive'.
        'none' - No filtering is applied.
        'combinations' - Return true if `atoms` is composed of combinations of elements in the subset, false otherwise. I.e. does not require all of the specified elements to be present.
        'exclusive' - Return true if `atoms` contains *only* elements in the subset, false otherwise.
        'inclusive' - Return true if `atoms` contains all elements in the subset, false otherwise. I.e. allows additional elements.

    Returns:
    bool: True if the atoms pass the filter, False otherwise.
    """
    if filtering_type == "none":
        return True
    elif filtering_type == "combinations":
        atom_symbols = np.unique(atoms.symbols)
        return all(
            [x in element_subset for x in atom_symbols]
        )  # atoms must *only* contain elements in the subset
    elif filtering_type == "exclusive":
        atom_symbols = set([x for x in atoms.symbols])
        return atom_symbols == set(element_subset)
    elif filtering_type == "inclusive":
        atom_symbols = np.unique(atoms.symbols)
        return all(
            [x in atom_symbols for x in element_subset]
        )  # atoms must *at least* contain elements in the subset
    else:
        raise ValueError(
            f"Filtering type {filtering_type} not recognised. Must be one of 'none', 'exclusive', or 'inclusive'."
        )


def remove_non_unique_environments(df, decimals=4):
    descriptors = np.vstack(df["descriptor"])
    _, indices = np.unique(np.round(descriptors, decimals), return_index=True, axis=0)
    return df.iloc[indices].copy().sort_values(by="mp_id")
