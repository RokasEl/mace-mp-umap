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


def calculate_descriptors(atoms: t.List[ase.Atoms | ase.Atom], calc, cutoffs) -> None:
    for mol in tqdm(atoms):
        descriptors = calc.get_descriptors(mol, invariants_only=True)
        mol.arrays["mace_descriptors"] = descriptors
        cut = np.array([cutoffs[x] for x in mol.symbols])
        num_neighbours = calculate_local_atom_density(mol, cut)
        mol.arrays["num_neighbours"] = num_neighbours


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
        )  # atoms must *at least* contain elements in the subse
