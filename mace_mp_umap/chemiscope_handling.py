import chemiscope
import mendeleev
import numpy as np

ELEMENT_DICT = {}


def get_group_and_period(symbol):
    if symbol not in ELEMENT_DICT:
        element = mendeleev.element(symbol)
        group = element.group_id
        group = -1 if group is None else group
        ELEMENT_DICT[symbol] = (group, element.period)
    return ELEMENT_DICT[symbol]


def write_chemiscope_input(train_atoms, test_atoms, reducers):
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
        "PCA": {
            "target": "atom",
            "values": pca,
            "description": "PCA of per-atom representation of the structures",
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
        path="chemiscope_input.json",
        frames=all_atoms,
        properties=properties,
        # This is required to display properties with `target: "atom"`
        environments=chemiscope.all_atomic_environments(all_atoms),
    )
