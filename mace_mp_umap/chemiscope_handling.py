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


def get_reduced_embeddings(reducers, descriptors):
    pca = reducers[1][1].transform(descriptors)
    umap_emb = reducers[1][0].transform(descriptors)
    return pca, umap_emb


def get_train_test_split(train_atoms, test_atoms):
    split = ["train"] * sum([len(x) for x in train_atoms])
    split += ["test"] * sum([len(x) for x in test_atoms])
    return split


def get_atomic_properties(atoms):
    embeddings = np.vstack([a.arrays["mace_descriptors"] for a in atoms])
    symbols = np.hstack([a.symbols for a in atoms])
    groups, periods = np.vectorize(get_group_and_period)(symbols)
    if "num_neighbours" in atoms[0].arrays:
        neighbours = np.hstack([a.arrays["num_neighbours"] for a in atoms])
    else:
        neighbours = None
    return embeddings, symbols, groups, periods, neighbours


def create_property(name, values, description, target="atom"):
    return {f"{name}": {"target": target, "values": values, "description": description}}


def write_chemiscope_input(train_atoms, test_atoms, reducers, system_name):
    all_atoms = train_atoms + test_atoms

    (descriptors, symbols, groups, periods, neighbours) = get_atomic_properties(
        all_atoms
    )
    pca, umap_emb = get_reduced_embeddings(reducers, descriptors)
    train_test_split = get_train_test_split(train_atoms, test_atoms)
    properties = [
        create_property("0_UMAP", umap_emb, "UMAP Embeddings"),
        create_property("PCA", pca, "PCA Embeddings"),
        create_property("TrainTest", train_test_split, "Train/Test split"),
        create_property("element", symbols, "Atomic element"),
        create_property("group", groups, "Group"),
        create_property("period", periods, "Period"),
    ]
    if neighbours is not None:
        properties.append(
            create_property("num_neighbours", neighbours, "Number of neighbours")
        )
    properties = {k: v for d in properties for k, v in d.items()}
    # define better default settings for the viewer
    settings = {
        "map": {"color": {"property": "TrainTest"}, "palette": "brg"},
        "structure": [{"atomLabels": True}],
    }
    chemiscope.write_input(
        path=f"{system_name}_chemiscope_input.json",
        frames=all_atoms,
        properties=properties,
        settings=settings,
        # This is required to display properties with `target: "atom"`
        environments=chemiscope.all_atomic_environments(all_atoms),
    )
