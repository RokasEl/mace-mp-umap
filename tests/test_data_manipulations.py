import ase
import pytest

from mace_mp_umap.data_manipulations import filter_atoms


@pytest.mark.parametrize(
    "atoms, element_subset, expected",
    [
        (ase.Atoms("CHAlOPMnU"), ["C", "Al"], True),
        (ase.Atoms("CHAlOPMnU"), ["C", "Zn"], False),
        (ase.Atoms("SiO3AlHP"), ["Si", "O", "Al"], True),
    ],
)
def test_filter_atoms_inclusive(atoms, element_subset, expected):
    filtering_type = "inclusive"
    out = filter_atoms(atoms, element_subset, filtering_type)
    assert out == expected


@pytest.mark.parametrize(
    "atoms, element_subset, expected",
    [
        (ase.Atoms("CHAlOPMnU"), ["C", "Al"], False),
        (ase.Atoms("CHAlOPMnU"), ["C", "Zn"], False),
        (ase.Atoms("SiO3Al"), ["Si", "O", "Al"], True),
        (ase.Atoms("SiO3AlHP"), ["Si", "O", "Al"], False),
        (
            ase.Atoms("SiO2"),
            [
                "Si",
                "O",
                "Al",
            ],
            False,
        ),
        (ase.Atoms("SiO2"), ["Si", "O"], True),
        (ase.Atoms("SiO2"), ["Si"], False),
        (ase.Atoms("O2"), ["Al", "O"], False),
    ],
)
def test_filter_atoms_exclusive(atoms, element_subset, expected):
    filtering_type = "exclusive"
    out = filter_atoms(atoms, element_subset, filtering_type)
    assert out == expected


@pytest.mark.parametrize(
    "atoms, element_subset, expected",
    [
        (ase.Atoms("CHAlOPMnU"), ["C", "Al"], False),
        (ase.Atoms("CHAlOPMnU"), ["C", "Zn"], False),
        (ase.Atoms("SiO3Al"), ["Si", "O", "Al"], True),
        (ase.Atoms("SiO3AlHP"), ["Si", "O", "Al"], False),
        (
            ase.Atoms("SiO2"),
            [
                "Si",
                "O",
                "Al",
            ],
            True,
        ),
        (ase.Atoms("SiO2"), ["Si", "O"], True),
        (ase.Atoms("SiO2"), ["Si"], False),
        (ase.Atoms("O2"), ["Al", "O"], True),
        (ase.Atoms("C"), ["C", "H", "O"], True),
        (ase.Atoms("O2"), ["C", "H", "O"], True),
        (ase.Atoms("H2"), ["C", "H", "O"], True),
        (ase.Atoms("CH"), ["C", "H", "O"], True),
        (ase.Atoms("CO"), ["C", "H", "O"], True),
        (ase.Atoms("HO"), ["C", "H", "O"], True),
        (ase.Atoms("CHO"), ["C", "H", "O"], True),
    ],
)
def test_filter_atoms_combinations(atoms, element_subset, expected):
    filtering_type = "combinations"
    out = filter_atoms(atoms, element_subset, filtering_type)
    assert out == expected
