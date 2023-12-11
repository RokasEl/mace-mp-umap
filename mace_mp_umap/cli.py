import pathlib
import typing as t
from collections import defaultdict

import typer
from mace.calculators import mace_mp
from typing_extensions import Annotated

app = typer.Typer()
import warnings
from enum import Enum

import torch

from .analysis import find_closest_training_points
from .chemiscope_handling import write_chemiscope_input
from .data_manipulations import get_cleaned_dataframe
from .dim_reduction import (
    apply_dimensionality_reduction,
    fit_dimensionality_reduction,
)
from .plotting import plot_dimensionality_reduction
from .utils import get_layer_specific_feature_slices

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class FilterType(str, Enum):
    exclusive = "exclusive"
    inclusive = "inclusive"
    none = "none"


@app.command()
def produce_mace_chemiscope_input(
    data_path: str = typer.Argument(
        default=None,
        help="Path to XYZ file containing your system",
    ),
    mp_data_path: str = typer.Argument(default=None, help="Path to MP data"),
    filtering: FilterType = typer.Option(
        default=FilterType.none,
        case_sensitive=False,
        help="Whether to filter out structures that contain elements not in the subset or to include them.",
    ),
    element_subset: Annotated[
        t.List[str],
        typer.Option(
            "--add-element", "-e", help="List of elements to include in the subset."
        ),
    ] = [],
    create_plots: bool = typer.Option(
        default=False, help="Whether to create static UMAP and PCA plots."
    ),
):
    if DEVICE != "cuda":
        warnings.warn("CUDA not available, using CPU. Might be slow.")

    if filtering == FilterType.none:
        raise ValueError(
            "You must specify filtering type (either `--filtering exclusive` or `--filtering inclusive`).\n"
            "Exclusive mode means those and only those structures are kept that contail all elements supplied via `-e` flags.\n"
            "Inclusive mode means that other elements are allowed in addition to those supplied via `-e` flags.\n"
            "Most applications should use `--filtering inclusive`. However, for elemental compounds or molecular compounds like water `exclusive` mode is more appropriate."
        )

    # Load model
    calc = mace_mp(
        model="medium",
        device=DEVICE,
        default_dtype="float64",
    )
    print(
        f"Using the MACE cutoff ({calc.r_max} A) for neighbour analysis for all elements."
    )
    cutoff_dict = defaultdict(lambda: calc.r_max)

    # Load MP data
    train_atoms, training_data_df = get_cleaned_dataframe(
        mp_data_path, calc, element_subset, cutoff_dict, filtering_type=filtering
    )
    # Load test data
    test_atoms, test_data_df = get_cleaned_dataframe(
        data_path, calc, element_subset, cutoff_dict, filtering_type="none"
    )
    if len(test_data_df) == 0 or len(training_data_df) == 0:
        raise ValueError(
            f"No structures found in {data_path} or {mp_data_path}. Check your filtering settings."
        )
    system_name = pathlib.Path(data_path).stem
    print(f"Will use {system_name} for naming output files.")
    # Fit dimensionality reduction
    slices = get_layer_specific_feature_slices(calc)
    reducers = []
    for i, sli in enumerate(slices):
        tag = f"layer_{i}"
        umap_reducer, pca_reducer = fit_dimensionality_reduction(
            training_data_df, tag, sli
        )
        if create_plots:
            apply_dimensionality_reduction(
                test_data_df, tag, sli, umap_reducer, pca_reducer
            )
        reducers.append((umap_reducer, pca_reducer))
    # Produce plots if requested
    if create_plots:
        figure = plot_dimensionality_reduction(
            training_data_df, test_data_df, len(slices)
        )
        figure.savefig(f"{system_name}_dimensionality_reduction.pdf")
    # Find closest training points
    results_df = find_closest_training_points(training_data_df, test_data_df)
    results_df.to_csv(f"{system_name}_closest_training_points.csv", index=False)
    # Produce chemiscope input file
    write_chemiscope_input(train_atoms, test_atoms, reducers, system_name)


if __name__ == "__main__":
    app()
