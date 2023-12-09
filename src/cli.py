import typer
from mace.calculators import mace_mp

app = typer.Typer()
import warnings
from enum import Enum

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class FilterType(str, Enum):
    exclusive = "exclusive"
    inclusive = "inclusive"
    none = "none"


@app.command()
def produce_mace_chemiscope_input():
    if DEVICE != "cuda":
        warnings.warn("CUDA not available, using CPU. Might be slow.")
    # Load model
    calc = mace_mp(
        model="medium",
        device="cuda",
        default_dtype="float64",
    )

    # Setup filtering

    # Load MP data

    # Load test data

    # Fit dimensionality reduction

    # Produce plots if requested

    # Produce chemiscope input file
