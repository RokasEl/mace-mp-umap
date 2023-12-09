# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: mace
#     language: python
#     name: mace
# ---

# +
import glob

import umap
from ase.io import read
from mace.calculators import MACECalculator
from tqdm.auto import tqdm

calculator = MACECalculator(
    model_paths=["../2023-12-03-mace-128-L1_epoch-199.model"],
    device="cuda",
    default_dtype="float64",
)

n_umap_neighbors = 100

reducer = umap.UMAP(
    n_neighbors=n_umap_neighbors,
    n_components=2,
    metric="manhattan",
    random_state=77,
    low_memory=False,
    n_jobs=16,
)
# -

files = sorted(glob.glob("../dataset/mptrj-gga-ggapu/*.extxyz"))

import numpy as np

# +
import pandas as pd

index_levels = ["formula", "mp_id", "task_id", "calc_id", "ionic_step", "site"]
mptrj = pd.DataFrame(
    index=pd.MultiIndex.from_tuples([], names=index_levels), columns=np.arange(256)
)

data = {}

count = 0

for count, file in tqdm(enumerate(files)):
    # if count > 200:
    #     break

    traj = read(file, index=":")

    for atoms in traj[0:-1:10]:
        descriptors = calculator.get_descriptors(atoms)

        index = (
            atoms.get_chemical_formula(),
            atoms.info["mp_id"],
            atoms.info["task_id"],
            atoms.info["calc_id"],
            atoms.info["ionic_step"],
        )
        # data[index] = np.mean(descriptors, axis=0)

        unique_elements, element_indices = np.unique(atoms.symbols, return_inverse=True)

        # Use np.add.at to accumulate vectors for each element
        element_sums = np.zeros((len(unique_elements), descriptors.shape[1]))
        np.add.at(element_sums, element_indices, descriptors)

        # Divide the sums by the count to get the element-wise average
        element_counts = np.bincount(element_indices, minlength=len(unique_elements))
        elementwise_avg = element_sums / element_counts[:, None]

        for i, elem in enumerate(unique_elements):
            mptrj.loc[index + (elem,), :] = elementwise_avg[
                i
            ]  # np.mean(descriptors, axis=0)

# -

mptrj.to_csv("mptrj-mace-descriptors.csv")
mptrj.dropna(inplace=True)
reducer.fit(mptrj)

# +
umap_pts = reducer.transform(mptrj)

umap_df = pd.DataFrame(umap_pts, index=mptrj.index)

# +
from ase import Atom

umap_df["number"] = umap_df.index.get_level_values("site").map(lambda e: Atom(e).number)

umap_df.to_csv("umap_pts.csv")
# -

umap_df

import matplotlib as mpl

# +
from matplotlib import pyplot as plt
from matplotlib.patheffects import withStroke
from pymatgen.core import Element

with plt.style.context("default"):
    fig, axes = plt.subplot_mosaic(
        "ab",
        figsize=(7.5, 3),
        layout="constrained",
    )

    seed = 80124
    size = 1
    fontsize = 8
    nsamples = 15

    i = "a"

    vmin = umap_df["number"].min()
    vmax = umap_df["number"].max()

    sc = axes[i].scatter(
        umap_df[0],
        umap_df[1],
        c=umap_df["number"],
        # zorder=np.argsort(c),
        cmap="terrain",
        norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax),
        s=size,
    )

    # Randomly choose 10 indices

    np.random.seed(seed)
    random_indices = np.random.choice(len(umap_df), size=nsamples, replace=False)

    # Annotate 10 randomly chosen points with their atomic numbers
    for idx in random_indices:
        axes[i].annotate(
            Atom(umap_df["number"].iloc[idx]).symbol,
            (umap_df[0].iloc[idx], umap_df[1].iloc[idx]),
            size=fontsize,
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
            color="white",  # Text color
            path_effects=[withStroke(linewidth=2, foreground="black")],  # White contour
        )

    cbar = axes[i].figure.colorbar(
        sc,
        boundaries=np.arange(vmin, vmax + 1) + 0.5,
        ticks=np.arange(vmin, vmax + 1, 10),
        label="atomic numbers",
        aspect=25,
    )
    xlo, xhi = axes[i].get_xlim()
    ylo, yhi = axes[i].get_ylim()
    lim = (min(xlo, ylo), max(xhi, yhi))
    axes[i].set(
        # aspect='equal',
        xlabel="UMAP 1",
        ylabel="UMAP 2",
        # xlim=lim, ylim=lim
    )

    i = "b"

    umap_df["group"] = list(
        map(lambda a: Element(Atom(a).symbol).group, umap_df["number"])
    )

    vmin = 0  # umap_df['group'].min()
    vmax = 18  # umap_df['group'].max()

    sc = axes[i].scatter(
        umap_df[0],
        umap_df[1],
        c=umap_df["group"],
        # zorder=np.argsort(c),
        cmap="rainbow",
        norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax),
        s=size,
    )

    # Randomly choose 10 indices
    np.random.seed(seed)
    random_indices = np.random.choice(len(umap_df), size=nsamples, replace=False)

    # Annotate 10 randomly chosen points with their atomic numbers
    for idx in random_indices:
        axes[i].annotate(
            Atom(umap_df["number"].iloc[idx]).symbol,
            (umap_df[0].iloc[idx], umap_df[1].iloc[idx]),
            size=fontsize,
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
            color="white",  # Text color
            path_effects=[withStroke(linewidth=2, foreground="black")],  # White contour
        )

    cbar = axes[i].figure.colorbar(
        sc,
        boundaries=np.arange(vmin, vmax + 1) + 0.5,
        ticks=np.arange(vmin, vmax + 1),
        label="group",
        aspect=25,
    )
    xlo, xhi = axes[i].get_xlim()
    ylo, yhi = axes[i].get_ylim()
    lim = (min(xlo, ylo), max(xhi, yhi))
    axes[i].set(
        # aspect='equal',
        xlabel="UMAP 1",
        ylabel="UMAP 2",
        # xlim=lim, ylim=lim
    )

    plt.savefig("mptrj-umap.pdf")
    plt.savefig("mptrj-umap.png")
    plt.show()

# -
