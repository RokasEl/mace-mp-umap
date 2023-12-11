# Tool for generating chemiscope input file

## Installation

```bash
git clone https://github.com/RokasEl/mace-mp-umap.git
cd mace-mp-umap; pip install .
```

‼️ You need to install the latest version of MACE separately!‼️


## Usage

The package comes with a CLI tool. The basic usage is:

```bash
mace_mp_umap_analysis PATH_TO_TEST_SYSTEM \
 PATH_TO_MP_DATA \
 -e Al -e O \
 --filtering exclusive \
 --create-plots
```
This will generate three files in the same path as your current directory. System name is inferred from the name of your data file:
 1. `YOUR_SYSTEM_NAME_dimensionality_reduction.pdf` The static plots. Only generated with `--create-plots`
 2. `YOUR_SYSTEM_NAME_clostest_training_points.csv` A csv file that ranks the similarity of your test system to each structure in the MP data that passed the filtering criteria.
 3. `YOUR_SYSTEM_NAME_chemiscope_input.json` Input for https://chemiscope.org/ where you can interactively visualise the dimensionality reduction plots and see 3D renders of each atomic environment.

First file is always your test system, the second file is the relaxed MP data. See link below. Make sure your test system contains a few representative structures (multi frame input is supported) but don't use hundreds of very similar structures as it will only slow down the plotting and analysis without giving additional information.

You must supply either an `exclusive` or `inclusive` filtering flag:

   - `exclusive` will keep those MP structures which exactly contain **only** those elements supplied via the `-e` flag. In the example above, only structures containing Al and O will be kept.
   - `inclusive` will allows other elements in addition to those supplied via the `-e` flag. In the example above, structures containing Al and O **and other elements** will be kept.

In most cases, you should use `inclusive` filtering: the UMAP should naturally seggregate additional elements and it might reveal interesting unexpected connection in the data. For example, if your system is catalysis on Indium oxide, you might want to include all structures containing Indium and Oxygen but also allow other elements to see if your catalysis example is similar to other bulk structures.

However, for elemental or small molecular systems (like H2O), you should use `exclusive` filtering. For example, if you're doing Silicon melts including other elements would be too permisive as Si is a very frequent element in the MP data.

To create static UMAP and PCA plots use the `--create-plots` flag:

 - `--create-plots` plots will be created. Colours are assigned based on the local atom density using the MACE model's cutoff to build the neighbourlist.
 - `--no-create-plots` plots will not be created. This is the default behaviour.

You can also use the `--help` flag to see all the options.

```bash
mace_mp_umap_analysis --help
```

## MP data

Download from this link: https://www.dropbox.com/scl/fi/3n9bikjlk4m3o5py7dq5h/universal_train_corr.xyz?rlkey=w58zho1ow70fp9miiw3nh8jp7&dl=0

Data taken from Materials Project which is distributed under a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).
