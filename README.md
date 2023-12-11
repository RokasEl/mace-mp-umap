# Tool for generating chemiscope input file

## Installation

```bash
git clone https://github.com/RokasEl/mace-mp-umap.git
cd mace-mp-umap; pip install .
```

<span style="color:red;">You need to install the latest version of MACE separately!</span>


## Usage

The package comes with a CLI tool. The basic usage is:

```bash
mace_mp_umap_analysis PATH_TO_TEST_SYSTEM \
 PATH_TO_MP_DATA \
 -e Al -e O \
 --filtering exclusive \
 --element-cutoffs 3 \
 --create-plots
```
First file is always your test system, the second file is the relaxed MP data. See link below.

Filtering flag is used as such:
    - `exclusive` will keep only those MP structures which exactly contain *only* those elements supplied via the `-e` flag. In the example above, only structures containing Al and O will be kept.
    - `inclusive` will allows other elements in addition to those supplied via the `-e` flag. In the example above, structures containing Al and O **and other elements** will be kept.

Element cutoffs flag let's you analyse local atom density:
    - If `--element-cutoffs` is not supplied, no analysis is done.
    - If one number is supplied, it is used as the cutoff for all elements. E.g. `--element-cutoffs 3`. The number of neighours can be seen in the generated plots or with chemiscope.
    - If you want to specify different cutoffs for each element, then supply the cutoffs the same number of times as there are `-e` flags. E.g. `--element-cutoffs 3 --element-cutoffs 4`. In this case, the first cutoff will be used for the first element, the second cutoff for the second element, etc.

To create static UMAP and PCA plots use the `--create-plots` flag:
    - `--create-plots` plots will be created.
    - `--no-create-plots` plots will not be created. This is the default behaviour.

You can also use the `--help` flag to see all the options.

```bash
mace_mp_umap_analysis --help
```

## MP data

Download from this link: https://www.dropbox.com/scl/fi/3n9bikjlk4m3o5py7dq5h/universal_train_corr.xyz?rlkey=w58zho1ow70fp9miiw3nh8jp7&dl=0

Data taken from Materials Project which is distributed under a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).
