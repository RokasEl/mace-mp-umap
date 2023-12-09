# Tool for generating chemiscope input file

## Installation

Git clone the repo and `pip install .`. This will install all the required packages for the analysis. You need to install the latest version of MACE separately.

## Usage

```bash
mace_mp_umap_analysis --help
```

Will show the relument options.

First file is always your test system, the second file is the relaxed MP data. I'll attach a link to that separately.

Filtering flag is used as such:
    - `exclusive` will keep only those MP structures which exactly contain *only* those elements supplied via the `-e` flag.
    - `inclusive` will allows other elements in addition to those supplied via the `-e` flag.

Add multiple elements via multiple `-e` flags. E.g.
```bash
mace_mp_umap_analysis test_system.xyz universal_train_corr.xyz -e Li -e O -filtering exclusive
```

## MP data

Download from this link: https://we.tl/t-MP4Bi4T6Ky

Data taken from Materials Project which is distributed under a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).
