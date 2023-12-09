import io
from zipfile import ZipFile

import ase
import ase.io.extxyz
import pandas as pd
from tqdm import tqdm

__author__ = "Janosh Riebesell"
__date__ = "2023-11-22"


def load_mp_traj_xyz(zip_path="2023-11-22-mp-trj-extxyz-by-yuan.zip"):
    mp_trj_atoms: dict[str, list[ase.Atoms]] = {}

    # extract extXYZ files from zipped directory without unpacking the whole archive
    for name in tqdm((zip_file := ZipFile(zip_path)).namelist()):
        if name.startswith("mptrj-gga-ggapu/mp-"):
            mp_id = name.split("/")[1].split(".")[0]
            assert mp_id.startswith("mp-")
            assert mp_id not in mp_trj_atoms

            with zip_file.open(name) as file:
                # wrap byte stream with TextIOWrapper to use as file
                text_file = io.TextIOWrapper(file, encoding="utf-8")
                atoms_list = list(ase.io.extxyz.read_xyz(text_file, index=slice(None)))
            mp_trj_atoms[mp_id] = atoms_list

    assert len(mp_trj_atoms) == 145_919  # number of unique MP IDs

    df_mp_trj = pd.DataFrame(
        {
            f"{atm.info['task_id']}-{atm.info['calc_id']}-{atm.info['ionic_step']}": {
                "formula": str(atm.symbols)
            }
            | {key: atm.arrays.get(key) for key in ("forces", "magmoms")}
            | atm.info
            for atoms_list in mp_trj_atoms.values()
            for atm in atoms_list
        }
    ).T.convert_dtypes()  # convert object columns to float/int where possible
    df_mp_trj.index.name = "frame_id"
    assert len(df_mp_trj) == 1_580_312  # number of total frames

    return df_mp_trj
