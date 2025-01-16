from typing import *
import subprocess
from pathlib import Path

def run_biosaur2(
    mzml_fh: Path,
    out: Path,
    hill: int,
    charge_min: int,
    charge_max: int,
) -> None:
    """ " """
    out_fh = (
        out / f"{mzml_fh.stem}.features.tsv"
    )
    out_fh.parent.mkdir(exist_ok=True, parents=True)
    subprocess.call(
        [
            "biosaur2",
            "-cmin",
            str(charge_min),
            "-cmax",
            str(charge_max),
            "-minlh",
            str(hill),
            mzml_fh,
            "-o",
            out_fh,
        ],
    )