from typing import *  # Replace with specific imports
import subprocess
from pathlib import Path

def run_biosaur2(
    mzml_file: Path,
    out: Path,
    hill_value: int,
    charge_min: int,
    charge_max: int,
) -> None:
    """Run Biosaur2 feature detection on an mzML file.

    Args:
        mzml_file: Path to input mzML file
        out: Output directory path
        hill_value: Minimum length of hills (features)
        charge_min: Minimum charge state to consider
        charge_max: Maximum charge state to consider

    Returns:
        None. Writes feature file to output directory.
    """
    out_fh = out / f"{mzml_file.stem}.features.tsv"
    out_fh.parent.mkdir(exist_ok=True, parents=True)
    subprocess.call(
        [
            "biosaur2",
            "-cmin",
            str(charge_min),
            "-cmax",
            str(charge_max),
            "-minlh",
            str(hill_value),
            mzml_file,
            "-o",
            out_fh,
        ],
    )

    