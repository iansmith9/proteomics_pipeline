from typing import *
import subprocess
from pathlib import Path

def run_comet(
    ms2_fh: Path,
    param: Path,
    path_comet: Path,
) -> None:
    """Run Comet database search on an MS2 file.

    Args:
        ms2_fh: Path to input MS2 file
        param: Path to Comet parameter file
        path_comet: Path to Comet executable

    Returns:
        None. Writes search results to same directory as input file.
    """
    subprocess.call(
        [
            f"{path_comet}",
            f"-P{param}",
            ms2_fh,
        ]
    )
    