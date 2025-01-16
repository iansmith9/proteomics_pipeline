from typing import *
import subprocess
from pathlib import Path

def run_comet(
    ms2_fh: Path,
    param: Path,
    path_comet: Path,
) -> None:
    """ """
    
    subprocess.call(
        [
            f"{path_comet}",
            # "C:/proteomics_pipeline/Comet/comet.win64.exe",
            f"-P{param}",
            ms2_fh,
        ]
    )
    
    