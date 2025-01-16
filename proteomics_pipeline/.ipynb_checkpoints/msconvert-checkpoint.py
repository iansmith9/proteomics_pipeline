import os
from typing import *
from pathlib import Path
import subprocess

import tqdm
import pandas as pd
from pyteomics import mzml
import numpy as np

    
def msconvert_run_linux(
    path: Union[Path, str],
) -> None:
    """ """
    # make sure locations for docker are accurate
    # run msconvert on each file in a loop
    msconvert_cmd = (
        "docker run --rm -e WINEDEBUG=-all "
        + "-v {path}:/data "
        + "chambm/pwiz-skyline-i-agree-to-the-vendor-licenses wine msconvert "
        + '/data/raw/{file} -o /data/mzml/ --zlib' 
        + '--filter "peakPicking true 1- '
        + '"'
        # + 'File:"""^<SourcePath^>""", NativeID:"""^<Id^>""""'
    )
    for file in Path(path).glob("raw/*.raw"):
        subprocess.call(
            msconvert_cmd.format(file=file.name, path=Path(path).absolute()),
            shell=True,
        )    

        
def msconvert_run_local(
    path: Union[Path, str],
    path_msconvert: Union[Path, str],
) -> None:
    """ """
    # make sure locations for docker are accurate
    # run msconvert on each file in a loop
    # path_msconvert2 = path_msconvert
    # path_msconvert = Path(path_msconvert)
    
    # path = Path(path) / raw
    # out = Path(path) / mzml
    
    msconvert_cmd = (
        '"{path_msconvert}" '
        + '{path}/raw/{file} -o {path}/mzml/ --zlib ' 
        + '--filter "peakPicking vendor msLevel=1-" --filter "titleMaker <RunId>.<ScanNumber>.<ScanNumber>.<ChargeState> '
        + '"'
        # + 'File:"""^<SourcePath^>""", NativeID:"""^<Id^>""""'
    )
    for file in Path(path).glob("raw/*.raw"):
        subprocess.call(
            msconvert_cmd.format(file=file.name, path=Path(path).absolute(), path_msconvert=Path(path_msconvert).absolute()),
            shell=True,
        )