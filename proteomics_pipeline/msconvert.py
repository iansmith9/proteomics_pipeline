import os
from typing import *
from pathlib import Path
import subprocess


def msconvert_run_linux(path: Union[Path, str]) -> None:
    """Run MSConvert on RAW files using Docker container.
    
    Converts Thermo RAW files to mzML format using ProteoWizard MSConvert
    running in a Docker container.
 
    Args:
        path: Directory containing 'raw' subdirectory with .raw files.
             Will output to 'mzml' subdirectory.
 
    Returns:
        None. Writes converted mzML files to output directory.
    """
    msconvert_cmd = (
        "sudo docker run --rm -e WINEDEBUG=-all "
        + "-v {path}:/data "
        + "chambm/pwiz-skyline-i-agree-to-the-vendor-licenses wine msconvert "
        + '/data/raw/{file} -o /data/mzml/ --zlib '
        + '--filter "peakPicking true 1- '
        + '"'
        # + 'File:"""^<SourcePath^>""", NativeID:"""^<Id^>""""'
    )
 
    for file in Path(path).glob("raw/*.raw"):
        subprocess.call(
            msconvert_cmd.format(
                file=file.name,
                path=Path(path).resolve()
            ),
            shell=True
        )
 


def msconvert_run_local(
    path: Union[Path, str],
    path_msconvert: Union[Path, str],
) -> None:
    """Run MSConvert on RAW files using local installation.
    
    Converts Thermo RAW files to mzML format using a local installation
    of ProteoWizard MSConvert.

    Args:
        path: Directory containing 'raw' subdirectory with .raw files.
             Will output to 'mzml' subdirectory.
        path_msconvert: Path to MSConvert executable

    Returns:
        None. Writes converted mzML files to output directory.
    """
    msconvert_cmd = (
        '"{path_msconvert}" '
        + '{path}/raw/{file} -o {path}/mzml/ --zlib ' 
        + '--filter "peakPicking vendor msLevel=1-" --filter "titleMaker <RunId>.<ScanNumber>.<ScanNumber>.<ChargeState> '
        + '"'
    )

    for file in Path(path).glob("raw/*.raw"):
        subprocess.call(
            msconvert_cmd.format(
                file=file.name,
                path=Path(path).absolute(),
                path_msconvert=Path(path_msconvert).absolute()
            ),
            shell=True
        )