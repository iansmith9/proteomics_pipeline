import sys
from pathlib import Path
import logging
from datetime import datetime
from typing import *
import functools

import pandas as pd
import numpy as np
from tqdm import tqdm
import click
import ppx
import shutil
import mokapot
from importlib import resources, import_module



from .msconvert import *
from .comet_db_search import *
from .lfq_biosaur2 import *
from .tmt_quant_tool import *
from .fdr_wrangle_quant_merge import *

# Configure basic logging settings for the application
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(stream=sys.stdout)],
)

logger = logging.getLogger(__name__)

def arg_logger(f):
    @functools.wraps(f)
    def func(*args, **kwargs):
        logger.info("Start: %s -> args: %s, kwargs: %s" % (f.__name__, args, kwargs))
        res = f(*args, **kwargs)
        logger.info("Finish: %s" % f.__name__)
        return res

    return func

# Replace configure_logger with this. 
def configure_logger(fw: Optional[Callable]=None):
    def decorator(f: Callable):
        @functools.wraps(fw if fw else f)
        def func(*args, **kwargs):
            logger = logging.getLogger()
            context = click.get_current_context()
            subcommand = context.info_name
            path = context.params["path"]
            # Add file handler that logs into the experiment directory.
            file_handler = logging.FileHandler(
                "{path}/{:%Y%m%d-%H%M%S}__{cmd}.log".format(
                    datetime.now(), cmd=subcommand, path=path
                ),
                mode="w",
            )
            # Get default stream handler from root logger.
            stream_handler = [
                h
                for h in logging.getLogger().handlers
                if isinstance(h, logging.StreamHandler)
            ]
            if len(stream_handler) > 0:
                stream_handler = stream_handler[0]
                file_handler.setFormatter(stream_handler.formatter)
            logger.addHandler(file_handler)
            return f(*args, **kwargs)
        return func

    return decorator
# def arg_logger(f):
#     def func(*args, **kwargs):
#         logger.info("Start: %s -> args: %s, kwargs: %s" % (f.__name__, args, kwargs))
#         res = f(*args, **kwargs)
#         logger.info("Finish: %s" % f.__name__)
#         return res

#     return func


# def configure_logger(f):
#     def func(*args, **kwargs):
#         logger = logging.getLogger()
#         context = click.get_current_context()
#         subcommand = context.info_name
#         path = context.params["path"]
        
#         # Add file handler that logs into the experiment directory
#         file_handler = logging.FileHandler(
#             "{path}/{:%Y%m%d-%H%M%S}__{cmd}.log".format(
#                 datetime.now(), cmd=subcommand, path=path
#             ),
#             mode="w",
#         )
        
#         # Get default stream handler from root logger
#         stream_handler = [
#             h
#             for h in logging.getLogger().handlers
#             if isinstance(h, logging.StreamHandler)
#         ]
#         if len(stream_handler) > 0:
#             stream_handler = stream_handler[0]
#             file_handler.setFormatter(stream_handler.formatter)
#         logger.addHandler(file_handler)
#         return f(*args, **kwargs)

#     return func


@click.group()
def main():
    """ """
    pass





@main.command("pride_repo_file_display", help="List relevant files from pride repo.")
@click.option("--path", help="Path to project directory.")
@click.option("--pride_repo", help="Pride extension eg PXD000000.")
@click.option("--file_extension", default=None, help="File extension of desired file eg *.raw")
@configure_logger
def _pride_repo_file_display(*args: Any, **kwargs: Any) -> None:
    """CLI wrapper for pride_repo_file_display function."""
    pride_repo_file_display(*args, **kwargs)


@arg_logger
def pride_repo_file_display(
    path: Union[Path, str],
    pride_repo: str = "",
    file_extension: str = None,
) -> List[str]:
    """List files available in a PRIDE repository.

    Args:
        path: Path to project directory
        pride_repo: PRIDE repository identifier (e.g. PXD000000)
        file_extension: Optional file extension filter (e.g. *.raw)

    Returns:
        List of file paths from the PRIDE repository matching the criteria
    """
    path = Path(path)
    proj = ppx.find_project(pride_repo, local=path)
    if not file_extension:
        return proj.remote_files()
    else:
        return proj.remote_files(file_extension)
        
@main.command("internal_repo_file_display", help="List files from internal repository.")
@click.option("--path", help="Path to project directory.")
@click.option("--interal_repo_path", default=None, help="Internal repository directory path.")
@click.option("--file_extension", default=None, help="File extension filter (e.g. *.raw)")
@configure_logger
def _internal_repo_file_display(*args: Any, **kwargs: Any) -> None:
    """CLI wrapper for internal_repo_file_display function."""
    internal_repo_file_display(*args, **kwargs)

@arg_logger
def internal_repo_file_display(
    path: Union[Path, str],
    interal_repo_path: Union[Path, str, None] = None,
    file_extension: str = None,
) -> List[Path]:
    """List files available in an internal repository directory.

    Args:
        path: Path to project directory
        interal_repo_path: Path to internal repository directory
        file_extension: Optional file extension filter (e.g. *.raw)

    Returns:
        List of file paths from the internal repository matching the criteria
    """
    path = Path(path)
    interal_repo_path = Path(interal_repo_path)
    
    file_list = []
    pattern = file_extension if file_extension else "*"
    for file_path in interal_repo_path.glob(pattern):
        file_list.append(file_path)
    return file_list

    




@main.command("raw_file_extract", help="Download or point to raw files.")
@click.option("--path", help="Path to project directory.")
@click.option("--pride_or_internal", default="pride", help="Source internal or pride.")
@click.option("--interal_repo_path", default=None, help="Internal Repo directory.")
@click.option("--pride_repo", default=None, help="Pride extension eg PXD000000.")
@click.option("--aws_copy", is_flag=True, default=False, help="AWS s3 copy or local copy")
@click.option("--file_extension", default=None, help="pride only, File extension of desired file eg *.raw")
@click.option("--file_list", default=None, help="Custom list of files when all_files is False.")
@click.option("--out", default=None, help="Path to output folder.")
@configure_logger
def _raw_file_extract(*args: Any, **kwargs: Any) -> None:
    """CLI wrapper for raw_file_extract function."""
    raw_file_extract(*args, **kwargs)


@arg_logger
def raw_file_extract(
    path: Union[Path, str],
    pride_or_internal: str = "pride",
    pride_repo: str = None,
    interal_repo_path: str = None,
    aws_copy: bool = False,
    file_extension: str = None,
    file_list: List[str] = None,
    out: Union[Path, str, None] = None,
) -> None:
    """Extract raw files from either PRIDE repository or internal storage.

    Downloads files from PRIDE repository or copies files from internal storage location.
    Supports filtering by file extension and specific file lists.

    Args:
        path: Path to project directory
        pride_or_internal: Source of files - either "pride" or "internal"
        pride_repo: PRIDE repository identifier (e.g. PXD000000)
        interal_repo_path: Path to internal repository directory
        aws_copy: Whether to use AWS S3 copy instead of local copy
        file_extension: Optional file extension filter (e.g. *.raw)
        file_list: Optional list of specific files to extract
        out: Optional custom output directory path

    Raises:
        Exception: If pride_repo is not provided when pride_or_internal is "pride"
    """
    path = Path(path)
    if not out:
        out = Path(path) / "raw"
    else:
        out = Path(out) / "raw"
    out.mkdir(exist_ok=True)

    if pride_or_internal == "pride":
        if not pride_repo:
            raise Exception("Must have PRIDE repo name PDX0000000")
        else:
            proj = ppx.find_project(pride_repo, local=out)
            if not file_list:
                if file_extension:
                    files = proj.remote_files(file_extension)
                    proj.download(files)
                else:
                    proj.download()
            else:
                proj.download(file_list)
    else:
        interal_repo_path = Path(interal_repo_path)
        if not file_list:
            if file_extension:
                for ms_file in interal_repo_path.glob(file_extension):
                    ms_file_repo_location = interal_repo_path / ms_file
                    if aws_copy:
                        transfer_cmd = f'aws s3 cp {ms_file_repo_location} {out}'
                        os.system(transfer_cmd)
                    else:
                        shutil.copy(ms_file_repo_location, out)
            else:
                for ms_file in interal_repo_path.glob("*"):
                    ms_file_repo_location = interal_repo_path / ms_file
                    if aws_copy:
                        transfer_cmd = f'aws s3 cp {ms_file_repo_location} {out}'
                        os.system(transfer_cmd)
                    else:
                        shutil.copy(ms_file_repo_location, out)
        else:
            for file_name in file_list:
                ms_file_location = interal_repo_path / file_name
                if aws_copy:
                    transfer_cmd = f'aws s3 cp {ms_file_location} {out}'
                    os.system(transfer_cmd)
                else:
                    shutil.copy(ms_file_location, out)


@main.command("raw_to_mzml", help="Convert raw to mzml.")
@click.option("--path", help="Path to project directory.")
@click.option(
    "--path_msconvert",
    default=None,
    help="Path to msconvert if local, not linux."
)
@click.option(
    "--linux",
    is_flag=True,
    default=False,
    help="Linux or local msconvert."
)
@configure_logger
def _raw_to_mzml(*args: Any, **kwargs: Any) -> None:
    """CLI wrapper for raw_to_mzml function."""
    raw_to_mzml(*args, **kwargs)


@arg_logger
def raw_to_mzml(
    path: Union[Path, str],
    path_msconvert: Union[Path, str, None] = None,
    linux: bool = False,
) -> None:
    """Convert RAW mass spec files to mzML format.

    Args:
        path: Path to directory containing RAW files
        path_msconvert: Path to msconvert executable for local conversion
        linux: Whether to use Linux msconvert (True) or local msconvert (False)

    Raises:
        Exception: If local conversion is selected but no msconvert path provided
    """
    if linux:
        msconvert_run_linux(path)
    else:
        if not path_msconvert:
            raise Exception("Require local path for msconvert or incorrect path provided")
        else:
            msconvert_run_local(path, path_msconvert)

    
@main.command("db_search", help="Set up database search using Comet.")
@click.option("--path", help="Path to project directory.")
@click.option("--path_comet", help="Path to Comet executable.")
@click.option(
    "--param_default",
    default=None,
    help="Standard parameter presets: human_lfq_tryptic, human_tmt18_tryptic_itms2, "
    "human_tmt18_tryptic_otms2, human_tmt11_tryptic_itms2, human_tmt11_tryptic_otms2"
)
@click.option(
    "--params",
    default=None,
    help="Custom Comet search params file.",
)
@configure_logger
def _db_search(*args: Any, **kwargs: Any) -> None:
    """CLI wrapper for db_search function."""
    db_search(*args, **kwargs)


@arg_logger
def db_search(
    path: Union[str, Path],
    path_comet: Union[str, Path],
    params: Union[str, Path, None] = None,
    param_default: str = None,
) -> None:
    """Run Comet database search on mzML files.

    Args:
        path: Path to directory containing mzML files
        path_comet: Path to Comet executable
        params: Path to custom Comet parameters file
        param_default: Name of standard parameter preset to use

    Raises:
        Exception: If neither params nor valid param_default is provided
    """
    path = Path(path)
    # package_path = Path(__file__).parent
    import_module("proteomics_pipeline")

    if param_default:
        params_map = {
            "lfq_tryptic": "human_lfq_tryptic.params",
            "tmt18_tryptic_itms2": "human_tmt18_tryptic_itms2.params",
            "tmt11_tryptic_otms2": "human_tmt11_tryptic_otms2.params",
            "tmt18_tryptic_itms2": "human_tmt18_tryptic_itms2.params",
            "tmt11_tryptic_otms2": "human_tmt11_tryptic_otms2.params"
        }

        if param_default not in params_map:
            raise Exception("Invalid parameter preset name provided")

        params = package_path / "comet_params_defaults" / params_map[param_default]

    elif not params:
        raise Exception("Must provide either params file or valid parameter preset name")

    # Run Comet search on all mzML files in directory
    for ms2_file in path.glob("mzml/*.mzML"):
        run_comet(ms2_file, params, path_comet)

@main.command("detect_ms1_features", help="Use biosaur for ms1 feature detection.")
@click.option("--path", help="Path to project directory.")
@click.option("--out", default=None, help="Path to output files.")
@click.option("--hills", default="5", help="Comma-separated list of hills to run.")
@click.option(
    "--charge",
    default=8,
    type=int,
    help="Maximum charge state to consider"
)
@configure_logger
def _detect_ms1_features(*args: Any, **kwargs: Any) -> None:
    """CLI wrapper for detect_ms1_features function."""
    detect_ms1_features(*args, **kwargs)


@arg_logger
def detect_ms1_features(
    path: Union[Path, str],
    hills: str = "5",
    charge_min: int = 2,
    charge_max: int = 6,
    out: Union[Path, str, None] = None,
) -> None:
    """Run MS1 feature detection using Biosaur2.

    Args:
        path: Path to directory containing mzML files
        hills: Miniumum hill values to process
        charge_min: Minimum charge state to consider
        charge_max: Maximum charge state to consider
        out: Output directory path. If None, uses "{path}/ms1_features"

    The function processes each mzML file in the path/mzml directory
    using the specified hill values and charge state range.
    """
    path = Path(path)
    if not out:
        out = Path(path) / "ms1_features"
    else:
        out = Path(out) / "ms1_features"

    for hill_value in map(int, hills.split(",")):
        for mzml_file in path.glob("mzml/*.mzML"):
            run_biosaur2(
                mzml_file,
                out,
                hill_value,
                charge_min,
                charge_max
            )

            
@main.command("tmt_quant_function", help="Quantify TMT reporter ions from MS3 scans.")
@click.option("--path", help="Path to project directory containing mzML files.")
@click.option(
    "--Da_tol",
    default=0.003,
    type=float,
    help="Mass tolerance in Daltons for matching reporter ions."
)
@click.option("--out", default=None, help="Path to output directory. Defaults to {path}/ms3_features.")
@configure_logger
def _tmt_quant_function(*args: Any, **kwargs: Any) -> None:
    """CLI wrapper for tmt_quant_function."""
    tmt_quant_function(*args, **kwargs)


@arg_logger
def tmt_quant_function(
    path: Union[Path, str],
    Da_tol: float = 0.003,
    out: Union[Path, str, None] = None,
) -> None:
    """Extract and quantify TMT reporter ions from MS3 scans.

    Args:
        path: Path to directory containing mzML files
        Da_tol: Mass tolerance in Daltons for matching reporter ions
        out: Output directory path. If None, uses "{path}/ms3_features"

    The function processes each mzML file in the path/mzml directory,
    extracting TMT reporter ion intensities from MS3 scans within
    the specified mass tolerance.
    """
    path = Path(path)
    if not out:
        out = Path(path) / "ms3_features"
    else:
        out = Path(out) / "ms3_features"

    out.mkdir(exist_ok=True, parents=True)
    print(f"Writing output to: {out}")
    for mzml_file in path.glob("mzml/*.mzML"):
        print(f"Processing file: {mzml_file}")
        run_tmt_quant(mzml_file, Da_tol, out)
        
        
@main.command("train_fdr_model", help="Generate joint pin file for mokapot brew.")
@click.option("--path", help="Path to project.")
@click.option("--fasta", help="Path to fasta file.")
@click.option("--enzyme_regex", default="[KR](?!P)", help="Enzyme regular expression pattern.")
@click.option("--lfq_tmt", default="tmt", type=str, help="Choice of quant type: 'lfq' or 'tmt'")
@click.option("--faims", is_flag=True, default=False, help="Whether FAIMS was used.")
@click.option("--out_loc", help="Path to folder to store results.")
@click.option("--file_list", default=None, help="Custom list of files when all_files is False.")
@click.option("--psm_fdr", default=0.01, type=float, help="PSM-level FDR threshold (0 to 1)")
@click.option("--peptide_fdr", default=1.0, type=float, help="Peptide-level FDR threshold (0 to 1)")
@click.option("--protein_fdr", default=0.01, type=float, help="Protein-level FDR threshold (0 to 1)")
@click.option("--threads", type=int, default=3, help="Number of threads for parallel processing")
@click.option("--jobs", type=int, default=3, help="Number of concurrent jobs")
@configure_logger
def _train_fdr_model(*args: Any, **kwargs: Any) -> None:
    """CLI wrapper for train_fdr_model."""
    train_fdr_model(*args, **kwargs)


@arg_logger
def train_fdr_model(
    path: Union[Path, str],
    fasta: Union[Path, str],
    enzyme_regex: str = "[KR](?!P)",
    lfq_tmt: str = "tmt",
    faims: bool = False,
    out_loc: Union[Path, str, None] = None,
    file_list: Optional[List[str]] = None,
    psm_fdr: float = 0.01,
    peptide_fdr: float = 1.0,
    protein_fdr: float = 0.01,
    threads: int = 3,
    jobs: int = 3,    
) -> mokapot.dataset.PsmDataset:
    """Train FDR model using mokapot and process PSM data.

    Args:
        path: Path to project directory
        fasta: Path to FASTA protein sequence database
        enzyme_regex: Regular expression pattern for enzyme cleavage
        lfq_tmt: Quantification type ('lfq' or 'tmt')
        faims: Whether FAIMS was used in the experiment
        out_loc: Output directory path (defaults to "{path}/mokapot_results")
        file_list: Optional list of specific files to process
        psm_fdr: PSM-level FDR threshold
        peptide_fdr: Peptide-level FDR threshold
        protein_fdr: Protein-level FDR threshold
        threads: Number of threads for parallel processing
        jobs: Number of concurrent jobs

    Returns:
        PsmDataset containing processed results

    Raises:
        Exception: If lfq_tmt is not 'tmt' or 'lfq'
    """
    path = Path(path)
    if not out_loc:
        out_loc = Path(path) / "mokapot_results"
    else:
        out_loc = Path(out_loc)
    out_loc.mkdir(exist_ok=True)
    
    proteins = mokapot.read_fasta(
        fasta,
        enzyme=enzyme_regex,
        decoy_prefix="decoy_",
        missed_cleavages=2,
        min_length=5,
        max_length=50
    )

    if not file_list:
        psm_list = []
        list_quant_files = []
        output_prefix = []
        for pin_fh in (Path(path) / "mzml").glob("*.pin"):
            pin_file = mokapot.read_pin(pin_fh)
            psm_list.append(pin_file)
            output_prefix.append(f"{pin_fh.stem}")
            
            if lfq_tmt == "tmt":
                handle = (Path(path) / "ms3_features") / f"{pin_fh.stem}_ms3_quant.csv"
                list_quant_files.append(handle)
            elif lfq_tmt == "lfq":
                handle = (Path(path) / "ms1_features") / f"{pin_fh.stem}.features.tsv"
                list_quant_files.append(handle)
            else:
                raise Exception("Must designate 'tmt' or 'lfq' with lfq_tmt parameter.")

        [p.add_proteins(proteins) for p in psm_list]
        
        results, models = mokapot.brew(
            psm_list,
            mokapot.model.PercolatorModel(
                rng=np.random.default_rng(seed=42),
                override=True,
            ),
            max_workers=threads,
            rng=np.random.default_rng(seed=42),
        )
        
        dump_models_and_results(
            psm_list,
            models,
            results,
            output_prefix,
            list_quant_files,
            path,
            out_loc,
            psm_fdr,
            peptide_fdr,
            protein_fdr,
            lfq_tmt,
            faims,
        )

        

if __name__ == "__main__":
    main()