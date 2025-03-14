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
from pyascore import *


from .msconvert import *
from .comet_db_search import *
from .lfq_biosaur2 import *
from .tmt_quant_tool import *
from .fdr_wrangle_quant_merge import *
from .fdr_wrangle_quant_merge_no_protein_group import *
from .pyascore_site_localization import *
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
        pride_repo: PRIDE repository identifier (e.g. PXD000000) (ignored if pride_or_internal is 'internal')
        interal_repo_path: Path to internal repository directory (ignored if pride_or_internal is 'pride')
        aws_copy: Whether to use AWS S3 copy instead of local copy (bool: False (default, internal), True (aws s3 cp))
        file_extension: Optional file extension filter (e.g. *.raw)
        file_list: Optional list of specific files to extract (list of strings; eg ['file1.raw', 'file2.raw'])
        out: Optional custom output directory path (default: {path}/raw)

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
                ms_file_location = str(ms_file_location).replace('s3:/', 's3://')
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
    help="Path to msconvert if local, not linux (str: path to msconvert executable), linux will use docker version and no path is required."
)
@click.option(
    "--linux",
    is_flag=True,
    default=False,
    help="Linux or local msconvert (bool: False (local), True (linux))"
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
        path: Path to project directory
        path_msconvert: Path to msconvert executable for local conversion (str: path to msconvert executable), ignored if linux is True
        linux: Whether to use a linux dockerized msconvert (True) or local msconvert (False) (bool: False (local), True (linux))
 
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
    help="Standard parameter presets: human_lfq_tryptic_itms2, human_lfq_tryptic_otms2, human_tmt18_tryptic_itms2, "
    "human_tmt18_tryptic_otms2, human_tmt11_tryptic_itms2, human_tmt11_tryptic_otms2"
)
@click.option(
    "--params",
    default=None,
    help="Custom Comet search params file. (ignored if param_default is provided)",
)
@click.option("--species", default="human", help="Species of the database: human, mouse, or yeast as defaults (when param_default is provided)")
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
    species: str = "human",
) -> None:
    """Run Comet database search on mzML files.

    Args:
        path: Path to parent directory
        path_comet: Full path to Comet executable
        params: Path to custom Comet parameters file (ignored if param_default is provided)
        param_default: Name of standard parameter preset to use (ignored if params is provided): human_lfq_tryptic_itms2, human_lfq_tryptic_otms2, human_tmt18_tryptic_itms2, human_tmt18_tryptic_otms2, human_tmt11_tryptic_itms2, human_tmt11_tryptic_otms2
        species: Species of the database: human, mouse, or yeast as defaults (when param_default is provided)
    Raises:
        Exception: If neither params nor valid param_default is provided
    """
    path = Path(path)
    path_comet = Path(path_comet)
    # package_path = Path(__file__).parent
    import_module("proteomics_pipeline")
    

    if param_default:
        params_map = {
            "human_lfq_tryptic_otms2": "high_high_lfq_tryptic.params",
            "human_lfq_tryptic_itms2": "high_low_lfq_tryptic.params",
            "human_tmt11_tryptic_itms2": "high_low_tmt11_tryptic.params",
            "human_tmt11_tryptic_otms2": "high_high_tmt11_tryptic.params",
            "human_tmt18_tryptic_itms2": "high_low_tmt18_tryptic.params",
            "human_tmt18_tryptic_otms2": "high_high_tmt18_tryptic.params"
        }

        if param_default not in params_map:
            raise Exception("Invalid parameter preset name provided")

        params_default_file = Path(import_module("proteomics_pipeline").__path__[0]) / "comet_params_defaults" / f"{params_map[param_default]}"
        fasta_file_default = Path(import_module("proteomics_pipeline").__path__[0]) / "comet_params_defaults" 
        
        params_default_file_out = path / f"{params_map[param_default]}"

        if species == "human":
            fasta_file_default_full =   fasta_file_default / 'uniprot_human_20250111_can.fasta'
        elif species == "mouse":
            fasta_file_default_full =   fasta_file_default / 'uniprot_mouse_20250114_can.fasta'
        elif species == "yeast":
            fasta_file_default_full =   fasta_file_default / 'uniprot_yeast_20250114_can.fasta'
        else:
            raise Exception("Must designate 'human', 'mouse', or 'yeast' with species parameter for param_default=True.")

        with open(params_default_file, 'r') as f_in, open(params_default_file_out, 'w') as f_out:
            param_string = f_in.read().strip()
            f_out.write(param_string.format(file_loc=f'{fasta_file_default_full}'))

        shutil.copy(fasta_file_default_full, path)
        params = params_default_file_out

    elif not params:
        raise Exception("Must provide either params file or valid parameter preset name")
    else:
        params = Path(params)
        path_comet = Path(path_comet)
    # Run Comet search on all mzML files in directory
    for ms2_file in path.glob("mzml/*.mzML"):
        run_comet(ms2_file, params, path_comet)

@main.command("detect_ms1_features", help="Use biosaur for ms1 feature detection.")
@click.option("--path", help="Path to project directory.")
@click.option("--out", default=None, help="Path to output files.")
@click.option("--hills", default=5, help="Comma-separated list of hills to run.")
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
    hills: int = 5,
    charge_min: int = 2,
    charge_max: int = 6,
    out: Union[Path, str, None] = None,
) -> None:
    """Run MS1 feature detection using Biosaur2.

    Args:
        path: Path to parent directory
        hills: Miniumum length of points in chromatogram hill values to process (default int: 5)
        charge_min: Minimum charge state to consider (default int: 2)
        charge_max: Maximum charge state to consider (default int: 6)
        out: Optional output directory path. If None, uses "{path}/ms1_features"

    The function processes each mzML file in the path/mzml directory
    using the specified hill values and charge state range.
    """
    path = Path(path)
    if not out:
        out = Path(path) / "ms1_features"
    else:
        out = Path(out) / "ms1_features"

    for mzml_file in path.glob("mzml/*.mzML"):
        run_biosaur2(
            mzml_file,
            out,
            hills,
            charge_min,
            charge_max
        )

            
@main.command("tmt_quant_function", help="Quantify TMT reporter ions from MS2 or MS3 scans.")
@click.option("--path", help="Path to project directory containing mzML files.")
@click.option(
    "--da_tol",
    default=0.003,
    type=float,
    help="Mass tolerance in Daltons for matching reporter ions."
)
@click.option("--ms_level", default=3, help="MS level for TMT reporter ion quantification.")
@click.option("--out", default=None, help="Path to output directory. Defaults to {path}/ms3_features.")
@configure_logger
def _tmt_quant_function(*args: Any, **kwargs: Any) -> None:
    """CLI wrapper for tmt_quant_function."""
    tmt_quant_function(*args, **kwargs)


@arg_logger
def tmt_quant_function(
    path: Union[Path, str],
    da_tol: float = 0.003,
    ms_level: int = 3,
    out: Union[Path, str, None] = None,
) -> None:
    """Extract and quantify TMT reporter ions from MS3 scans.

    Args:
        path: Path to project directory
        da_tol: Mass tolerance in Daltons for matching reporter ions (default float: 0.003)
        ms_level: MS level for TMT reporter ion quantification (default int: 3 for MS3 and 2 for MS2)
        out: Optional output directory path. If None, uses "{path}/ms3_features or {path}/ms2_features"

    The function processes each mzML file in the path/mzml directory,
    extracting TMT reporter ion intensities from MS2 or MS3 scans within
    the specified mass tolerance.
    """

    path = Path(path)
    if not out:
        out = Path(path) / f"ms{ms_level}_features"
    else:
        out = Path(out) / f"ms{ms_level}_features"

    out.mkdir(exist_ok=True, parents=True)
    print(f"Writing output to: {out}")
    for mzml_file in path.glob("mzml/*.mzML"):
        print(f"Processing file: {mzml_file}")
        run_tmt_quant(mzml_file, da_tol, ms_level, out)
        
        
@main.command("train_fdr_model", help="Generate joint pin file for mokapot brew.")
@click.option("--path", help="Path to project.")
@click.option("--fasta", help="Path to fasta file.")
@click.option("--enzyme_regex", default="[KR](?!P)", help="Enzyme regular expression pattern.")
@click.option("--lfq_tmt", default="tmt", type=str, help="Choice of quant type: 'lfq' or 'tmt'")  
@click.option("--ms_level_tmt_quant", default=3, type=int, help="MS level for TMT reporter ion quantification.")
@click.option("--faims", is_flag=True, default=False, help="Whether FAIMS was used.")
@click.option("--out", default=None, help="Path to folder to store results.")
@click.option("--psm_fdr", default=0.01, type=float, help="PSM-level FDR threshold (0 to 1)")
@click.option("--peptide_fdr", default=1.0, type=float, help="Peptide-level FDR threshold (0 to 1)")
@click.option("--protein_fdr", default=0.01, type=float, help="Protein-level FDR threshold (0 to 1)")
@click.option("--threads", type=int, default=3, help="Number of threads for parallel processing")
@click.option("--jobs", type=int, default=3, help="Number of concurrent jobs")
@click.option("--protein_group_off", is_flag=True, default=False, help="Whether protein grouping is turned off.")
@configure_logger
def _train_fdr_model(*args: Any, **kwargs: Any) -> None:
    """CLI wrapper for train_fdr_model."""
    train_fdr_model(*args, **kwargs)


@arg_logger
def train_fdr_model(
    path: Union[Path, str],
    fasta: Union[Path, str] = None,
    enzyme_regex: str = "[KR](?!P)",
    lfq_tmt: str = "tmt",
    ms_level_tmt_quant: int = 3,
    faims: bool = False,
    out: Union[Path, str, None] = None,
    psm_fdr: float = 0.01,
    peptide_fdr: float = 0.01,
    protein_fdr: float = 0.01,
    threads: int = 3,
    jobs: int = 3, 
    protein_group_off: bool = False,
) -> mokapot.dataset.PsmDataset:
    """Train FDR model using mokapot with a joint model to generate psm-level results filtered at the psm, peptide, and protein FDR levels defined by user.

    Args:
        path: Path to project directory
        fasta: Path to FASTA protein sequence database (if not provided, will search for *.fasta in path, only have one fasta in path)
        enzyme_regex: Regular expression pattern for enzyme cleavage (default: "[KR](?!P)", only for protein grouping, need to match search defined in params file)
        lfq_tmt: Quantification type ('lfq' or 'tmt', must run respective ms1_feature_detection or tmt_quant_function first)
        ms_level_tmt_quant: MS level for TMT reporter ion quantification (default (int): 3 for MS3 and 2 for MS2)
        faims: Whether FAIMS was used in the experiment (only needed for 'lfq', bool: False (default), True (faims))
        out: Optional output directory path (defaults to "{path}/mokapot_results")
        psm_fdr: PSM-level FDR threshold (default: 0.01)
        peptide_fdr: Peptide-level FDR threshold (default: 0.01)
        protein_fdr: Protein-level FDR threshold (default: 0.01)
        threads: Number of threads for parallel processing (default: 3)
        jobs: Number of concurrent jobs (default: 3)
        protein_group_off: Whether peptidome was used (default: False)
    Returns:
        PsmDataset containing processed results

    Raises:
        Exception: If lfq_tmt is not 'tmt' or 'lfq'
    """
    path = Path(path)
    if not fasta:
        for fasta_file in (Path(path)).glob("*.fasta"):
            fasta = fasta_file
            break

    if not out:
        out = Path(path) / "mokapot_results"
    else:
        out = Path(out)
    out.mkdir(exist_ok=True)
    
    if not protein_group_off:
        proteins = mokapot.read_fasta(
            fasta,
            enzyme=enzyme_regex,
            decoy_prefix="decoy_",
            missed_cleavages=2,
            min_length=5,
            max_length=50
        )

    psm_list = []
    list_quant_files = []
    output_prefix = []
    for pin_fh in (Path(path) / "mzml").glob("*.pin"):
        pin_file = mokapot.read_pin(pin_fh)
        psm_list.append(pin_file)
        output_prefix.append(f"{pin_fh.stem}")
        
        if lfq_tmt == "tmt":
            handle = (Path(path) / f"ms{ms_level_tmt_quant}_features") / f"{pin_fh.stem}_ms{ms_level_tmt_quant}_quant.csv"
            list_quant_files.append(handle)
        elif lfq_tmt == "lfq":
            handle = (Path(path) / "ms1_features") / f"{pin_fh.stem}.features.tsv"
            list_quant_files.append(handle)
        else:
            raise Exception("Must designate 'tmt' or 'lfq' with lfq_tmt parameter.")

    if not protein_group_off:
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
    if not protein_group_off:
        dump_models_and_results(
            psm_list,
            models,
            results,
            output_prefix,
            list_quant_files,
            path,
            out,
            psm_fdr,
            peptide_fdr,
            protein_fdr,
            lfq_tmt,
            ms_level_tmt_quant,
            faims,
        )
    else:
        dump_models_and_results_no_protein_group(
            psm_list,
            models,
            results,
            output_prefix,
            list_quant_files,
            path,
            out,
            psm_fdr,
            peptide_fdr,
            protein_fdr,
            lfq_tmt,
            ms_level_tmt_quant,
            faims,
        )


@main.command("pyascore_site_localization", help="Perform site localization with PyAscore.")
@click.option("--path", help="Path to project.")
@click.option("--modification_dict_add", help="Add modifications to PyAscore dictionary if not standard Mox, Camid, or phosphoSTY.")
@click.option("--static_mod_dictionary", help="Add static modifications to Pyascore Identification Parser.")
@click.option("--ascore_mod_mass", help="Add modification mass to Pyascore Identification Parser.")
@click.option("--ascore_aa", help="Add modification mass to Pyascore Identification Parser.")
@click.option("--out", default=None, help="Path to folder to store results.")
@configure_logger
def _train_fdr_model(*args: Any, **kwargs: Any) -> None:
    """CLI wrapper for train_fdr_model."""
    train_fdr_model(*args, **kwargs)

@arg_logger
def pyascore_site_localization(
    path: Union[Path, str],
    modification_dict_add: Dict[str, float] = None,
    static_mod_dictionary: Dict[str, float] = None,
    ascore_mod_mass: float = None,
    ascore_aa: str = None,
    out: Union[Path, str, None] = None,
) -> None:
    """Run PyAscore site localization on PSM results.

    Args:
        path (Union[Path, str]): Path to project directory
        modification_dict_add (Dict[str, float], optional): Dictionary of modifications to add to PyAscore dictionary if not standard Mox, Camid, or phosphoSTY. Defaults to None.
        static_mod_dictionary (Dict[str, float], optional): Dictionary of static modifications to add to Pyascore Identification Parser (eg {'C': 57.021464, 'nK': 304.207}). Defaults to None.
        ascore_mod_mass (float, optional): Modification mass to use for site localization scoring. Defaults to None.
        ascore_aa (str, optional): Amino acid residue(s) to consider for site localization. Defaults to None.
        out (Union[Path, str, None], optional): Output directory path. Defaults to "{path}/site_localization_outputs".

    Current modification dictionary:
    modifications = {"n": 42.010565, # N-term acetylation
                 "M": 15.9949,   # Methionine oxidation
                 "S": 79.966331, # Serine Phoshorylation
                 "T": 79.966331, # Threonine Phosphorylation
                 "Y": 79.966331, # Tyrosine Phosphorylation
                 "C": 57.021464}

    """
    path = Path(path)

    if not out:
        out = Path(path) / "site_localization_outputs"
    else:
        out = Path(out)
    out.mkdir(exist_ok=True)

    list_ascore_outputs = []
    for psm_fh in (Path(path) / "mokapot_results").glob("*_pyascore_input.txt"):
        base_name = psm_fh.stem.replace("_pyascore_input", "")
        mzml_file_path = (Path(path) / f"mzml") / f"{base_name}.mzML"
        
        ascore_output = run_pyascore(
            str(psm_fh),
            str(mzml_file_path),            
            modification_dict_add,
            static_mod_dictionary,
            ascore_mod_mass,
            ascore_aa
        )
        ascore_output.to_csv(out / f"{base_name}_quant_fdr_site_localization_results.csv")  
        list_ascore_outputs.append(ascore_output)     
    if len(list_ascore_outputs) > 1:
        combined_output = pd.concat(list_ascore_outputs, axis=0, ignore_index=True)
        combined_output.to_csv(out / "combined_quant_fdr_site_localization_results.csv")
        
if __name__ == "__main__":
    main()