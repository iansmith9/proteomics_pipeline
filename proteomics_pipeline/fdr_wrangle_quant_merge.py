"""Functions for processing and merging FDR results with quantification data."""

from pathlib import Path
from typing import *
from multiprocessing import Pool
from itertools import repeat
import re
import sys
import os
from re import search
from numpy import median

import pandas as pd
import numpy as np
import pyopenms as oms
from tqdm import tqdm
from joblib import dump
import mokapot
from pyteomics import mzml


class MS2OnlyMzML(mzml.MzML):
    """Reads MS2 scans from mzML files.
    
    Extends pyteomics.mzml.MzML to only read MS2 level scans.
    """

    _default_iter_path = (
        '//spectrum[./*[local-name()="cvParam" and @name="ms level" and @value="2"]]'
    )
    _use_index = False
    _iterative = False
    
def parse_scan_ms2(s: Dict[str, Any], faims=False) -> Dict[str, Any]:
    """Parse MS2 scan metadata from mzML scan dictionary.

    Args:
        s: Dictionary containing scan metadata from mzML
        faims: Whether FAIMS data is present

    Returns:
        Dictionary containing parsed scan metadata, or None if not MS2
    """
    scan_dict = {}
    # Only extracting MS/MS data: removing the MS1 scans
    if s["ms level"] == 1:
        return None
    # Scan Number
    scan_dict["ScanNum"] = s["index"] + 1
    scan_dict["RetentionTime"] = s["scanList"]["scan"][0]["scan start time"]
    scan_dict["Charge"]=s["precursorList"]["precursor"][0]["selectedIonList"]["selectedIon"][0]["charge state"]
    if faims:
        scan_dict["FAIMScv"] = s['FAIMS compensation voltage']

    return scan_dict


def retention_time_fetch(path: Union[str, Path], op: str, faims: bool) -> pd.DataFrame:
    """Extract retention time information from mzML file.

    Args:
        path: Path to data directory
        op: Output prefix for file naming
        faims: Whether FAIMS data is present

    Returns:
        DataFrame containing scan metadata
    """
    mzml_file_path = Path(path) / 'mzml' / f'{op}.mzML'
    scan_list_ms2 = []
    
    for z in MS2OnlyMzML(source=str(mzml_file_path)):
        if z["ms level"] == 2:
            mzml_read = parse_scan_ms2(z, faims)
            scan_list_ms2.append(mzml_read)
            
    scan_list_ms2 = [s for s in scan_list_ms2 if s is not None]
    return pd.DataFrame.from_records(scan_list_ms2)


def merge_ms1_quant(psm_filtered_by_peptides: pd.DataFrame, faims: bool, 
                   path: Union[str, Path], op: Union[str, Path]) -> pd.DataFrame:
    """Merge MS1 quantification data with PSM results.

    Args:
        psm_filtered_by_peptides: DataFrame containing filtered PSM results
        faims: Whether FAIMS data is present  
        path: Path to data directory
        op: Output prefix for file naming

    Returns:
        DataFrame with merged MS1 quantification data
    """
    dinosaur_file_path = Path(path) / 'ms1_features' / f'{op}.features.tsv'
    dinosaur = pd.read_csv(dinosaur_file_path, delimiter="\t")
    summary_quants = []

    for i in range(0,len(psm_filtered_by_peptides)):
        test_example = psm_filtered_by_peptides.iloc[i]
        charge_val = test_example["Charge"]
        mass_val = test_example["CalcMass"]
        scan_num = test_example["ScanNr"]
        rt_val = test_example["RetentionTime"]

        theo_mz = (mass_val/charge_val) + ((1.007825 * (charge_val - 1))/charge_val)
        
        if faims:
            psm = dinosaur[
                (dinosaur["rtStart"] < test_example["RetentionTime"]) &
                (dinosaur["charge"] == test_example["Charge"]) & 
                (dinosaur["rtEnd"] > test_example["RetentionTime"]) &
                (dinosaur["mz"] > theo_mz - dinosaur["mz"]*0.000025) &
                (dinosaur["mz"] < theo_mz + dinosaur["mz"]*0.000025) &
                (dinosaur["FAIMS"] == test_example["FAIMScv"])
            ]
        else:
            psm = dinosaur[
                (dinosaur["rtStart"] < test_example["RetentionTime"]) &
                (dinosaur["charge"] == test_example["Charge"]) &
                (dinosaur["rtEnd"] > test_example["RetentionTime"]) &
                (dinosaur["mz"] > theo_mz - dinosaur["mz"]*0.000025) &
                (dinosaur["mz"] < theo_mz + dinosaur["mz"]*0.000025)
            ]

        if psm.shape[0] > 1:
            index_num3 = psm["intensityApex"].idxmax()
            psm = psm[psm.index == index_num3]
            
        original_peptide = test_example["Peptide"][2:-2]
        psm = psm.assign(scan=scan_num)
        summary_quants.append(psm)

    dino_df_quants = pd.concat(summary_quants)
    return pd.merge(psm_filtered_by_peptides, dino_df_quants, 
                   left_on='ScanNr', right_on='scan', how='left')


def merge_tmt(fdr_input: pd.DataFrame, path: Union[str, Path],ms_level_tmt_quant: int, 
              out_prefix: str) -> pd.DataFrame:
    """Merge TMT quantification data with PSM results.

    Args:
        fdr_input: DataFrame containing PSM results
        path: Path to data directory  
        ms_level_tmt_quant: MS level for TMT reporter ion quantification
        out_prefix: Output prefix for file naming

    Returns:
        DataFrame with merged TMT quantification data
    """
    tmt_quant_path = Path(path) / f'ms{ms_level_tmt_quant}_features' / f'{out_prefix}_ms{ms_level_tmt_quant}_quant.csv'
    tmt_quant_df = pd.read_csv(tmt_quant_path, delimiter=',')
    if ms_level_tmt_quant == 2:
        return pd.merge(fdr_input, tmt_quant_df, left_on='ScanNr', 
                   right_on='ScanNum', how='left')
    elif ms_level_tmt_quant == 3:
        return pd.merge(fdr_input, tmt_quant_df, left_on='ScanNr', 
                   right_on='RefScan', how='left')


def dump_models_and_results(
    pin: Union[mokapot.dataset.PsmDataset, List[mokapot.dataset.PsmDataset]],
    models: Tuple[mokapot.Model],
    results: Union[mokapot.confidence.LinearConfidence, List[mokapot.dataset.PsmDataset]],
    out_prefix: Union[str, List[str]],
    list_quant_files: Union[str, List[str]],
    path: Path,
    out: Path,
    psm_fdr: float = 0.01,
    peptide_fdr: float = 1.0,
    protein_fdr: float = 0.01,
    lfq_tmt: str = "tmt",
    ms_level_tmt_quant: int = 3,
    faims: bool = False,
) -> pd.DataFrame:
    """Process FDR results and merge with quantification data.

    Args:
        pin: PSM dataset(s) from mokapot
        models: Trained mokapot models
        results: Confidence results from mokapot
        out_prefix: Output prefix(es) for file naming
        list_quant_files: List of quantification files
        path: Path to data directory
        out: Output directory path
        psm_fdr: PSM-level FDR threshold
        peptide_fdr: Peptide-level FDR threshold  
        protein_fdr: Protein-level FDR threshold
        lfq_tmt: Quantification type ('tmt' or 'lfq')
        ms_level_tmt_quant: MS level for TMT reporter ion quantification
        faims: Whether FAIMS data is present

    Returns:
        DataFrame containing processed and merged results

    Raises:
        Exception: If input parameters are inconsistent or invalid
    """
    if hasattr(pin, "__iter__"):
        if not hasattr(results, "__iter__"):
            raise Exception("pin and results must both be iterables if one is an iterable.")
        if not isinstance(out_prefix, list):
            raise Exception("out_prefix must be a list of prefixes if pin is an iterable.")

        psm_filtered_dfs = []

        for r, p, qf, op in zip(results, pin, list_quant_files, out_prefix):
            # labels = ["exp_1", "exp_2", "exp_3"]
            r.to_txt(dest_dir = out, file_root = op, sep="\t", decoys=False)
            # result_files = [r.to_txt(file_root=l) for l, r in zip(labels, result_list)]

            psm_df = pd.concat([
                r.confidence_estimates["psms"],
                r.decoy_confidence_estimates["psms"]
            ], axis=0, ignore_index=True)
            psm_df["filename"] = str(op)
            psm_df = p._data.merge(
                psm_df.loc[:, ["SpecId"] + 
                psm_df.columns[psm_df.columns.str.startswith("mokapot")].tolist()],
                on="SpecId",
                how="left"
            )

            peptide_df = pd.concat([
                r.confidence_estimates["peptides"],
                r.decoy_confidence_estimates["peptides"]
            ], axis=0, ignore_index=True)

            peptide_df = p._data.merge(
                peptide_df.loc[:, ["SpecId"] + 
                peptide_df.columns[peptide_df.columns.str.startswith("mokapot")].tolist()],
                on="SpecId",
                how="left"
            )

            protein_df = pd.concat([
                r.confidence_estimates["proteins"],
                r.decoy_confidence_estimates["proteins"]
            ], axis=0, ignore_index=True)

            if not protein_fdr:
                list_proteins_unique = protein_df["mokapot protein group"].unique()
            else:
                protein_df = protein_df[protein_df["mokapot q-value"] < 0.01].copy()
                list_proteins_unique = protein_df["mokapot protein group"].unique()

            if not peptide_fdr:
                list_peptides_unique = peptide_df["Peptide"].unique()
            else:
                peptide_df = peptide_df[peptide_df["mokapot q-value"] < 0.01].copy()
                list_peptides_unique = peptide_df["Peptide"].unique()

            psm_filtered_by_peptides = psm_df[
                psm_df["Peptide"].isin(list_peptides_unique) &
                ~psm_df["mokapot score"].isna() &
                psm_df["Proteins"].isin(list_proteins_unique)
            ].copy()

            psm_filtered_by_peptides = psm_filtered_by_peptides[
                psm_filtered_by_peptides["mokapot q-value"] < 0.01
            ].copy()

            retention_time_df = retention_time_fetch(path, op, faims)
            psm_filtered_by_peptides = pd.merge(
                psm_filtered_by_peptides, 
                retention_time_df,
                left_on='ScanNr',
                right_on='ScanNum',
                how='left'
            )

            if lfq_tmt == "tmt":
                psm_filtered_by_peptides = merge_tmt(psm_filtered_by_peptides, path, ms_level_tmt_quant, op)
            elif lfq_tmt == "lfq":
                psm_filtered_by_peptides = merge_ms1_quant(
                    psm_filtered_by_peptides, faims, path, op
                )
            else:
                raise Exception("Must designate 'tmt' or 'lfq' with lfq_tmt parameter.")

            psm_filtered_by_peptides.to_csv(
                out / f"{op}_psm{psm_fdr}_peptide{peptide_fdr}_protein{protein_fdr}_fdr.csv"
            )

            psm_filtered_dfs.append(psm_filtered_by_peptides)

        psm_filtered_by_peptides = pd.concat(psm_filtered_dfs, ignore_index=True)

            ###
        lead_protein =[]
        redundancy = []
        for i in list_proteins:
            lead_protein.append(i.split("\t")[0])
            redundancy.append(len(lead_protein)-1)
        psm_filtered_by_peptides["lead_protein"] = lead_protein
        psm_filtered_by_peptides["redundancy"] = redundancy


        psm_filtered_by_peptides.to_csv(
            out / f"combined_psm{psm_fdr}_peptide{peptide_fdr}_protein{protein_fdr}_fdr.csv"
        )

    else:
        psm_df = pd.concat([
            results.confidence_estimates["psms"],
            results.decoy_confidence_estimates["psms"]
        ], axis=0, ignore_index=True)

        psm_df = pin._data.merge(
            psm_df.loc[:, ["SpecId"] + 
            psm_df.columns[psm_df.columns.str.startswith("mokapot")].tolist()],
            on="SpecId",
            how="left"
        )

        peptide_df = pd.concat([
            results.confidence_estimates["peptides"],
            results.decoy_confidence_estimates["peptides"]
        ], axis=0, ignore_index=True)

        peptide_df = pin._data.merge(
            peptide_df.loc[:, ["SpecId"] + 
            peptide_df.columns[peptide_df.columns.str.startswith("mokapot")].tolist()],
            on="SpecId",
            how="left"
        )

        protein_df = pd.concat([
            results.confidence_estimates["proteins"],
            results.decoy_confidence_estimates["proteins"]
        ], axis=0, ignore_index=True)

        if not protein_fdr:
            list_proteins_unique = protein_df["mokapot protein group"].unique()
        else:
            protein_df = protein_df[protein_df["mokapot q-value"] < 0.01].copy()
            list_proteins_unique = protein_df["mokapot protein group"].unique()

        if not peptide_fdr:
            list_peptides_unique = peptide_df["Peptide"].unique()
        else:
            peptide_df = peptide_df[peptide_df["mokapot q-value"] < 0.01].copy()
            list_peptides_unique = peptide_df["Peptide"].unique()

        psm_filtered_by_peptides = psm_df[
            psm_df["Peptide"].isin(list_peptides_unique) &
            ~psm_df["mokapot score"].isna() &
            psm_df["Proteins"].isin(list_proteins_unique)
        ].copy()

        psm_filtered_by_peptides = psm_filtered_by_peptides[
            psm_filtered_by_peptides["mokapot q-value"] < 0.01
        ].copy()

        retention_time_df = retention_time_fetch(path, op, faims)
        psm_filtered_by_peptides = pd.merge(
            psm_filtered_by_peptides,
            retention_time_df,
            left_on='ScanNr',
            right_on='ScanNum',
            how='left'
        )

        if lfq_tmt == "tmt":
            psm_filtered_by_peptides = merge_tmt(psm_filtered_by_peptides, path, ms_level_tmt_quant, op)
        elif lfq_tmt == "lfq":
            psm_filtered_by_peptides = merge_ms1_quant(
                psm_filtered_by_peptides, faims, path, op
            )
        else:
            raise Exception("Must designate 'tmt' or 'lfq' with lfq_tmt parameter.")
        #add lead protein and redundancy columns
        lead_protein =[]
        redundancy = []
        for i in list_proteins:
            lead_protein.append(i.split("\t")[0])
            redundancy.append(len(lead_protein)-1)
        psm_filtered_by_peptides["lead_protein"] = lead_protein
        psm_filtered_by_peptides["redundancy"] = redundancy
        
        psm_filtered_by_peptides.to_csv(
            out / f"{out_prefix}_psm{psm_fdr}_peptide{peptide_fdr}_protein{protein_fdr}_fdr.csv"
        )

    dump(models, out / "svc.model")

    return psm_filtered_by_peptides
