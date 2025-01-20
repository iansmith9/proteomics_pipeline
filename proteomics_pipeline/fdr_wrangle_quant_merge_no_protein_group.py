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
from .fdr_wrangle_quant_merge import *



def dump_models_and_results_no_protein_group(
    pin: Union[mokapot.dataset.PsmDataset, List[mokapot.dataset.PsmDataset]],
    models: Tuple[mokapot.Model],
    results: Union[mokapot.confidence.LinearConfidence, List[mokapot.dataset.PsmDataset]],
    out_prefix: Union[str, List[str]],
    list_quant_files: Union[str, List[str]],
    path: Path,
    out: Path,
    psm_fdr: float = 0.01,
    peptide_fdr: float = 0.01,
    protein_fdr: float = 1,
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
            # r.to_txt(dest_dir = out, file_root = op, sep="\t", decoys=False)
            # result_files = [r.to_txt(file_root=l) for l, r in zip(labels, result_list)]

            psm_df = pd.concat([
                r.confidence_estimates["psms"],
                r.decoy_confidence_estimates["psms"]
            ], axis=0, ignore_index=True)
            
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

            # protein_df = pd.concat([
            #     r.confidence_estimates["proteins"],
            #     r.decoy_confidence_estimates["proteins"]
            # ], axis=0, ignore_index=True)

            # if not protein_fdr:
            #     list_proteins_unique = protein_df["mokapot protein group"].unique()
            # else:
            #     protein_df = protein_df[protein_df["mokapot q-value"] < 0.01].copy()
            #     list_proteins_unique = protein_df["mokapot protein group"].unique()

            if not peptide_fdr:
                list_peptides_unique = peptide_df["Peptide"].unique()
            else:
                peptide_df = peptide_df[peptide_df["mokapot q-value"] < 0.01].copy()
                list_peptides_unique = peptide_df["Peptide"].unique()

            psm_filtered_by_peptides = psm_df[
                psm_df["Peptide"].isin(list_peptides_unique) &
                ~psm_df["mokapot score"].isna() 
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

            psm_filtered_by_peptides["filename"] = str(op)

            list_proteins = psm_filtered_by_peptides["Proteins"]
            lead_protein =[]
            redundancy = []
            for i in list_proteins:
                lead_protein.append(i.split("\t")[0])
                redundancy.append(len(i.split("\t"))-1)
            psm_filtered_by_peptides["lead_protein"] = lead_protein
            psm_filtered_by_peptides["redundancy"] = redundancy

            psm_filtered_by_peptides.to_csv(
                out / f"{op}_psm{psm_fdr}_peptide{peptide_fdr}_protein{protein_fdr}_fdr.csv"
            )
            psm_filtered_by_peptides.to_csv(
                out / f"{op}_pyascore_input.txt" , sep ="\t"
            )

            psm_filtered_dfs.append(psm_filtered_by_peptides)

        psm_filtered_by_peptides = pd.concat(psm_filtered_dfs, ignore_index=True)


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

        # protein_df = pd.concat([
        #     results.confidence_estimates["proteins"],
        #     results.decoy_confidence_estimates["proteins"]
        # ], axis=0, ignore_index=True)

        # if not protein_fdr:
        #     list_proteins_unique = protein_df["mokapot protein group"].unique()
        # else:
        #     protein_df = protein_df[protein_df["mokapot q-value"] < 0.01].copy()
        #     list_proteins_unique = protein_df["mokapot protein group"].unique()

        if not peptide_fdr:
            list_peptides_unique = peptide_df["Peptide"].unique()
        else:
            peptide_df = peptide_df[peptide_df["mokapot q-value"] < 0.01].copy()
            list_peptides_unique = peptide_df["Peptide"].unique()

        psm_filtered_by_peptides = psm_df[
            psm_df["Peptide"].isin(list_peptides_unique) &
            ~psm_df["mokapot score"].isna() 
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
        psm_filtered_by_peptides["filename"] = str(op)
        list_proteins = psm_filtered_by_peptides["Proteins"]
        lead_protein =[]
        redundancy = []
        for i in list_proteins:
            lead_protein.append(i.split("\t")[0])
            redundancy.append(len(i.split("\t"))-1)
        psm_filtered_by_peptides["lead_protein"] = lead_protein
        psm_filtered_by_peptides["redundancy"] = redundancy
        
        psm_filtered_by_peptides.to_csv(
            out / f"{out_prefix}_psm{psm_fdr}_peptide{peptide_fdr}_protein{protein_fdr}_fdr.csv"
        )
        psm_filtered_by_peptides.to_csv(
            out / f"{out_prefix}_pyascore_input.txt" , sep ="\t"
        )

    dump(models, out / "svc.model")

    return psm_filtered_by_peptides
