from pathlib import Path
from typing import *
from multiprocessing import Pool
from itertools import repeat
import re

import pandas as pd
import numpy as np
import pyopenms as oms
from tqdm import tqdm
from joblib import dump
import mokapot

import sys
from re import search
from numpy import median
import os

from pyteomics import mzml

class MS2OnlyMzML(mzml.MzML):
    """Reads ms2 scans from mzml."""

    _default_iter_path = (
        '//spectrum[./*[local-name()="cvParam" and @name="ms level" and @value="2"]]'
    )
    _use_index = False
    _iterative = False
    
def parse_scan_ms2(s: Dict[str, Any], faims=False) -> Dict[str, Any]:
    """ """
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

#for r, p, qf, op in zip(results, pin, list_quant_files, out_prefix):

# function to run msconvert 


def retention_time_fetch(path,op, faims):
    mzml_file_path = Path(path) / 'mzml' / f'{op}.mzML' 
    mzml_file_path = Path(mzml_file_path)
    # out_path = path / "parquet"
    # out_path.mkdir(exist_ok=True)
    # for input_mzml_path in path.glob("mzml/*.mzML"):
    scan_list_ms2 = []
    for z in MS2OnlyMzML(source=str(mzml_file_path)):
        if z["ms level"] == 2:
            mzml_read7 = parse_scan_ms2(z, faims)
            scan_list_ms2.append(mzml_read7)
    scan_list_ms2 = [s for s in scan_list_ms2 if s is not None]
    scan_list_ms2_full = pd.DataFrame.from_records(scan_list_ms2)
    return scan_list_ms2_full
    


def merge_ms1_quant(psm_filtered_by_peptides, faims, path, op):
    ## read in the .csv output from Dinosaur (Dinosaur run just on the mzML itself)
    dinosaur_file_path = Path(path) / 'ms1_features' / f'{op}.features.tsv' 
    dinosaur = pd.read_csv(dinosaur_file_path, delimiter = "\t")
    summary_quants=[]
    ### Iterate through all PSMs to append the MS1 quantifications
    for i in range(0,len(psm_filtered_by_peptides)):
        ## Extract by row from PSM dataframe
        test_example = psm_filtered_by_peptides.iloc[i]
        charge_val = test_example["Charge"]
        mass_val = test_example["CalcMass"]
        scan_num = test_example["ScanNr"]
        rt_val = test_example["RetentionTime"]
        
        ## Calculate the theoretical mass of the monoisotopic peak for matching to the Dinosaur data
        theo_mz = (mass_val/charge_val) + ((1.007825 * (charge_val -1))/charge_val)
        ### feature map within boundaries of peak retention time and 50 ppm from monoisotopic peak mass
        if faims:
            psm=dinosaur[(dinosaur["rtStart"]<test_example["RetentionTime"])&(dinosaur["charge"]==test_example["Charge"])  & (dinosaur["rtEnd"]>test_example["RetentionTime"]) & (dinosaur["mz"]> theo_mz-dinosaur["mz"]*0.000025) & (dinosaur["mz"]< theo_mz+dinosaur["mz"]*0.000025) & (dinosaur["FAIMS"] == test_example["FAIMScv"])]
        else:
            psm=dinosaur[(dinosaur["rtStart"]<test_example["RetentionTime"])&(dinosaur["charge"]==test_example["Charge"])  & (dinosaur["rtEnd"]>test_example["RetentionTime"]) & (dinosaur["mz"]> theo_mz-dinosaur["mz"]*0.000025) & (dinosaur["mz"]< theo_mz+dinosaur["mz"]*0.000025) ]
        if psm.shape[0]>1:
            index_num3=psm["intensityApex"].idxmax()
            psm=psm[psm.index == index_num3]
        original_peptide=  test_example["Peptide"][2:-2]
        psm = psm.assign(scan=scan_num)
        summary_quants.append(psm)    
    dino_df_quants=pd.concat(summary_quants)
    
    psm_filtered_by_peptides = pd.merge(psm_filtered_by_peptides, dino_df_quants, left_on='ScanNr', right_on='scan', how='left')
    return(psm_filtered_by_peptides)


# def merge_ms1(summary_df, dino_df_quants):
#     final_summary_df=pd.merge(summary_df, dino_df_quants, left_on='ScanNr', right_on='scan', how='left')
#     return(final_summary_df)


def merge_tmt(fdr_input, path, out_prefix):
    # tmt_quant_df = pd.read_csv(mzml_file.strip("\.mzML") + "_ms3_quant.csv", delimiter=',')
    tmt_quant_path = Path(path) / 'ms3_features' / f'{out_prefix}_ms3_quant.csv' 
    # tmt_quant_path = str(tmt_quant_path)
    tmt_quant_df = pd.read_csv(tmt_quant_path, delimiter=',')
    final_summary_df=pd.merge(fdr_input, tmt_quant_df, left_on='ScanNr', right_on='RefScan', how='left')
    return(final_summary_df)







def dump_models_and_results(
    pin: Union[mokapot.dataset.PsmDataset, List[mokapot.dataset.PsmDataset]],
    models: Tuple[mokapot.Model],
    results: Union[
        mokapot.confidence.LinearConfidence,
        List[mokapot.dataset.PsmDataset],
    ],
    out_prefix: Union[str, List[str]],
    list_quant_files: Union[str, List[str]],
    path: Path,
    out_loc: Path,
    psm_fdr: int=0.01,
    peptide_fdr: int=1,
    protein_fdr: int=0.01,
    lfq_tmt: str = "tmt",
    faims: bool = False,
    
    # model_prefix: Union[str, None] = None,
) -> pd.DataFrame:
    """ """
    if hasattr(pin, "__iter__"):
        if not hasattr(results, "__iter__"):
            raise Exception(
                "pin and results must both be iterables if one is an iterable."
            )
        if not isinstance(out_prefix, list):
            raise Exception(
                "out_prefix must be a list of prefixes if pin is an iterable."
            )
        # if not model_prefix:
        #     raise Exception("Must provide model prefix if pin is an iterable.")

        # psm_dfs = []
        psm_filtered_dfs = []

        for r, p, qf, op in zip(results, pin, list_quant_files, out_prefix):
            # r._data.to_parquet(out_loc / f"{op}_scores.parquet")
            psm_df = pd.concat(
                [
                    r.confidence_estimates["psms"],
                    r.decoy_confidence_estimates["psms"],
                ],
                axis=0,
                ignore_index=True,
            )
            psm_df = p._data.merge(
                psm_df.loc[
                    :,
                    ["SpecId"]
                    + psm_df.columns[psm_df.columns.str.startswith("mokapot")].tolist(),
                ],
                on="SpecId",
                how="left",
            )
            # psm_df.to_parquet(out_loc / f"{op}_psms.parquet")

            peptide_df = pd.concat(
                [
                    r.confidence_estimates["peptides"],
                    r.decoy_confidence_estimates["peptides"],
                ],
                axis=0,
                ignore_index=True,
            )            
            
            peptide_df = p._data.merge(
                peptide_df.loc[
                    :,
                    ["SpecId"]
                    + peptide_df.columns[
                        peptide_df.columns.str.startswith("mokapot")
                    ].tolist(),
                ],
                on="SpecId",
                how="left",
            )
            # peptide_df.to_parquet(out_loc / f"{op}_peptides.parquet")
            # psm_dfs.append(psm_df)
            
            protein_df = pd.concat(
                [
                    r.confidence_estimates["proteins"],
                    r.decoy_confidence_estimates["proteins"],
                ],
                axis=0,
                ignore_index=True,
            )
            
            
            # protein_df.to_parquet(out_loc / f"{op}_proteins.parquet")
            
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
                psm_df["Peptide"].isin(list_peptides_unique)
                & ~psm_df["mokapot score"].isna()
                & psm_df["Proteins"].isin(list_proteins_unique)
            ].copy()
            
            psm_filtered_by_peptides = psm_filtered_by_peptides[
                psm_filtered_by_peptides["mokapot q-value"] < 0.01
            ].copy()
            
            ## fetch retention times from mzml and append to id file 
            retention_time_df = retention_time_fetch(path, op,faims)   
            psm_filtered_by_peptides = pd.merge(psm_filtered_by_peptides, retention_time_df,left_on='ScanNr', right_on='ScanNum', how='left')
            
            # psm_filtered_by_peptides['Charge'] = psm_filtered_by_peptides['SpecId'].str.split("_").str[-2]
            
            ## append quantification
            if lfq_tmt == "tmt":
                psm_filtered_by_peptides  = merge_tmt(psm_filtered_by_peptides, path, op)                
            elif lfq_tmt == "lfq":              
                psm_filtered_by_peptides = merge_ms1_quant(psm_filtered_by_peptides, faims, path, op)
                # final_df = merge_ms1(summarized_psms, dino_df_quants)
            else:
                raise Exception("Must designate tmt or lfq with lfq_tmt parameter.")
            
            psm_filtered_by_peptides.to_csv(
                out_loc / f"{op}_psm{psm_fdr}_peptide{peptide_fdr}_protein{protein_fdr}_fdr.csv"
            )
                       
            psm_filtered_dfs.append(psm_filtered_by_peptides)

        psm_filtered_by_peptides = pd.concat(
            psm_filtered_dfs,
            ignore_index=True,
        )
        psm_filtered_by_peptides.to_csv(
            out_loc / f"combined_psm{psm_fdr}_peptide{peptide_fdr}_protein{protein_fdr}_fdr.csv"
        )

    else:
        # if not model_prefix:
        #     model_prefix = out_prefix
        # results._data.to_parquet(out_loc / f"{out_prefix}_scores.parquet")
        psm_df = pd.concat(
            [
                results.confidence_estimates["psms"],
                results.decoy_confidence_estimates["psms"],
            ],
            axis=0,
            ignore_index=True,
        )
        psm_df = pin._data.merge(
            psm_df.loc[
                :,
                ["SpecId"]
                + psm_df.columns[psm_df.columns.str.startswith("mokapot")].tolist(),
            ],
            on="SpecId",
            how="left",
        )
        # psm_df.to_parquet(out_loc / f"{out_prefix}_psms.parquet")

        peptide_df = pd.concat(
            [
                results.confidence_estimates["peptides"],
                results.decoy_confidence_estimates["peptides"],
            ],
            axis=0,
            ignore_index=True,
        )
        peptide_df = pin._data.merge(
            peptide_df.loc[
                :,
                ["SpecId"]
                + peptide_df.columns[
                    peptide_df.columns.str.startswith("mokapot")
                ].tolist(),
            ],
            on="SpecId",
            how="left",
        )
        # peptide_df.to_parquet(out_loc / f"{out_prefix}_peptides.parquet")

        
        
        
        ###
        protein_df = pd.concat(
                [
                    r.confidence_estimates["proteins"],
                    r.decoy_confidence_estimates["proteins"],
                ],
                axis=0,
                ignore_index=True,
            )
            
            
        # protein_df.to_parquet(out_loc / f"{out_prefix}_proteins.parquet")
        
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
            psm_df["Peptide"].isin(list_peptides_unique)
            & ~psm_df["mokapot score"].isna()
            & psm_df["Proteins"].isin(list_proteins_unique)
        ].copy()

        psm_filtered_by_peptides = psm_filtered_by_peptides[
            psm_filtered_by_peptides["mokapot q-value"] < 0.01
        ].copy()
        
        ## fetch retention times from mzml and append to id file 
        retention_time_df = retention_time_fetch(path, op,faims)   
        psm_filtered_by_peptides = pd.merge(psm_filtered_by_peptides, retention_time_df,left_on='ScanNr', right_on='ScanNum', how='left')
        
        ## append quantification
        if lfq_tmt == "tmt":
            psm_filtered_by_peptides  = merge_tmt(psm_filtered_by_peptides, path, op)                
        elif lfq_tmt == "lfq":
            # psm_filtered_by_peptides  = merge_tmt(psm_filtered_by_peptides, path, op)
            psm_filtered_by_peptides = merge_ms1_quant(psm_filtered_by_peptides, faims, path, op)
            # final_df = merge_ms1(summarized_psms, dino_df_quants)
        else:
            raise Exception("Must designate tmt or lfq with lfq_tmt parameter.")
        
        psm_filtered_by_peptides.to_csv(
            out_loc / f"{out_prefix}_psm{psm_fdr}_peptide{peptide_fdr}_protein{protein_fdr}_fdr.csv"
        )
        # psm_filtered_by_peptides.to_parquet(
        #     out_loc / f"{out_prefix}_psm_peptide_fdr_filtered.parquet"
        # )

    dump(
        models,
        out_loc / "svc.model",
    )

    return psm_filtered_by_peptides
