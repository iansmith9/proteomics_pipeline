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
import pyascore

def run_pyascore(
    psm_path: str = None,
    mzml_path: str = None,
    modification_dict_add: Dict[str, str] = None,
    static_mod_dictionary: Dict[str, str] = None,
    ascore_mod_mass: float = None,
    ascore_aa: str = None,
) -> pd.DataFrame:
    """Run PyAscore site localization on PSM results.

    Args:
        psm_path: Path to PSM results file
        mzml_path: Path to mzML spectrum file
        modification_dict_add: Dictionary of additional modifications to add to PyAscore dictionary 
            beyond standard Mox, Camid, or phosphoSTY
        static_mod_dictionary: Dictionary of static modifications to add to PyAscore Identification Parser
        ascore_mod_mass: Modification mass to use for site localization scoring
        ascore_aa: Amino acid residue(s) to consider for site localization

    Returns:
        DataFrame containing site localization results
    """

    modifications = {"n": 42.010565, # N-term acetylation
                 "M": 15.9949,   # Methionine oxidation
                 "S": 79.966331, # Serine Phoshorylation
                 "T": 79.966331, # Threonine Phosphorylation
                 "Y": 79.966331, # Tyrosine Phosphorylation
                 "C": 57.021464} # Cysteine Carbamidomethylation

    modifications.update(modification_dict_add)
    mass_corrector_val = pyascore.MassCorrector(modifications, mz_tol = 1.5)

    id_parser = pyascore.IdentificationParser(psm_path, id_file_format = "mokapotTXT", static_mods = static_mod_dictionary, mass_corrector = mass_corrector_val)
    spectral_parser = pyascore.SpectraParser(mzml_path, 'mzML')
    mod_mass = ascore_mod_mass
    ascore = pyascore.PyAscore(bin_size=100, n_top=10, mod_group = ascore_aa, mod_mass = mod_mass)
    psm_objects = id_parser.to_list()
    spectra_objects = spectral_parser.to_dict()


    # 4) Score PSMs
    pyascore_results = []
    for psm in psm_objects:
        # 4.1) Check for modification of interest
        mod_select = np.isclose(psm["mod_masses"], mod_mass)
        nmods = np.sum(mod_select)

        if nmods >= 1:
            # 4.2) Grab spectrum
            spectrum = spectra_objects[psm["scan"]]

            # 4.3) Gather other modifications into aux mods
            aux_mod_pos = psm["mod_positions"][~mod_select].astype(np.uint32)
            aux_mod_masses = psm["mod_masses"][~mod_select].astype(np.float32)

            # 4.4) Run scoring algorithm
            ascore.score(mz_arr = spectrum["mz_values"],
                        int_arr = spectrum["intensity_values"],
                        peptide = psm["peptide"],
                        n_of_mod = np.sum(mod_select),
                        # max_fragment_charge = psm["charge_state"] - 1,
                        aux_mod_pos = aux_mod_pos,
                        aux_mod_mass = aux_mod_masses)

            # 4.5) Place scores into an object to use later
            pyascore_results.append({"scan" : psm["scan"],
                                    "localized_peptide" : ascore.best_sequence,
                                    "pepscore" : ascore.best_score,
                                    "ascores" : ";".join([str(s) for s in ascore.ascores])})
    pyascore_results = pd.DataFrame.from_records(pyascore_results)

    id_dataframe = pd.read_csv(psm_path, sep = "\t")
    ascore_final_results = pd.merge(id_dataframe, pyascore_results, left_on = "ScanNr", right_on = "scan", how = "left")

    return ascore_final_results
