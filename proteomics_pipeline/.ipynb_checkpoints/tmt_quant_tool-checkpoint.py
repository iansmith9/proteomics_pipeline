from pyteomics import mzml   
import pandas as pd
#import re
import numpy as np
from pyteomics import mgf
import statistics
from datetime import datetime
from pathlib import Path


### MS3 extraction of relavent values
#,CE_value = 55
def extract_ms3(s):
    scan_dict={}
    if s["ms level"] == 1:
        return None
    if s["ms level"] == 2:
        return None
    if s["ms level"] == 3:
        scan_dict["ScanNum"]=s["index"]+1 
        #scan_dict["RefScan"]=s["master scan number"]
        scan_dict["RefScan"]=int(s["precursorList"]["precursor"][0]["spectrumRef"].split(" ")[2].replace("scan=",""))
        scan_dict["MS3_scan_type"] = s["precursorList"]["precursor"][0]["activation"]["collision energy"]
        val = len(s["precursorList"]["precursor"])
        sps_ions=[]
        if val > 1:
            for i in range(0,val):
                if i == val-1:
                    scan_dict["Precursor m/z"] = s["precursorList"]["precursor"][i]['selectedIonList']['selectedIon'][0]['selected ion m/z']
                else:
                    sps_final = s["precursorList"]["precursor"][i]['selectedIonList']['selectedIon'][0]['selected ion m/z']
                    sps_ions.append(sps_final)
        else:
            sps_ions.append("None") 
        scan_dict["SPS ions"] = sps_ions
        scan_dict["SPS count"] = val-1
        scan_dict["m/zArray"] = s["m/z array"]
        scan_dict["IntensityArray"] = s["intensity array"]
    return scan_dict

### TMT reference dataframe

tmt_list = [126.127726,127.124761,127.131081,128.128116,
           128.134436,129.131471,129.13779,130.134825,
           130.141145,131.13818,131.1445,132.141535,
           132.147855,133.14489,133.15121,134.148245,
           134.154565,135.1516]
tmt_channel = ["TMT1","TMT2","TMT3","TMT4",
                         "TMT5","TMT6","TMT7","TMT8",
                         "TMT9","TMT10","TMT11","TMT12",
                         "TMT13","TMT14","TMT15","TMT16",
                         "TMT17","TMT18"]
tmt_reference_file = pd.DataFrame({'TheoMZ':tmt_list,'TMTchannel':tmt_channel})



# In[256]:
#, CE_value = 55
def get_mzml(mzML_file):
    spectra_reader = mzml.MzML(f'{mzML_file}')
    scan_list = []
    for scan in spectra_reader:
        scan_list.append(extract_ms3(scan))
    scan_list = [s for s in scan_list if s is not None]
    data_ms3 = pd.DataFrame.from_records(scan_list)
    return(data_ms3)


def reporter_quant(data_ms3, Da_tol = 0.003):
    MS3_fig1=data_ms3
    summary_list=[]
    for i in range(0,len(MS3_fig1)):
        if i % 5000 == 0:
            print("Reporter Quant on",i, "out of",MS3_fig1.shape[0],"MS3 scans" )
        elif i == MS3_fig1.shape[0]-1:
            print("Reporter Quant on",i+1, "out of",MS3_fig1.shape[0],"MS3 scans" )

        example=MS3_fig1.iloc[i,:]
        mz = example['m/zArray']
        intensity =example['IntensityArray']
        
        tmt_reference_file["MS3_scan_type"] = example["MS3_scan_type"]
        tmt_reference_file["ScanNum"] = example["ScanNum"]
        tmt_reference_file["RefScan"] = example["RefScan"]
        tmt_reference_file["Precursor m/z"] =example["Precursor m/z"]
        tmt_reference_file["SPS ions"] =  ", ".join(str(x) for x in example["SPS ions"])
        tmt_reference_file["SPS count"] = example["SPS count"]
        tmt_reference_file["Noise"] = statistics.median(intensity)
        tmt_reference_file["Noise_min"] = min(intensity)
        index_value = 0 
        ## track back index so only go through observed spectra from where last left off from previous annotated peak
        back_index = 0 
        val_options = []
        list_annotation_indicies = []
        list_intensities = []
        ### List of theoretical masses for peptide annotation
        theo_mass =  tmt_reference_file['TheoMZ'].tolist()
        ### For each theoretical fragment mass look for matching observed peak signal within MS/MS ppm tolerance
        for i in range(0,len(theo_mass)):
            ## theoretical m/z mass of fragment to match to
            value = theo_mass[i]
            ## keep looking for observed features in MS/MS if observed signal has mass less than 25 ppm above the theoretical mass and is not the end of the array of observed m/z's
            while mz[index_value] < value + Da_tol and index_value < len(mz) -1:
                ## if an observed MS/MS peak maps within the 50ppm tolerance of the theoretical fragment mass
                if mz[index_value] > value - Da_tol and mz[index_value] < value + Da_tol:
                    ## append the options for the matched MS/MS observed peak or peaks that can be assigned to that theoretical mass
                    val_options.append(intensity[index_value])
                    ## Increase the index value 
                    index_value += 1
                ## If the observed m/z masses are below the threoretical mass lowest boundary for ppm tolerance (25 ppm below theoretical mass) then move back index up to this index value
                elif mz[index_value] < value - Da_tol:
                    back_index = index_value
                    ## keep moving up the index value until within range
                    index_value += 1
                ## keep moving up index. this enables looking for multiple features until you are outside the 25 ppm tolerance above.
                else: 
                    index_value += 1
            ## If there are multiple observed MS/MS peaks m/z's that fall within the 50 ppm tolerance of the theoretical mass then max intensity feature is choosen
            if len(val_options) >0 :
                ## index location appended 
                list_annotation_indicies.append(i)
                ## add annotation of the theoretical mass with the max intensity of matched MS/MS peak that falls within 50 ppm of the theoretical.
                list_intensities.append(max(val_options))
                ## reset valid options for matched observed to next theoretical mass
                val_options=[]
            ## initiate new back index
            index_value = back_index
        ## reset indexes of dataframe because not all theoretical features will have an observed MS/MS
        df_final = tmt_reference_file.iloc[list_annotation_indicies].reset_index()
        df_final['intensity'] = list_intensities
        summary_list.append(df_final)
        
    ms3_data_quants=pd.concat(summary_list)


    # In[298]:


    ms3_data_quants_drop = ms3_data_quants.drop(['TheoMZ',"index"],axis=1) 
    ms3_data_quants_drop = ms3_data_quants_drop.pivot(index = ["MS3_scan_type","ScanNum","RefScan","Precursor m/z","SPS ions","SPS count","Noise","Noise_min"],
                                                     columns = "TMTchannel",
                                                     values="intensity")
    ms3_data_quants_drop2 = ms3_data_quants_drop.reset_index()
    ms3_data_quants_drop3 = ms3_data_quants_drop2[["ScanNum","RefScan","Precursor m/z","MS3_scan_type","SPS ions","SPS count","Noise","Noise_min","TMT1","TMT2","TMT3","TMT4",
                             "TMT5","TMT6","TMT7","TMT8",
                             "TMT9","TMT10","TMT11","TMT12",
                             "TMT13","TMT14","TMT15","TMT16",
                             "TMT17","TMT18"]]
    ms3_data_quants_drop3 = ms3_data_quants_drop3.sort_values(by=['ScanNum'])
    return(ms3_data_quants_drop3)




###Main function

def run_tmt_quant(mzML_file,Da_tol, out):
    # start_time = datetime.now()
    spectra = get_mzml(mzML_file)
    # print("mzml import")
    print("Start reporter quant")
    reporter_df = reporter_quant(spectra, Da_tol)
    # print("Reporter Ion Quant Complete")
    mzML_file_name = (
        out / f"{mzML_file.stem}_ms3_quant.csv"
    )
    Output_name = str(mzML_file_name)
    # Output_name = mzML_file_string.strip(".mzML") + "_ms3_quant.csv"
    reporter_df.to_csv(Output_name, sep=',', index=False) 
    # print("export success")  
    # end_time = datetime.now()
    # print('Duration: {}'.format(end_time - start_time))