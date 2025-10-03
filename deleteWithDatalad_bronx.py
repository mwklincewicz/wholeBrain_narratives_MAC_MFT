# SCRIPT FOR DOWNLOADING BRONX STORY fMRI IMAGES THROUGH DATALAD

import os
import subprocess
import pathlib

alias_dir = "C:\\Users\\micha\\PycharmProjects\\wholeBrain_narrative_MAC_MFT\\allDataAliases\\fmriprep"

for root, dirs, files in os.walk(alias_dir):
    for file in files:
        if "bronx" in file and file.endswith("space-MNI152NLin2009cAsym_res-native_desc-preproc_bold.nii.gz"):
            epi = os.path.join(root,file)
            print("Removing data file for %s" % (epi))
            subprocess.run(["datalad", "remove", epi], shell=True)
        if "bronx" in file and file.endswith("desc-confounds_regressors.tsv"):
            epi = os.path.join(root, file)
            #exe_path = pathlib.PureWindowsPath(epi).as_posix()
            print("Removing data file for %s" % (epi))
            subprocess.call(["datalad", "remove", epi], shell=True)