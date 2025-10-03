# SCRIPT FOR DOWNLOADING BRONX STORY fMRI IMAGES THROUGH DATALAD

import os
from subprocess import run

alias_dir = "C:\\Users\\micha\\PycharmProjects\\wholeBrain_narrative_MAC_MFT\\allDataAliases\\fmriprep"

for root, dirs, files in os.walk(alias_dir):
    for file in files:
        if "bronx" in file and file.endswith("space-MNI152NLin2009cAsym_res-native_desc-preproc_bold.nii.gz"):
            epi = os.path.join(root,file)
            print("Downloading data for %s" % (epi))
            run(["datalad", "get", epi])
        if "bronx" in file and file.endswith("desc-confounds_regressors.tsv"):
            epi = os.path.join(root, file)
            print("Downloading data for %s" % (epi))
            run(["datalad", "get", epi])