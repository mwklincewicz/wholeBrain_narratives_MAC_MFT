# SCRIPT FOR DOWNLOADING STORY fMRI IMAGES THROUGH DATALAD

import os
import subprocess

alias_dir = "C:\\Users\\micha\\PycharmProjects\\wholeBrain_narrative_MAC_MFT\\data\\allDataAliases\\fmriprep"


def run( task ):
    for root, dirs, files in os.walk(alias_dir):
        for file in files:
            if str(task) in file and file.endswith("space-MNI152NLin2009cAsym_res-native_desc-preproc_bold.nii.gz"):
                epi = os.path.join(root,file)
                print("Downloading data for %s" % (epi))
                subprocess.call(["datalad","get", epi], shell=True)
            if str(task) in file and file.endswith("desc-confounds_regressors.tsv"):
                epi = os.path.join(root, file)
                print("Downloading data for %s" % (epi))
                subprocess.call(["datalad", "get", epi], shell=True)