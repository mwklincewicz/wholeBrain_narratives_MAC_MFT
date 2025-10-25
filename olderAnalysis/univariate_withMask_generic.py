import os

import nilearn.image
import numpy as np
from nilearn import image
import numpy as np
import pandas as pd

alias_data_dir = "/fmriprep"
processed_dir = "G:/fMRI_project/processed_first_level_per_sentence/uni_TPJ/"
mask_dir = "../masks/"

def load_participants(story):
    #return a list of all participant numbers for a story
    participants = []
    story = story + "_"
    for root, dirs, files in os.walk(alias_data_dir):
        for file in files:
            if story in file and file.endswith("space-MNI152NLin2009cAsym_res-native_desc-preproc_bold.nii.gz"):
                participant_number = file[4:7]
                #print("Getting participant number %03s for %04s" % (participant_number, story))
                if story == 'tunnel' and (participant_number == '004' or participant_number != '013'):
                    print( "excluding participant " + participant_number)
                else:
                    participants.append(participant_number)
    return participants

def load_image_files(story):
    # return a list of files
    story = story + "_"
    files = []
    for root, dirs, files in os.walk(alias_data_dir):
        for file in files:
            if story in file and file.endswith("space-MNI152NLin2009cAsym_res-native_desc-preproc_bold.nii.gz"):
                if "run-1" in file:
                    print( "Appending " + file)
                    files.append(file)
                elif "run-2" in file:
                    print( "Ignoring data for run 2")
                else:
                    files.append( file )
    return files

def load_image_data(sub,story):
    # Load image
    story = story + "_"
    for root, dirs, files in os.walk(alias_data_dir):
        for file in files:
            if story in file and file.endswith("space-MNI152NLin2009cAsym_res-native_desc-preproc_bold.nii.gz") and "sub-"+str(sub) in file:
                if "run-1" in file:
                    img_path = os.path.join(alias_data_dir,"sub-%03s/func/sub-%03s_task-%srun-1_space-MNI152NLin2009cAsym_res-native_desc-preproc_bold.nii.gz" % (sub, sub, story))
                elif "run-2" in file:
                    print( "Ignoring data for run 2")
                else:
                    img_path = os.path.join(alias_data_dir,"sub-%03s/func/sub-%03s_task-%sspace-MNI152NLin2009cAsym_res-native_desc-preproc_bold.nii.gz" % (sub, sub, story))
                print("Loading data from %s" % (img_path))
                img = image.load_img(img_path)
    return img, img_path

def load_regressor(sub,story):
    # Load tsv file with regressors
    story = story + "_"

    for root, dirs, files in os.walk(alias_data_dir):
        for file in files:
            if story in file and file.endswith("desc-confounds_regressors.tsv") and "sub-"+sub in file:
                if "run-1" in file:
                    regressor_location = os.path.join(alias_data_dir,"sub-%03s/func/sub-%03s_task-%04srun-1_desc-confounds_regressors.tsv" % (sub, sub, story))
                elif "run-2" in file:
                    print( "Ignoring regressors for run 2")
                else:
                    regressor_location = os.path.join(alias_data_dir,"sub-%03s/func/sub-%03s_task-%04sdesc-confounds_regressors.tsv" % (sub, sub, story))

                print("Loading regressors from %s" % (regressor_location))
                regressor = pd.read_csv(regressor_location, sep='\t')
    return regressor, regressor_location

def run(story, mask):
    # ----------------------------
    # Load prep data structures
    # ----------------------------
    participants = load_participants(story)
    n_participants = len(participants)
    mask_orig = image.load_img(mask_dir + mask)

    # ----------------------------
    # Reference image (for resampling)
    # ----------------------------
    ref_img = load_image_data(participants[0], story)[0]

    # Resample masks to match the first image’s space
    mask_res = image.resample_to_img(mask_orig, ref_img.slicer[..., 0], interpolation='nearest')

    mask_data = mask_res.get_fdata().astype(bool)

    n_voxels = np.sum(mask_data)
    n_timepoints = ref_img.shape[-1]
    print(f"Each participant has {n_timepoints} timepoints")

    # ----------------------------
    # Preallocate arrays (voxels × time × participants)
    # ----------------------------
    array_3d = np.zeros((n_voxels, n_timepoints, n_participants))

    # ----------------------------
    # Loop through participants
    # ----------------------------
    i = 0
    for participant in participants:
        # Load image and confounds
        img = load_image_data(participant, story)[0]
        confounds_df = load_regressor(participant, story)[0]

        # Select confounds
        confound_cols = [c for c in confounds_df.columns if any(x in c for x in [
            "trans_x", "trans_y", "trans_z",
            "rot_x", "rot_y", "rot_z"
        ])]
        confounds = confounds_df[confound_cols].fillna(0).values

        # Clean the image (regress confounds, detrend, standardize)
        clean_data = image.clean_img(
            img,
            confounds=confounds,
            detrend=True,
            standardize=True
        )

        clean_img_data = clean_data.get_fdata()

        # Resample masks to participant space (if needed)
        mask_res = image.resample_to_img(mask_orig, clean_data.slicer[..., 0], interpolation='nearest')

        mask_data = mask_res.get_fdata().astype(bool)

        # Extract voxel time series (voxels × time)
        ts = clean_img_data[mask_data, :]

        # Store in 3D array
        array_3d[:, :, i] = ts
        i += 1

    # ----------------------------
    # Save outputs
    # ----------------------------
    os.makedirs(processed_dir + '/' + story + '/', mode=0o777,exist_ok=True)  # this checks if the directory exists and creates it, if not
    np.save(processed_dir + '/' + story + "/" + mask.split('.')[0] + "_3D_clean.npy", array_3d)

    print("Mask array shape:", array_3d.shape, "(voxels × time × subjects)")
