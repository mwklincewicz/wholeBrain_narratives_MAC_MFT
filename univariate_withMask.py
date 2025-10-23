import os

import nilearn.image
import numpy as np
from nilearn import image
import numpy as np
import pandas as pd

left_mask_path = "./masks/TPJ_refined_mask_L.nii.gz"
right_mask_path = "./masks/TPJ_refined_mask_R.nii.gz"

alias_data_dir = "C:\\Users\\micha\\PycharmProjects\\wholeBrain_narrative_MAC_MFT\\fmriprep"
processed_dir = "G:/fMRI_project/processed_first_level_per_sentence/uni_TPJ/"

def load_participants(story):
    #return a list of all participant numbers for a story
    participants = []
    story = story + "_"
    for root, dirs, files in os.walk(alias_data_dir):
        for file in files:
            if story in file and file.endswith("space-MNI152NLin2009cAsym_res-native_desc-preproc_bold.nii.gz"):
                participant_number = file[4:7]
                #print("Getting participant number %03s for %04s" % (participant_number, story))
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

# ----------------------------
# Load prep data structures
# ----------------------------
participants = load_participants('21styear')
n_participants = len(participants)
left_mask_orig = image.load_img(left_mask_path)
right_mask_orig = image.load_img(right_mask_path)

# ----------------------------
# Reference image (for resampling)
# ----------------------------
ref_img = load_image_data(participants[0], '21styear')[0]

# Resample masks to match the first image’s space
left_mask_res = image.resample_to_img(left_mask_orig, ref_img.slicer[..., 0], interpolation='nearest')
right_mask_res = image.resample_to_img(right_mask_orig, ref_img.slicer[..., 0], interpolation='nearest')

left_mask_data = left_mask_res.get_fdata().astype(bool)
right_mask_data = right_mask_res.get_fdata().astype(bool)

n_left_voxels = np.sum(left_mask_data)
n_right_voxels = np.sum(right_mask_data)
n_timepoints = ref_img.shape[-1]
print(f"Each participant has {n_timepoints} timepoints")
print(f"Left TPJ voxels: {n_left_voxels}, Right TPJ voxels: {n_right_voxels}")

# ----------------------------
# Preallocate arrays (voxels × time × participants)
# ----------------------------
left_3d = np.zeros((n_left_voxels, n_timepoints, n_participants))
right_3d = np.zeros((n_right_voxels, n_timepoints, n_participants))

# ----------------------------
# Loop through participants
# ----------------------------
for participant in participants:
    # Load image and confounds
    img = load_image_data(participant, '21styear')[0]
    confounds_df = load_regressor(participant, '21styear')[0]

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
    left_mask_res = image.resample_to_img(left_mask_orig, clean_data.slicer[..., 0], interpolation='nearest')
    right_mask_res = image.resample_to_img(right_mask_orig, clean_data.slicer[..., 0], interpolation='nearest')

    left_mask_data = left_mask_res.get_fdata().astype(bool)
    right_mask_data = right_mask_res.get_fdata().astype(bool)

    # Extract voxel time series (voxels × time)
    left_ts = clean_img_data[left_mask_data, :]
    right_ts = clean_img_data[right_mask_data, :]

    # Store in 3D array
    left_3d[:, :, i] = left_ts
    right_3d[:, :, i] = right_ts

# ----------------------------
# Save outputs
# ----------------------------
os.makedirs(processed_dir + '21styear/', mode=0o777,exist_ok=True)  # this checks if the directory exists and creates it, if not
np.save(processed_dir + "21styear/left_TPJ_3D_clean.npy", left_3d)
np.save(processed_dir + "21styear/right_TPJ_3D_clean.npy", right_3d)

print("Left TPJ array shape:", left_3d.shape, "(voxels × time × subjects)")
print("Right TPJ array shape:", right_3d.shape, "(voxels × time × subjects)")
