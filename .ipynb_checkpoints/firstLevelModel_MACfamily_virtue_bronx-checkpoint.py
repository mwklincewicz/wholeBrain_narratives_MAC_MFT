import numpy as np
import pandas as pd
import os
import nibabel as nib
import matplotlib.pyplot as plt

from nilearn import image, masking
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.plotting import plot_design_matrix
from nilearn.glm.first_level import FirstLevelModel
from nilearn.masking import compute_epi_mask, apply_mask, unmask
from nilearn import plotting
from subprocess import call

#
# USE MAC FOUNDATION FAMILY VIRTUE PEAKS AS TRIALS IN SOCIAL VERSION OF THE NARRATIVE
#

#
#   Constants
#
story = "bronx"

testSubject = [315]
# Define subjects
participants = [49, 58, 95, 115, 127, 181, 186, 190, 191, 200, 201] + list(range(206, 238)) + list(range(239, 254))
#exclude subj 238, see paper

onsets = [15,37,49,71,91,103,117,145,180,199,226,261,275,297,318,333,356,388,416,430,475,517]
durations = [22,12,22,20,12,14,28,35,19,27,35,14,22,21,15,23,32,28,14,45,42,35]
eventNames = ["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22"]

epidataTest_dir = "./testData/fmri"
confoundsTest_dir = "./testData/confounds/"
processed_Testdir = "./testData/processed_first_level_MAC_family/"

epidata_dir = "G:/fMRI_project/narrative_mri/data/"
confounds_dir = "G:/fMRI_project/narrative_mri/confounds/"
processed_dir = "G:/fMRI_project/processed_first_level_MAC_family/"

alias_data_dir = "C:\\Users\\micha\\PycharmProjects\\wholeBrain_narrative_MAC_MFT\\allDataAliases\\fmriprep"
alias_confounds_dir = ""
segmentFileDF_social = pd.read_excel("./foundationScores/bronx_transcript_segment_MFT_MAC.xlsx")

#filter to keep all values above .2 in the MAC Family Virtue column and change the rest to .000001
segmentFileDF_social['familyMAC_filterAbovePointOneNine']  = segmentFileDF_social['seg_MAC_a_family_virtue'].apply(lambda x: x if x >= .19 else 0.000001)

segmentValues_social = segmentFileDF_social[['segment',
                               'seg_MFT_a_care_virtue',
                               'seg_MFT_a_fairness_virtue',
                               'seg_MFT_a_loyalty_virtue',
                               'seg_MFT_a_authority_virtue',
                               'seg_MFT_a_sanctity_virtue',
                               'seg_MFT_a_care_vice',
                               'seg_MFT_a_fairness_vice',
                               'seg_MFT_a_loyalty_vice',
                               'seg_MFT_a_authority_vice',
                               'seg_MFT_a_sanctity_vice',
                               'seg_MAC_a_fairness_virtue',
                               'seg_MAC_a_group_virtue',
                               'seg_MAC_a_deference_virtue',
                               'seg_MAC_a_heroism_virtue',
                               'seg_MAC_a_reciprocity_virtue',
                               'seg_MAC_a_family_virtue',
                               'seg_MAC_a_property_virtue',
                               'seg_MAC_a_fairness_vice',
                               'seg_MAC_a_group_vice',
                               'seg_MAC_a_deference_vice',
                               'seg_MAC_a_heroism_vice',
                               'seg_MAC_a_reciprocity_vice',
                               'seg_MAC_a_family_vice',
                               'seg_MAC_a_property_vice',
                               'familyMAC_filterAbovePointOneNine']]

eventFoundations_social = segmentValues_social.drop_duplicates(subset=['segment'], keep='first', ignore_index=True)



#
#   Helper functions
#

def load_participants(story):
    #return a list of all participant numbers for a story
    participants = []
    for root, dirs, files in os.walk(alias_data_dir):
        for file in files:
            if story in file and file.endswith("space-MNI152NLin2009cAsym_res-native_desc-preproc_bold.nii.gz"):
                participant_number = file[4:7]
                print("Getting participant number %03s for %04s" % (participant_number, story))
                participants.append(participant_number)
    return participants

def load_epi_data(sub,story):
    # Load MRI file (in Nifti format)
    for root, dirs, files in os.walk(alias_data_dir):
        for file in files:
            if story in file and file.endswith("space-MNI152NLin2009cAsym_res-native_desc-preproc_bold.nii.gz") and "sub-"+str(sub) in file:
                epi_in = os.path.join(alias_data_dir,"sub-%03s/func/sub-%03s_task-%s_space-MNI152NLin2009cAsym_res-native_desc-preproc_bold.nii.gz" % (sub, sub, story))
                epi_data = nib.load(epi_in)
                print("Loading data from %s" % (epi_in))
    return epi_data

def load_regressor(sub,story):
    # Load tsv file with regressors
    for root, dirs, files in os.walk(alias_data_dir):
        for file in files:
            if story in file and file.endswith("desc-confounds_regressors.tsv") and "sub-"+sub in file:
                regressor_location = os.path.join(alias_data_dir,"sub-%03s/func/sub-%03s_task-%04s_desc-confounds_regressors.tsv" % (sub, sub, story))
                print("Loading regressors from %s" % (regressor_location))
                regressor = pd.read_csv(regressor_location, sep='\t')
    return regressor

#
#   Data transform
#

for participant in load_participants("bronx"):
    print ("Building first-level model for participant %s" % (participant))
    epi_data_socialNIFTI = load_epi_data(participant, story)
    events = pd.DataFrame({"trial_type": sorted([int(x) for x in eventNames]), "onset": onsets, "duration": durations})
    df = load_regressor(participant, story)
    confound_file1 = df[['csf', 'white_matter', 'trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']].to_numpy()

    # use only events from above .19 MAC family values
    modulationValues = eventFoundations_social['familyMAC_filterAbovePointOneNine']

    ## only keep the rows with modulationValues above .2
    for Trial in range(len(modulationValues)):

        #print('trial:', Trial)

        if modulationValues[Trial] < .19:
           events = events.drop(Trial)

    events['trial_type'] = 'macFamily'

    # Make an average
    mean_img = image.mean_img(epi_data_socialNIFTI, copy_header=True)

    mask = masking.compute_epi_mask(mean_img, lower_cutoff=0.2, upper_cutoff=0.85, opening=3, connected=True)

    # Clean and smooth data
    epi_data_socialNIFTI = image.clean_img(epi_data_socialNIFTI, standardize=False)
    epi_data_socialNIFTI = image.smooth_img(epi_data_socialNIFTI, 6.0)

    # get fdata
    epi_data_social = epi_data_socialNIFTI.get_fdata()

    frame_times = (
            np.arange(epi_data_social.shape[3]) * 1.5
    )

    # baseline first level model

    X_base = make_first_level_design_matrix(
        frame_times,
        events,
        add_regs=confound_file1,
        add_reg_names=['csf', 'white_matter', 'trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z'],
        hrf_model='glover',
    )

    FM1 = FirstLevelModel(mask_img=mask)
    FM1 = FM1.fit(epi_data_socialNIFTI, design_matrices=X_base)

    # contrast first level model

    # Let's compare it to the unmodulated block design
    fig, (ax1) = plt.subplots(
        figsize=(10, 6), nrows=1, ncols=1, constrained_layout=True
    )

    plot_design_matrix(X_base, axes=ax1)
    ax1.set_title("Block design matrix", fontsize=12)
    #plt.show()

    ## create contrast image
    contrast_name = "macFamily"

    z_map = FM1.compute_contrast(
        contrast_name,
        output_type="z_score"  # Can be ‘z_score’, ‘stat’, ‘p_value’, ‘effect_size’, ‘effect_variance’ or ‘all’
    )

    # Apply mask to z_map
    masked_data = apply_mask(z_map, mask)
    z_map_masked = unmask(masked_data, mask)

    # save contrast image for the testsubject (to be used at second level)
    z_map_masked.to_filename((processed_dir + "%03s_"+story+"_macFamily.nii.gz") % (participant))

    plotting.plot_stat_map(z_map_masked, bg_img=mean_img, title="Masked z-map")
