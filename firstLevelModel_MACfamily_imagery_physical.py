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
import datalad
from subprocess import call


#
# USE MAC FOUNDATION FAMILY IMAGERY WORD PEAKS AS TRIALS IN PHYSICAL VERSION OF THE NARRATIVE
#

#
#   Constants
#

testSubject = [221]
# Define subjects
participants = [49, 58, 95, 115, 127, 181, 186, 190, 191, 200, 201] + list(range(206, 238)) + list(range(239, 254))
#exclude subj 238, see paper

onsets = [44.7,62.6,76.4,110,145.4,154.6,182.7,199,213.8,232,243,263,281.2,302.5,312,331.7,363.7,376.6,390,406,421,438.5]
durations = [17.9,13.8,33.6,35.4,9.2,28.1,16.3,14.8,18.2,11,20,18.2,21.3,9.5,19.7,32,12.9,13.4,16,15,17.5,14.5]
eventNames = ["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22"]
wordPeaks = ["3","4","13","22"]

epidataTest_dir = "./testData/fmri"
confoundsTest_dir = "./testData/confounds/"
processed_Testdir = "./testData/processed_first_level_MAC_family/"

epidata_dir = "G:/fMRI_project/narrative_mri/data/"
confounds_dir = "G:/fMRI_project/narrative_mri/confounds/"
processed_dir = "G:/fMRI_project/processed_first_level_MAC_family/"

alias_dir = "C:\\Users\\micha\\PycharmProjects\\wholeBrain_narrative_MAC_MFT\\allDataAliases\\fmriprep"

segmentFileDF_social = pd.read_excel("./foundationScores/shapessocial_transcript_segment_MFT_MAC.xlsx")

#filter to keep all values in the MAC Family Virtue column for segments that have more than 12 'imagery' words
segmentFileDF_social['wordPeaks'] = segmentFileDF_social['segment'].apply(lambda x: x if x in wordPeaks else "0")

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
                               'seg_MAC_a_property_vice']]

eventFoundations_social = segmentValues_social.drop_duplicates(subset=['segment'], keep='first', ignore_index=True)

#
#   Helper functions
#

def load_epi_data_physical(sub):
    # Load MRI file (in Nifti format)
    epi_in = os.path.join(epidata_dir,"sub-%03d_task-shapesphysical_space-MNI152NLin2009cAsym_res-native_desc-preproc_bold.nii" % (sub))
    epi_data_physical = nib.load(epi_in)
    print("Loading data from %s" % (epi_in))
    return epi_data_physical

def load_epi_data_physical_alias(sub):
    # Load MRI file (in Nifti format)
    epi_in = os.path.join(alias_dir+"/sub-%03d/func/" %(sub),"sub-%03d_task-shapesphysical_space-MNI152NLin2009cAsym_res-native_desc-preproc_bold.nii.gz" % (sub))
    epi_data_physical = nib.load(epi_in)
    print("Loading data from %s" % (epi_in))
    return epi_data_physical

#
#   Data transform
#

for participant in testSubject:
    print ("Building first-level model for participant %s" % (participant))
    epi_data_physicalNIFTI = load_epi_data_physical_alias(participant)

    events = pd.DataFrame({"trial_type": sorted([int(x) for x in eventNames]), "onset": onsets, "duration": durations})

    fname2 = "sub-%03d_task-shapesphysical_desc-confounds_regressors.tsv" % participant
    confoundsAll_physical = confounds_dir + fname2

    df = pd.read_csv(confoundsAll_physical, sep='\t')
    confound_file1 = df[['csf', 'white_matter', 'trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']].to_numpy()

    # use segment column for determining modulation values
    modulationValues = eventFoundations_social['segment']

    ## Keep the trials with sufficient 'imagination' words as trials [listed in wordPeaks]
    for Trial in range(len(modulationValues)):
       #print('Processing Trial %s' % (Trial))

        if Trial in wordPeaks:
            events = events.drop(Trial)


    events['trial_type'] = 'wordPeak'

    # Make an average
    mean_img = image.mean_img(epi_data_physicalNIFTI, copy_header=True)

    mask = masking.compute_epi_mask(mean_img, lower_cutoff=0.2, upper_cutoff=0.85, opening=3, connected=True)

    # Clean and smooth data
    epi_data_physicalNIFTI = image.clean_img(epi_data_physicalNIFTI, standardize=False)
    epi_data_physicalNIFTI = image.smooth_img(epi_data_physicalNIFTI, 6.0)

    # get fdata
    epi_data_physical = epi_data_physicalNIFTI.get_fdata()

    frame_times = (
            np.arange(epi_data_physical.shape[3]) * 1.5
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
    FM1 = FM1.fit(epi_data_physicalNIFTI, design_matrices=X_base)

    # contrast first level model

    # Let's compare it to the unmodulated block design
    fig, (ax1) = plt.subplots(
        figsize=(10, 6), nrows=1, ncols=1, constrained_layout=True
    )

    plot_design_matrix(X_base, axes=ax1)
    ax1.set_title("Block design matrix", fontsize=12)
    #plt.show()

    ## create contrast image
    contrast_name = "wordPeak"

    z_map = FM1.compute_contrast(
        contrast_name,
        output_type="z_score"  # Can be ‘z_score’, ‘stat’, ‘p_value’, ‘effect_size’, ‘effect_variance’ or ‘all’
    )

    # Apply mask to z_map
    masked_data = apply_mask(z_map, mask)
    z_map_masked = unmask(masked_data, mask)

    # save contrast image for the testsubject (to be used at second level)
    z_map_masked.to_filename((processed_Testdir + "%03d_physical_macFamily_wordPeak.nii.gz") % (participant))


    plotting.plot_stat_map(z_map_masked, bg_img=mean_img, title="Masked z-map")
