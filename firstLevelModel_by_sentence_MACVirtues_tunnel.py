import numpy as np
import pandas as pd
import os
import nibabel as nib
import matplotlib.pyplot as plt
from nilearn import image, masking
from nilearn.glm.first_level import FirstLevelModel
from nilearn.masking import compute_epi_mask, apply_mask, unmask
from nilearn import plotting
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.plotting import plot_design_matrix, plot_contrast_matrix

#
# USE FOUNDATION PER SENTENCE AS TRIALS
#
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
                #print("Getting participant number %03s for %04s" % (participant_number, story))
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
                #print("Loading regressors from %s" % (regressor_location))
                regressor = pd.read_csv(regressor_location, sep='\t')
    return regressor

def get_top_foundation_per_sentence(story, foundations):
    #return a list of tuples (index, foundation name, foundation score, onset, duration)
    sentence_tuples = []
    for root, dirs, files in os.walk(foundationScores_dir):
        for file in files:
            if story in file:
                print("Getting top foundations per sentence in %03s " % (story))
                df = pd.read_excel(os.path.join(root, file))
                for index, row in df.iterrows():
                    if ( row[foundations].max() > 0 ):
                        tuple = row[foundations].idxmax(), row['start'], row['end'] - row['start']
                    else:
                        tuple = 'baseline', row['start'], row['end'] - row['start']
                    sentence_tuples.append(tuple)
    return sentence_tuples

def load_onsets(story):
    #return a list of timestamps for onset
    onsets = []
    for root, dirs, files in os.walk(foundationScores_dir):
        for file in files:
            if story in file:
                print("Getting sentence onsets for %03s " % (story))
                onsets = pd.read_excel(os.path.join(root, file), usecols=['start'] )
    return onsets

def load_durations(story):
    #return a list of timestamps for onset
    durations = []
    for root, dirs, files in os.walk(foundationScores_dir):
        for file in files:
            if story in file:
                print("Getting sentence durations for %03s " % (story))
                df = pd.read_excel(os.path.join(root, file), usecols=['start', 'end'] )
                durations.append( df['end'] - df['start'] )
    return durations
#
#   Constants
#

#the name of the task (story) should be changed at some point for a loop through all of them
story = "tunnel"
#each task (story) should have a different test subject number
testSubject = [1]

alias_data_dir = "C:\\Users\\micha\\PycharmProjects\\wholeBrain_narrative_MAC_MFT\\allDataAliases\\fmriprep"
alias_confounds_dir = ""
processed_dir = "G:/fMRI_project/processed_first_level_per_sentence/"
os.makedirs(processed_dir + story + "\\7_MAC\\", mode=0o777, exist_ok=True)  # this checks if the directory exists and creates it, if not
foundationScores_dir = "./foundationScores/"

# this creates a dataframe with per sentence and per segment scores for all foundations and column names that match them, plus segment file name as first element
segmentFileDF = pd.read_excel("./foundationScores/"+story+"_MFT_MAC.xlsx")
sentenceValues = segmentFileDF[['segment',
                               'MFT_a_care_virtue',
                               'MFT_a_fairness_virtue',
                               'MFT_a_loyalty_virtue',
                               'MFT_a_authority_virtue',
                               'MFT_a_sanctity_virtue',
                               'MFT_a_care_vice',
                               'MFT_a_fairness_vice',
                               'MFT_a_loyalty_vice',
                               'MFT_a_authority_vice',
                               'MFT_a_sanctity_vice',
                               'MAC_a_fairness_virtue',
                               'MAC_a_group_virtue',
                               'MAC_a_deference_virtue',
                               'MAC_a_heroism_virtue',
                               'MAC_a_reciprocity_virtue',
                               'MAC_a_family_virtue',
                               'MAC_a_property_virtue',
                               'MAC_a_fairness_vice',
                               'MAC_a_group_vice',
                               'MAC_a_deference_vice',
                               'MAC_a_heroism_vice',
                               'MAC_a_reciprocity_vice',
                               'MAC_a_family_vice',
                               'MAC_a_property_vice']]

segmentValues = segmentFileDF[['segment',
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

eventFoundations = segmentValues.drop_duplicates(subset=['segment'], keep='first', ignore_index=True)
#sentenceFoundations = sentenceValues.drop_duplicates(subset=['segment'], keep='first', ignore_index=True)

#
#   prepare for modeling and data transform
#

#This creates the events and durations FOR DESIGN MATRIX
eventNames = []
foundations_per_sentence = get_top_foundation_per_sentence(story,
                                                           ['MAC_a_fairness_virtue',
                                                            'MAC_a_group_virtue',
                                                            'MAC_a_deference_virtue',
                                                            'MAC_a_heroism_virtue',
                                                            'MAC_a_reciprocity_virtue',
                                                            'MAC_a_family_virtue',
                                                            'MAC_a_property_virtue'])

events = pd.DataFrame(foundations_per_sentence, columns=["trial_type", "onset", "duration"])

# baseline_count = (events['trial_type'] == 'baseline').sum()
# print('number of baseline: ', baseline_count)
# fai_count = (events['trial_type'] == 'MAC_a_fairness_virtue').sum()
# print('number of fairness virtue: ', fai_count)
# fam = (events['trial_type'] == 'MAC_a_family_virtue').sum()
# print('number of family virtue: ', fam)
# her_count = (events['trial_type'] == 'MAC_a_heroism_virtue').sum()
# print('number of heroism virtue: ', her_count)
# rec_count = (events['trial_type'] == 'MAC_a_reciprocity_virtue').sum()
# print('number of reciprocity virtue: ', rec_count)
# gr_count = (events['trial_type'] == 'MAC_a_group_virtue').sum()
# print('number of group virtue: ', gr_count)
# prop_count = (events['trial_type'] == 'MAC_a_property_virtue').sum()
# print('number of property virtue: ', prop_count)
# def_count = (events['trial_type'] == 'MAC_a_deference_virtue').sum()
# print('number of deference virtue: ', def_count)

#
#   Data transform
#
for participant in load_participants(story):
    print ("Building first-level models for participant %s" % (participant))
    epi_data_NIFTI = load_epi_data(participant, story)
    df = load_regressor(participant, story)
    confound_file1 = df[['csf', 'white_matter', 'trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']].to_numpy()

    # Make an average
    mean_img = image.mean_img(epi_data_NIFTI, copy_header=True)
    mask = masking.compute_epi_mask(mean_img, lower_cutoff=0.2, upper_cutoff=0.85, opening=3, connected=True)

    # Clean and smooth data
    epi_data_NIFTI = image.clean_img(epi_data_NIFTI, standardize=False)
    epi_data_NIFTI = image.smooth_img(epi_data_NIFTI, 6.0)

    # get fdata
    epi_data_social = epi_data_NIFTI.get_fdata()

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
    FM1 = FM1.fit(epi_data_NIFTI, design_matrices=X_base)

    # contrast first level model

    # Let's compare it to the unmodulated block design
    fig, (ax1) = plt.subplots(
        figsize=(10, 6), nrows=1, ncols=1, constrained_layout=True
    )

    plot_design_matrix(X_base, axes=ax1)
    ax1.set_title("Block design matrix", fontsize=12)

    #plt.savefig("design_matrix.jpg", dpi=300, bbox_inches='tight')

    #plt.show()

    ## create contrast image
    ## create contrast image F test (which voxels significant in any of the 7 foundations)
    contrast_val = np.eye(7)
    plot_contrast_matrix(contrast_val, X_base)

    contrast_name = "F_contrast"

    z_map = FM1.compute_contrast(
        contrast_val,
        stat_type='F',
        output_type="z_score"  # Can be ‘z_score’, ‘stat’, ‘p_value’, ‘effect_size’, ‘effect_variance’ or ‘all’
    )

    # Apply mask to z_map
    masked_data = apply_mask(z_map, mask)
    z_map_masked = unmask(masked_data, mask)

    # save contrast image (to be used at second level)
    os.makedirs(processed_dir + story + "\\7_MAC\\F_contrast\\", mode=0o777, exist_ok=True)  # this checks if the directory exists and creates it, if not
    z_map_masked.to_filename(processed_dir+story+"\\7_MAC\\F_contrast\\"+participant+"_"+story+"_F_contrast_7_MAC_perSentence.nii.gz")

    # determine for each moral foundations, where is more activation for that foundation vs an average of the other 6

    # foundation 1
    c1 = np.array([1, -1 / 6, -1 / 6, -1 / 6, -1 / 6, -1 / 6, -1 / 6])  # exact -1/6
    z_map_foundation1 = FM1.compute_contrast(c1, stat_type='t', output_type='z_score')

    # foundation 1
    c2 = np.array([-1 / 6, 1, -1 / 6, -1 / 6, -1 / 6, -1 / 6, -1 / 6])  # exact -1/6
    z_map_foundation2 = FM1.compute_contrast(c2, stat_type='t', output_type='z_score')

    # foundation 1
    c3 = np.array([-1 / 6, -1 / 6, 1, -1 / 6, -1 / 6, -1 / 6, -1 / 6])  # exact -1/6
    z_map_foundation3 = FM1.compute_contrast(c3, stat_type='t', output_type='z_score')

    # foundation 1
    c4 = np.array([-1 / 6, -1 / 6, -1 / 6, 1, -1 / 6, -1 / 6, -1 / 6])  # exact -1/6
    z_map_foundation4 = FM1.compute_contrast(c4, stat_type='t', output_type='z_score')

    # foundation 1
    c5 = np.array([-1 / 6, -1 / 6, -1 / 6, -1 / 6, 1, -1 / 6, -1 / 6])  # exact -1/6
    z_map_foundation5 = FM1.compute_contrast(c5, stat_type='t', output_type='z_score')

    # foundation 1
    c6 = np.array([-1 / 6, -1 / 6, -1 / 6, -1 / 6, -1 / 6, 1, -1 / 6])  # exact -1/6
    z_map_foundation6 = FM1.compute_contrast(c6, stat_type='t', output_type='z_score')

    # foundation 1
    c7 = np.array([-1 / 6, -1 / 6, -1 / 6, -1 / 6, -1 / 6, -1 / 6, 1])  # exact -1/6
    z_map_foundation7 = FM1.compute_contrast(c7, stat_type='t', output_type='z_score')

    os.makedirs(processed_dir + story + "\\7_MAC\\VsOther6\\", mode=0o777, exist_ok=True)  # this checks if the directory exists and creates it, if not
    z_map_foundation1.to_filename(processed_dir+story+"\\7_MAC\\VsOther6\\"+ participant + "_"+story+"_7_MAC_perSentence_z_map_foundation1_vsOther6.nii.gz")
    z_map_foundation2.to_filename(processed_dir+story+"\\7_MAC\\VsOther6\\"+ participant + "_"+story+"_7_MAC_perSentence_z_map_foundation2_vsOther6.nii.gz")
    z_map_foundation3.to_filename(processed_dir+story+"\\7_MAC\\VsOther6\\"+ participant + "_"+story+"_7_MAC_perSentence_z_map_foundation3_vsOther6.nii.gz")
    z_map_foundation4.to_filename(processed_dir+story+"\\7_MAC\\VsOther6\\"+ participant + "_"+story+"_7_MAC_perSentence_z_map_foundation4_vsOther6.nii.gz")
    z_map_foundation5.to_filename(processed_dir+story+"\\7_MAC\\VsOther6\\"+ participant + "_"+story+"_7_MAC_perSentence_z_map_foundation5_vsOther6.nii.gz")
    z_map_foundation6.to_filename(processed_dir+story+"\\7_MAC\\VsOther6\\"+ participant + "_"+story+"_7_MAC_perSentence_z_map_foundation6_vsOther6.nii.gz")
    z_map_foundation7.to_filename(processed_dir+story+"\\7_MAC\\VsOther6\\"+ participant + "_"+story+"_7_MAC_perSentence_z_map_foundation7_vsOther6.nii.gz")

    # foundation 1
    c1 = np.array([1, 0, 0, 0, 0, 0, 0, -1])  # exact -1/6
    z_map_foundation1vsbase = FM1.compute_contrast(c1, stat_type='t', output_type='z_score')

    # foundation 1
    c2 = np.array([0, 1, 0, 0, 0, 0, 0, -1])  # exact -1/6
    z_map_foundation2vsbase = FM1.compute_contrast(c2, stat_type='t', output_type='z_score')

    # foundation 1
    c3 = np.array([0, 0, 1, 0, 0, 0, 0, -1])  # exact -1/6
    z_map_foundation3vsbase = FM1.compute_contrast(c3, stat_type='t', output_type='z_score')

    # foundation 1
    c4 = np.array([0, 0, 0, 1, 0, 0, 0, -1])  # exact -1/6
    z_map_foundation4vsbase = FM1.compute_contrast(c4, stat_type='t', output_type='z_score')

    # foundation 1
    c5 = np.array([0, 0, 0, 0, 1, 0, 0, -1])  # exact -1/6
    z_map_foundation5vsbase = FM1.compute_contrast(c5, stat_type='t', output_type='z_score')

    # foundation 1
    c6 = np.array([0, 0, 0, 0, 0, 1, 0, -1])  # exact -1/6
    z_map_foundation6vsbase = FM1.compute_contrast(c6, stat_type='t', output_type='z_score')

    # foundation 1
    c7 = np.array([0, 0, 0, 0, 0, 0, 1, -1])  # exact -1/6
    z_map_foundation7vsbase = FM1.compute_contrast(c7, stat_type='t', output_type='z_score')

    os.makedirs(processed_dir + story + "\\7_MAC\\VsBaseline\\", mode=0o777, exist_ok=True)  # this checks if the directory exists and creates it, if not
    z_map_foundation1vsbase.to_filename(processed_dir+story+"\\7_MAC\\VsBaseline\\"+ participant + "_"+story+"_7_MAC_perSentence_z_map_foundation1_vsBaseline.nii.gz")
    z_map_foundation2vsbase.to_filename(processed_dir+story+"\\7_MAC\\VsBaseline\\"+ participant + "_"+story+"_7_MAC_perSentence_z_map_foundation2_vsBaseline.nii.gz")
    z_map_foundation3vsbase.to_filename(processed_dir+story+"\\7_MAC\\VsBaseline\\"+ participant + "_"+story+"_7_MAC_perSentence_z_map_foundation3_vsBaseline.nii.gz")
    z_map_foundation4vsbase.to_filename(processed_dir+story+"\\7_MAC\\VsBaseline\\"+ participant + "_"+story+"_7_MAC_perSentence_z_map_foundation4_vsBaseline.nii.gz")
    z_map_foundation5vsbase.to_filename(processed_dir+story+"\\7_MAC\\VsBaseline\\"+ participant + "_"+story+"_7_MAC_perSentence_z_map_foundation5_vsBaseline.nii.gz")
    z_map_foundation6vsbase.to_filename(processed_dir+story+"\\7_MAC\\VsBaseline\\"+ participant + "_"+story+"_7_MAC_perSentence_z_map_foundation6_vsBaseline.nii.gz")
    z_map_foundation7vsbase.to_filename(processed_dir+story+"\\7_MAC\\VsBaseline\\"+ participant + "_"+story+"_7_MAC_perSentence_z_map_foundation7_vsBaseline.nii.gz")

    # based on t maps rather than thresholded maps, minimum statistic conjunction
    # based on:
    # Thomas Nichols, Matthew Brett, Jesper Andersson, Tor Wager, Jean-Baptiste Poline,
    # Valid conjunction inference with the minimum statistic, NeuroImage, Volume 25, Issue 3, 2005

    min_stat_map = image.math_img(
        "np.minimum.reduce((img1, img2, img3, img4, img5, img6, img7))",
        img1=z_map_foundation1vsbase,
        img2=z_map_foundation2vsbase,
        img3=z_map_foundation3vsbase,
        img4=z_map_foundation4vsbase,
        img5=z_map_foundation5vsbase,
        img6=z_map_foundation6vsbase,
        img7=z_map_foundation7vsbase,
    )
    os.makedirs(processed_dir + story + "\\7_MAC\\Conjunction\\", mode=0o777, exist_ok=True)  # this checks if the directory exists and creates it, if not
    min_stat_map.to_filename(processed_dir+story+"\\7_MAC\\Conjunction\\"+ participant + "_"+story+"_7_MAC_perSentence_minimum_stat_conjunction.nii.gz")


    plotting.plot_stat_map(z_map_masked, bg_img=mean_img, title="Masked z-map")
