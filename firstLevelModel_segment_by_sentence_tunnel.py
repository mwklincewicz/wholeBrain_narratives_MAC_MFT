import numpy as np
import pandas as pd
import os
import nibabel as nib
import matplotlib.pyplot as plt
from subprocess import call
from nilearn import image, masking
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.plotting import plot_design_matrix
from nilearn.glm.first_level import FirstLevelModel
from nilearn.masking import compute_epi_mask, apply_mask, unmask
from nilearn import plotting


#
# USE FOUNDATION PER SENTENCE PEAKS AS KEYS FOR SEGMENTS TO BE USED AS TRIALS IN THE CONTRAST IMAGE
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
                print("Loading regressors from %s" % (regressor_location))
                regressor = pd.read_csv(regressor_location, sep='\t')
    return regressor

#
#   Constants
#

#the name of the task (story) should be changed at some point for a loop through all of them
story = "tunnel"
#each task (story) should have a different test subject number
testSubject = [1]
# this is the number of segments that will be selected for trials (this should be changed at some point to reflect a dynamic selection, based on SD or something like it)
numberOfTopSegments = 4

alias_data_dir = "C:\\Users\\micha\\PycharmProjects\\wholeBrain_narrative_MAC_MFT\\allDataAliases\\fmriprep"
alias_confounds_dir = ""
processed_dir = "G:/fMRI_project/processed_first_level_per_sentence/"

# this creates a dataframe with per sentence and per segment scores for all foundations and column names that match them, plus segment file name as first element
segmentFileDF = pd.read_excel("./foundationScores/"+story+"_transcript_segment_MFT_MAC.xlsx")
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
onsets = [0,14,39,63,79,95,125,136,164,187,203,219,227,250,258,294,306,323,328,359,367,397,438,460,487,505,527, 543, 568,633,654,688,733,756,769,790,814,824,859,875,901,920,943,963,982,992,1022,1046,1071,1107,1133,1150,1170,1183,1203,1234,1241,1272,1294,1307,1335,1368,1392,1440,1473,1510]
durations = [6,24,23,3,15,23,10,21,22,15,15,6,15,7,25,21,9,4,32,7,28,30,15,26,15,35,15,24,39,20,18,44,14,12,20,22,9,24,15,25,18,22,19,12,9,26,23,24,35,25,16,19,12,15,30,6,30,21,12,27,32,11,47,32,23,8]

for event in range(len(eventFoundations)):
    eventNames.append(str(event+1))
#print( eventNames, onsets, durations, sep='\n' )

# selects and orders values of foundations, removing sentences from the same segment that are not with the highest foundation score
valueWithSegment = []
listOfFoundationsWithValuesAndSegments = []
for column in sentenceValues.columns[1:]:
    #print(column)
    for cell in sentenceValues.iterrows():
        tuple = cell[1]['segment'], cell[1][column]
        valueWithSegment.append(tuple)
    #sort by value the list of tuples created from the segment name and value for foundation (the outer loop keeps track of which column)
    sortedByValues = sorted(valueWithSegment, key=lambda x: x[1], reverse=True)
    #this eliminates tuples from the list that list the same segment, keeping the one with the highest value
    mydict = {}
    for key, val in sortedByValues:
        mydict.setdefault(key, val)
    #sort the dictionary again, by value, turning it into a list
    sortedByValues = sorted(mydict.items(), key=lambda x: x[1], reverse=True)
    listOfFoundationsWithValuesAndSegments.append((column, sortedByValues[:numberOfTopSegments]))
    valueWithSegment = [] #this clears out the list of values so that it can be used again for the next column

#print(*listOfFoundationsWithValuesAndSegments,sep='\n' )

# this creates a helper dictionary with 'foundation name' as key and a list of top N segment numbers as value
topSegments = {}
for foundation, list in listOfFoundationsWithValuesAndSegments:
    segmentNumbers = []
    for tuple in list[:]:
        segmentNumbers.append(tuple[0].split('_')[2].split('.')[0])
    topSegments[foundation] = segmentNumbers
#print(topSegments,sep='\n')

#
#   Data transform
#
for participant in load_participants(story):
    print ("Building first-level models for participant %s" % (participant))
    epi_data_NIFTI_original = load_epi_data(participant, story)
    df = load_regressor(participant, story)
    confound_file1 = df[['csf', 'white_matter', 'trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']].to_numpy()
    for foundationUsedForModel, topSegmentsForFoundation in listOfFoundationsWithValuesAndSegments:
        events = pd.DataFrame({"trial_type": sorted([int(x) for x in eventNames]), "onset": onsets, "duration": durations})
        # now use events from the list of top segments per foundation
        modulationValues = eventFoundations[ "seg_" + foundationUsedForModel ]
        for Trial in range(len(modulationValues) ):
            if str( Trial + 1 ) not in topSegments[foundationUsedForModel]:
                events = events.drop(Trial)
        events['trial_type'] = str(foundationUsedForModel)
        print( events.to_string(index=False) )

        # Make an average
        mean_img = image.mean_img(epi_data_NIFTI_original, copy_header=True, verbose=11)
        mask = masking.compute_epi_mask(mean_img, lower_cutoff=0.2, upper_cutoff=0.85, opening=3, connected=True)

        # Clean and smooth data
        epi_data_NIFTI = image.clean_img(epi_data_NIFTI_original, standardize=False)
        epi_data_NIFTI = image.smooth_img(epi_data_NIFTI, 6.0)

        # get fdata
        epi_data = epi_data_NIFTI.get_fdata()
        frame_times = (
                np.arange(epi_data.shape[3]) * 1.5
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

        plot_design_matrix(X_base)
        ax1.set_title("Block design matrix", fontsize=12)
        #plt.show()

        ## create contrast image
        contrast_name = str(foundationUsedForModel)

        z_map = FM1.compute_contrast(
            contrast_name,
            output_type="z_score"  # Can be ‘z_score’, ‘stat’, ‘p_value’, ‘effect_size’, ‘effect_variance’ or ‘all’
        )

        # Apply mask to z_map
        masked_data = apply_mask(z_map, mask)
        z_map_masked = unmask(masked_data, mask)

        # save contrast image for the participant (to be used at second level)
        os.makedirs(processed_dir+story+"\\"+foundationUsedForModel, mode=0o777, exist_ok=True)
        z_map_masked.to_filename((processed_dir+story+"\\"+foundationUsedForModel + "\\"+ "%03s_"+story+"_"+str(foundationUsedForModel)+"_perSentence.nii.gz") % (participant))

        plotting.plot_stat_map(z_map_masked, bg_img=mean_img, title="Masked z-map")
