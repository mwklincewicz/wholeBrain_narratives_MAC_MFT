import nilearn as nl
import numpy as np
import matplotlib
import pandas as pd
import os
import nibabel as nib
import matplotlib.pyplot as plt
import openpyxl as xl

from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.plotting import plot_design_matrix
from nilearn.glm.first_level import FirstLevelModel
from pandas import read_excel

#
#   Constants
#

epidata_dir = "./testData/fmri"
confounds_dir = "./testData/confounds/"

testSubject = 221
# Define subjects
participants = [49]#, 58, 95, 115, 127, 181, 186, 190, 191, 200, 201] + list(range(206, 238)) + list(range(239, 254))
#exclude subj 238, see paper

onsets = [44.7,62.6,76.4,110,145.4,154.6,182.7,199,213.8,232,243,263,281.2,302.5,312,331.7,363.7,376.6,390,406,421,438.5]
durations = [17.9,13.8,33.6,35.4,9.2,28.1,16.3,14.8,18.2,11,20,18.2,21.3,9.5,19.7,32,12.9,13.4,16,15,17.5,14.5]
eventNames = ["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22"]

segmentFileDF = pd.read_excel("testData/foundationScores/shapessocial_transcript_segment_MFT_MAC.xlsx")
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
eventFoundations = eventFoundations.iloc[:,1:]

#
#   Helper functions
#

def load_epi_data_social(sub):
    # Load MRI file (in Nifti format)
    epi_in = os.path.join(epidata_dir,"sub-%03d_task-shapessocial_space-MNI152NLin2009cAsym_res-native_desc-preproc_bold.nii" % (sub))
    epi_data_social = nib.load(epi_in)
    print("Loading data from %s" % (epi_in))
    return epi_data_social

def load_epi_data_physical(sub):
    # Load MRI file (in Nifti format)
    epi_in = os.path.join(epidata_dir,"sub-%03d_task-shapesphysical_space-MNI152NLin2009cAsym_res-native_desc-preproc_bold.nii" % (sub))
    epi_data_physical = nib.load(epi_in)
    print("Loading data from %s" % (epi_in))
    return epi_data_physical

#
#   Data transform
#

epi_data_socialNIFTI = load_epi_data_social(testSubject)
epi_data_social = epi_data_socialNIFTI.get_fdata()

nifti_image_for_model = nib.Nifti1Image(epi_data_social,affine=epi_data_socialNIFTI.affine)

frame_times = (
    np.arange(epi_data_social.shape[3] ) * 1.5
)

events = pd.DataFrame( {"trial_type":sorted([int(x) for x in eventNames]),"onset":onsets,"duration":durations})
fname1 = "sub-%03d_task-shapessocial_desc-confounds_regressors.tsv" % testSubject
confoundsAll = confounds_dir + fname1

df = pd.read_csv(confoundsAll, sep='\t')
confound_file1 = df[['csf', 'white_matter', 'trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']].to_numpy()

#baseline first level model

X_base = make_first_level_design_matrix(
    frame_times,
    events,
    add_regs=confound_file1,
    add_reg_names=['csf', 'white_matter', 'trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z'],
    hrf_model='glover',
)

FM1 = FirstLevelModel()
FM1 = FM1.fit(nifti_image_for_model, design_matrices=X_base)

#contrast first level model

for modulationFoundations, modulationValues in eventFoundations.items():
    #print(modulationFoundations)
    #print(modulationValues.shape)
    modulated_events = pd.DataFrame(
        {
            "trial_type": sorted([int(x) for x in eventNames]),
            "onset": onsets,
            "duration": durations,
            "modulation": modulationValues,
        }
    )

X_modulated = make_first_level_design_matrix(
    frame_times,
    modulated_events,
    add_regs=confound_file1,
    add_reg_names=['csf', 'white_matter', 'trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z'],
    hrf_model='glover',
)

# Let's compare it to the unmodulated block design
fig, (ax1, ax2) = plt.subplots(
    figsize=(10, 6), nrows=1, ncols=2, constrained_layout=True
)

plot_design_matrix(X_base, axes=ax1)
ax1.set_title("Block design matrix", fontsize=12)
plot_design_matrix(X_modulated, axes=ax2)
ax2.set_title("Modulated block design matrix", fontsize=12)
plt.show()
