import nilearn as nl
import numpy as np
import matplotlib
import pandas as pd
import os
import nibabel as nib
import matplotlib.pyplot as plt


from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.plotting import plot_design_matrix
from nilearn.glm.first_level import FirstLevelModel

epidata_dir = "./testData/fmri"
confounds_dir = "./testData/confounds/"

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

def load_foundation_scores(eventName):
    segmentScore = ""
    return segmentScore

testSubject = 221

onsets = [44.7,62.6,76.4,110,145.4,154.6,182.7,199,213.8,232,243,263,281.2,302.5,312,331.7,363.7,376.6,390,406,421,438.5]
durations = [17.9,13.8,33.6,35.4,9.2,28.1,16.3,14.8,18.2,11,20,18.2,21.3,9.5,19.7,32,12.9,13.4,16,15,17.5,14.5]
eventNames = ["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22"]


epi_data_socialNIFTI = load_epi_data_social(testSubject)
epi_data_social = epi_data_socialNIFTI.get_fdata()

new_image = nib.Nifti1Image(epi_data_social,affine=epi_data_socialNIFTI.affine)

frame_times = (
    np.arange(epi_data_social.shape[3] ) * 1.5
)

events = pd.DataFrame( {"trial_type":sorted([int(x) for x in eventNames]),"onset":onsets,"duration":durations})
fname1 = "sub-%03d_task-shapessocial_desc-confounds_regressors.tsv" % testSubject
confoundsAll = confounds_dir + fname1

df1 = pd.read_csv(confoundsAll, sep='\t')
confound_file1 = df1[['csf', 'white_matter', 'trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']].to_numpy()

hrf_model = "glover"
X1 = make_first_level_design_matrix(
    frame_times,
    events,
    add_regs=confound_file1,
    add_reg_names=['csf', 'white_matter', 'trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z'],
    hrf_model=hrf_model,
)

plot_design_matrix(X1)
plt.show()

FM1 = FirstLevelModel()
FM1 = FM1.fit(new_image, design_matrices=X1)

