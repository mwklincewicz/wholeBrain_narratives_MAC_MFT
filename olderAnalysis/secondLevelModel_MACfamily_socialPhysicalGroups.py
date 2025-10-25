import numpy as np
import pandas as pd
import os
from nilearn.glm.second_level import SecondLevelModel
from nilearn.glm import threshold_stats_img

# ## SECOND LEVEL MODEL COMPARING ACROSS GROUPS
# # Second Level Model PHYSICAL STORY
contrastImg_physical_dir = "_results/processed_first_level_MAC_family/physical/"
contrastImg_physical_Testdir = "./_testData/processed_first_level_MAC_family/shapesphysical/"
contrastImg_dir = "_results/processed_first_level_MAC_family/social/"
contrastImg_Testdir = "./_testData/processed_first_level_MAC_family/shapessocial/"

processed_dir = "_results/processed_second_level_MAC_family/"

all_imgs = [
    os.path.join(contrastImg_dir, name)
    for name in os.listdir(contrastImg_dir)
]

second_level_input = all_imgs

all_imgs_physical = [
    os.path.join(contrastImg_physical_dir, name)
    for name in os.listdir(contrastImg_physical_dir)
]

#print(all_imgs_physical)

second_level_input_physical = all_imgs_physical
# load group number of participants
GroupIndex = pd.read_csv("../text/Narratives participants.csv")

# split
GroupIndex[['GroupNumber', 'TaskOrder']] = GroupIndex['Notes'].str.split(',', n=1, expand=True)

GroupIndex['GroupNumber'] = GroupIndex['GroupNumber'].str.strip()
GroupIndex['TaskOrder'] = GroupIndex['TaskOrder'].str.strip()

# create a design matrix for one sample t test, to be used as input for the second level model
design_matrix = pd.DataFrame(
    [1] * len(second_level_input),
    columns=["intercept"],
)

design_matrix
design_matrix = pd.get_dummies(GroupIndex[['GroupNumber']])
design_matrix["intercept"] = 1

design_matrix = design_matrix.astype(int)
contrast_matrix = np.eye(len(design_matrix.columns))[:3]
contrast_matrix
second_level_model = SecondLevelModel().fit(
    second_level_input=second_level_input_physical,
    design_matrix=design_matrix
)

z_map_ftest_physical_groupDifferences = second_level_model.compute_contrast(
    contrast_matrix,
    output_type="z_score"
)

z_map_ftest_physical_groupDifferences.to_filename(processed_dir + "SecondLevel_ftest_physical_groupDifferences_zscore_macFamily.nii.gz")

thresholded_map_Physical_groupDiff, threshold_Physical_groupDiff = threshold_stats_img(
    stat_img=z_map_ftest_physical_groupDifferences,  # or p_map
    alpha=0.05,
    height_control='fdr',  # or 'bonferroni'
    cluster_threshold=0,   # min cluster size in voxels
    two_sided=True
)
print(f"The p<.05 FDR-corrected threshold is z score of {threshold_Physical_groupDiff:.3g}")

# save as brain image
thresholded_map_Physical_groupDiff.to_filename(processed_dir + "threshold_"+f"{threshold_Physical_groupDiff:.3g}"+"_"+
                                                "SecondLevel_Physical_groupDiff_fdrcorrect_macFamily.nii.gz")

# F test shows general group differences for shapesphysical story, now compute group3 over group1 and 2

contrast_vec = np.array([-0.5, -0.5, 1, 0], dtype=float).ravel()

z_map_group3_over_group1_2 = second_level_model.compute_contrast(
    contrast_vec,
    output_type="z_score"
)

z_map_group3_over_group1_2.to_filename(processed_dir + "SecondLevel_physical_group3_over_group1_2_zscore_macFamily.nii.gz")

thresholded_map_Physical_group3vs1_2, threshold_Physical_group3vs1_2 = threshold_stats_img(
    stat_img=z_map_group3_over_group1_2,  # or p_map
    alpha=0.05,
    height_control='fdr',  # or 'bonferroni'
    cluster_threshold=0,   # min cluster size in voxels
    two_sided=True
)
print(f"The p<.05 FDR-corrected threshold is z score of {threshold_Physical_group3vs1_2:.3g}")

# save as brain image
thresholded_map_Physical_group3vs1_2.to_filename(processed_dir + "threshold_"+f"{threshold_Physical_group3vs1_2:.3g}"+"_"+
                                                "SecondLevel_Physical_group3_over_group1_2_fdrcorrect_macFamily.nii.gz")