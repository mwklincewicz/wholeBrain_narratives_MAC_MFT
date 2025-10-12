import pandas as pd
import os
from nilearn.glm.second_level import SecondLevelModel
from nilearn.glm import threshold_stats_img

# # Second level model WORDPEAK SOCIAL
contrastImg_wordpeak_social_dir = "results/processed_first_level_MAC_family/wordpeak_social/"
contrastImg_wordpeak_social_Testdir = "./testData/processed_first_level_MAC_family/wordpeak_social/"
processed_dir = "results/processed_second_level_MAC_family/"

all_imgs_wordpeak_social = [
    os.path.join(contrastImg_wordpeak_social_dir, name)
    for name in os.listdir(contrastImg_wordpeak_social_dir)
    if name.endswith(".nii.gz")
]

print(all_imgs_wordpeak_social)

second_level_input_wordpeak_social = all_imgs_wordpeak_social

# create a design matrix for one sample t test, to be used as input for the second level model
design_matrix_wordpeak_social = pd.DataFrame(
    [1] * len(second_level_input_wordpeak_social),
    columns=["intercept"],
)

# set up group analysis for one sample t test on the contrast images
second_level_model = SecondLevelModel()
second_level_model = second_level_model.fit(
    second_level_input_wordpeak_social,
    design_matrix=design_matrix_wordpeak_social,
)

# run one sample t test
z_map_wordpeak_social = second_level_model.compute_contrast(
    second_level_contrast="intercept",
    output_type="z_score",
)

z_map_wordpeak_social.to_filename(processed_dir + "SecondLevel_wordpeak_social_zscore_macFamily.nii.gz")

thresholded_map_wordpeak_social, threshold_wordpeak_social = threshold_stats_img(
    stat_img=z_map_wordpeak_social,  # or p_map
    alpha=0.05,
    height_control='fdr',  # or 'bonferroni'
    cluster_threshold=0,   # min cluster size in voxels
    two_sided=True
)
print(f"The p<.05 FDR-corrected threshold is z score of {threshold_wordpeak_social:.3g}")

# save as brain image
thresholded_map_wordpeak_social.to_filename(processed_dir + "threshold_"+f"{threshold_wordpeak_social:.3g}"+"_"+
                                     "SecondLevel_wordpeak_social_fdrcorrect_macFamily.nii.gz")