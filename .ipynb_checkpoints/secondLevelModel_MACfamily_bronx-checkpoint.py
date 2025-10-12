import pandas as pd
import os
from nilearn import plotting
from nilearn.glm.second_level import SecondLevelModel
from nilearn.glm import threshold_stats_img

# ## second level model SOCIAL STORY
contrastImg_dir = "G:/fMRI_project/processed_first_level_MAC_family/bronx/"

contrastImg_Testdir = "./_testData/processed_first_level_MAC_family/shapessocial/"

processed_dir = "./processed_second_level_MAC_family/"

all_imgs = [
    os.path.join(contrastImg_dir, name)
    for name in os.listdir(contrastImg_dir)
        if name.endswith(".nii.gz")
]

second_level_input = all_imgs


# create a design matrix for one sample t test, to be used as input for the second level model
design_matrix = pd.DataFrame(
    [1] * len(second_level_input),
    columns=["intercept"],
)
design_matrix
# set up group analysis for one sample t test on the contrast images
second_level_model = SecondLevelModel() 
second_level_model = second_level_model.fit(
    second_level_input,
    design_matrix=design_matrix,
)
# run one sample t test
z_map = second_level_model.compute_contrast(
    second_level_contrast="intercept",
    output_type="z_score",
) 

z_map.to_filename(processed_dir + "SecondLevel_bronx_zscore_macFamily.nii.gz")

#output_type{‘z_score’, ‘stat’, ‘p_value’, ‘effect_size’, ‘effect_variance’, ‘all’},
# #### fdr correction


thresholded_map, threshold = threshold_stats_img(
    stat_img=z_map,  # or p_map
    alpha=0.05,
    height_control='fdr',  # or 'bonferroni'
    cluster_threshold=0,   # min cluster size in voxels
    two_sided=True
)
print(f"The p<.05 FDR-corrected threshold is z score of {threshold:.3g}")

# save as brain image
thresholded_map.to_filename(processed_dir + "threshold_"+f"{threshold:.3g}"+"_"+
                            "SecondLevel_bronx_fdrcorrect_macFamily.nii.gz")
# quick visualization
plotting.plot_stat_map(
    thresholded_map,
    title="Thresholded z map, fdr < .05",
    threshold=threshold,
)

plotting.plot_stat_map(
    thresholded_map,
    threshold=threshold,
    display_mode="z",
    title="fdr < .05",
)

plotting.plot_stat_map(
    thresholded_map,
    threshold=threshold,
    display_mode="x",
    title="fdr < .05",
)

# #### bonferroni correction
thresholded_map2, threshold2 = threshold_stats_img(
    z_map, alpha=0.05, height_control="bonferroni"
)
print(f"The p<.05 Bonferroni-corrected threshold is z score of {threshold2:.3g}")

# save as brain image
thresholded_map2.to_filename(processed_dir + "threshold_"+f"{threshold2:.3g}"+"_"+
                             "SecondLevel_bronx_bonfcorrect_macFamily.nii.gz")

# quick visualization
plotting.plot_stat_map(
    thresholded_map2,
    title="Thresholded z map, bonferroni < .05",
    threshold=threshold2,
)

# quick visualization
plotting.plot_stat_map(
    thresholded_map2,
    title="bonferroni < .05",
    threshold=threshold2,
    display_mode = 'z'
)