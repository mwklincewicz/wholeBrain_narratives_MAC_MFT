import pandas as pd
import os
from nilearn import image
from nilearn import plotting
from nilearn.glm.second_level import SecondLevelModel
from nilearn.glm import threshold_stats_img

# # Second Level Model PHYSICAL STORY
contrastImg_physical_dir = "_results/processed_first_level_MAC_family/physical/"
contrastImg_physical_Testdir = "./_testData/processed_first_level_MAC_family/shapesphysical/"
processed_dir = "_results/processed_second_level_MAC_family/"

all_imgs_physical = [
    os.path.join(contrastImg_physical_dir, name)
    for name in os.listdir(contrastImg_physical_dir)
]

#print(all_imgs_physical)

second_level_input_physical = all_imgs_physical

# create a design matrix for one sample t test, to be used as input for the second level model
design_matrix_physical = pd.DataFrame(
    [1] * len(second_level_input_physical),
    columns=["intercept"],
)

# set up group analysis for one sample t test on the contrast images
second_level_model = SecondLevelModel()
second_level_model = second_level_model.fit(
    second_level_input_physical,
    design_matrix=design_matrix_physical,
)

# run one sample t test
z_map_physical = second_level_model.compute_contrast(
    second_level_contrast="intercept",
    output_type="z_score",
)

z_map_physical.to_filename(processed_dir + "SecondLevel_physical_zscore_macFamily.nii.gz")

thresholded_map_physical, threshold_physical = threshold_stats_img(
    stat_img=z_map_physical,  # or p_map
    alpha=0.05,
    height_control='fdr',  # or 'bonferroni'
    cluster_threshold=0,   # min cluster size in voxels
    two_sided=True
)
print(f"The p<.05 FDR-corrected threshold is z score of {threshold_physical:.3g}")

# save as brain image
thresholded_map_physical.to_filename(processed_dir + "threshold_"+f"{threshold_physical:.3g}"+"_"+
                                     "SecondLevel_physical_fdrcorrect_macFamily.nii.gz")

# quick visualization
plotting.plot_stat_map(
    thresholded_map_physical,
    title="Thresholded z map, fdr < .05",
    threshold=threshold_physical,
)

contrastImg_dir = "_results/processed_first_level_MAC_family/social/"
all_imgs = [
    os.path.join(contrastImg_dir, name)
    for name in os.listdir(contrastImg_dir)
]

second_level_input = all_imgs

# ## SECOND LEVEL MODEL: SOCIAL MINUS PHYSICAL
all_imgs_physical_sorted = sorted(all_imgs_physical)
all_imgs_social_sorted = sorted(all_imgs)
diff_imgs = [image.math_img("img1 - img2", img1=con1, img2=con2)
             for con1, con2 in zip(all_imgs_social_sorted, all_imgs_physical_sorted)]
design_matrix_socialMinusPhysical = pd.DataFrame([1] * len(diff_imgs), columns=["intercept"])

model = SecondLevelModel().fit(diff_imgs, design_matrix=design_matrix_socialMinusPhysical)
zmap_socialMinusPhysical = model.compute_contrast("intercept", output_type="z_score")

zmap_socialMinusPhysical.to_filename(processed_dir + "SecondLevel_SocialMinusPhysical_zscore_macFamily.nii.gz")



thresholded_map_socialMinusPhysical, threshold_socialMinusPhysical = threshold_stats_img(
    stat_img=zmap_socialMinusPhysical,  # or p_map
    alpha=0.05,
    height_control='fdr',  # or 'bonferroni'
    cluster_threshold=0,   # min cluster size in voxels
    two_sided=True
)
print(f"The p<.05 FDR-corrected threshold is z score of {threshold_socialMinusPhysical:.3g}")

# save as brain image
thresholded_map_socialMinusPhysical.to_filename(processed_dir + "threshold_"+f"{threshold_socialMinusPhysical:.3g}"+"_"+
                                                "SecondLevel_SocialMinusPhysical_fdrcorrect_macFamily.nii.gz")

# # Second level model, PHYSICAL MINUS SOCIAL
diff_imgs_2 = [image.math_img("img1 - img2", img1=con1, img2=con2)
             for con1, con2 in zip(all_imgs_physical_sorted, all_imgs_social_sorted)]

design_matrix_PhysicalMinusSocial = pd.DataFrame([1] * len(diff_imgs_2), columns=["intercept"])

model = SecondLevelModel().fit(diff_imgs_2, design_matrix=design_matrix_PhysicalMinusSocial)
zmap_PhysicalMinusSocial = model.compute_contrast("intercept", output_type="z_score")

zmap_PhysicalMinusSocial.to_filename(processed_dir + "SecondLevel_PhysicalMinusSocial_zscore_macFamily.nii.gz")

thresholded_map_PhysicalMinusSocial, threshold_PhysicalMinusSocial = threshold_stats_img(
    stat_img=zmap_PhysicalMinusSocial,  # or p_map
    alpha=0.05,
    height_control='fdr',  # or 'bonferroni'
    cluster_threshold=0,   # min cluster size in voxels
    two_sided=True
)
print(f"The p<.05 FDR-corrected threshold is z score of {threshold_PhysicalMinusSocial:.3g}")

# save as brain image
thresholded_map_PhysicalMinusSocial.to_filename(processed_dir + "threshold_"+f"{threshold_PhysicalMinusSocial:.3g}"+"_"+
                                                "SecondLevel_PhysicalMinusSocial_fdrcorrect_macFamily.nii.gz")