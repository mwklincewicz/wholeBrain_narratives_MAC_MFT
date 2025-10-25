import pandas as pd
import os
from nilearn import plotting
from nilearn.glm.second_level import SecondLevelModel
from nilearn.glm import threshold_stats_img

def run(task, foundation):
    # ## second level model directories for PER SENTENCE
    contrastImg_dir = "G:/fMRI_project/processed_first_level_per_sentence/" + task + "/7_MAC/VsBaseline/"  # Or /F_contrast/
    contrastImg_Testdir = ""
    processed_dir = "G:/fMRI_project/processed_first_level_per_sentence/"+task+"/7_MAC/SecondLevel_contrast/"
    #main loop over foundations
    all_imgs = [
        os.path.join(contrastImg_dir, name)
        for name in os.listdir(contrastImg_dir)
            if name.endswith(f"foundation{foundation}_vsBaseline.nii.gz")
    ]

    second_level_input = all_imgs
    print(second_level_input)

    # create a design matrix for one sample t test, to be used as input for the second level model
    design_matrix = pd.DataFrame(
        [1] * len(second_level_input),
        columns=["intercept"],
    )

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
    os.makedirs(processed_dir + "/", mode=0o777,exist_ok=True)  # this checks if the directory exists and creates it, if not
    (z_map.to_filename
     (processed_dir + "/" + "SecondLevel_"+task+'_foundation'+str(foundation)+"_VsBaseline_per_sentence_MACVirtues_zscore.nii.gz"))

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
    thresholded_map.to_filename(processed_dir + "/threshold_"+f"{threshold:.3g}"+"_"+
                                "SecondLevel_"+task+'_foundation'+str(foundation)+"_VsBaseline_fdrcorrect_per_sentence_MACVirtues.nii.gz")