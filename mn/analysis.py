import subprocess
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from nilearn import image, masking
from nilearn.glm.first_level import FirstLevelModel
from nilearn.masking import apply_mask, unmask
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.plotting import plot_design_matrix, plot_contrast_matrix
import whisperx
import pandas as pd
import os
import stanza
from nilearn import plotting
from nilearn.glm.second_level import SecondLevelModel
from nilearn.glm import threshold_stats_img

alias_dir = ".\\fmriprep"
foundationScores_dir = "./text/foundationScores/"
mask_dir = "./masks/"

#
#   Use PYTHON 12, ffmpeg needs to be installed, latest everything else
#
#   This transcribes audio files in using WHISPERX, which builds on OpenAI whisper model for speech-to-text
#

def transcribe(task):
    device = "cpu"
    batch_size = 4  # reduce if low on GPU mem
    compute_type = "int8"  # change to "int8" if low on GPU mem (may reduce accuracy)
    # try different models if the transcription is failing; large-v3 or large-v2 works well for complex text, tiny or small for regular dialogue speech patterns
    # if that fails again, then use transcribe_x from text/timestamps instead of the provided transcript for foundation scoring
    model = whisperx.load_model("large-v3", device, compute_type=compute_type, language="en")
    transcript_text = ""

    audio_file = ".\\audio\\" + task + "_audio.wav"
    if os.path.exists(audio_file):
        print("The audio file " + audio_file + " exists.")
    else:
        print("The audio file " + audio_file + " DOES NOT EXIST.")
    print( "Speech to text from: ./audio/" + task + "_audio.wav")
    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, language="en", batch_size=batch_size)
    # print(result["segments"]) # before alignment

    # 2. Align whisper output
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    df_words = pd.DataFrame(columns=["phrase","word","start", "end"])
    df_phrases = pd.DataFrame(columns=["text","start", "end"])
    phraseNumber = 0
    for segment in result['segments']:
        print( segment )
        phraseNumber = phraseNumber + 1
        df_phrases.loc[phraseNumber] = segment
        for word in segment['words']:
            df_words.loc[len(df_words)] = [phraseNumber, word['word'], word['start'], word['end']]
            transcript_text = transcript_text + " " + word['word']
    os.makedirs("./text/timestamps/"+task+"/", mode=0o777, exist_ok=True)  # this checks if the directory exists and creates it, if not
    df_words.to_csv("./text/timestamps/"+task+"/"+task+"_transcription_per_word_x.csv", index=False)
    df_phrases.to_csv("./text/timestamps/"+task+"/"+task+"_transcription_per_phrase_x.csv", index=False)
    with open("./text/timestamps/"+task+"/"+task+"_transcription_x.txt", "w+") as fh:
        nlp = stanza.Pipeline(language="en", processors="tokenize")
        parsedDoc = nlp(transcript_text)
        for sentence in parsedDoc.sentences:
            fh.write(f"{sentence.text}\n")

    import gc; import torch; gc.collect(); torch.cuda.empty_cache(); del model_a

# FUNCTION FOR DELETING STORY fMRI IMAGES THROUGH DATALAD

def dropStory(task):
    if task=="prettymouthaffair" or task=="prettymouthparanoia":
        task="prettymouth"
    elif task=="milkywayoriginal" or task=="milkywaysynonyms" or task=="milkywayvodka":
        task="milkyway"
    else:
        task=task
    for root, dirs, files in os.walk(alias_dir):
        for file in files:
            if task in file and file.endswith("space-MNI152NLin2009cAsym_res-native_desc-preproc_bold.nii.gz"):
                epi = os.path.join(root,file)
                print("Removing data file for %s" % (epi))
                subprocess.run(["datalad", "drop", epi], shell=True)
            if task in file and file.endswith("desc-confounds_regressors.tsv"):
                epi = os.path.join(root, file)
                #exe_path = pathlib.PureWindowsPath(epi).as_posix()
                print("Removing data file for %s" % (epi))
                subprocess.call(["datalad", "drop", epi], shell=True)

# FUNCTION FOR LOADING ALL IMAGE FILE NAMES FOR A STORY

def load_image_files(story):
    # return a list of files
    story = story + "_"
    files = []
    for root, dirs, files in os.walk(alias_dir):
        for file in files:
            if story in file and file.endswith("space-MNI152NLin2009cAsym_res-native_desc-preproc_bold.nii.gz"):
                if "run-1" in file:
                    print( "Appending " + file)
                    files.append(file)
                elif "run-2" in file:
                    print( "Ignoring data for run 2")
                else:
                    files.append( file )
    return files

# FUNCTION FOR LOADING SINGLE .NII IMAGE FOR A PARTICIPANT IN A STORY

def load_image_data(sub,task):
    # Load image
    story = ""
    if task=="prettymouthaffair" or task=="prettymouthparanoia":
        story="prettymouth"
    elif task=="milkywayoriginal" or task=="milkywaysynonyms" or task=="milkywayvodka":
        story="milkyway"
    else:
        story=task
    story = story + "_"
    for root, dirs, files in os.walk(alias_dir):
        for file in files:
            if story in file and file.endswith("space-MNI152NLin2009cAsym_res-native_desc-preproc_bold.nii.gz") and "sub-"+str(sub) in file:
                if "run-1" in file:
                    img_path = os.path.join(alias_dir,"sub-%03s/func/sub-%03s_task-%srun-1_space-MNI152NLin2009cAsym_res-native_desc-preproc_bold.nii.gz" % (sub, sub, story))
                elif "run-2" in file:
                    print( "Ignoring data for run 2")
                else:
                    img_path = os.path.join(alias_dir,"sub-%03s/func/sub-%03s_task-%sspace-MNI152NLin2009cAsym_res-native_desc-preproc_bold.nii.gz" % (sub, sub, story))
                #print("Loading data from %s" % (img_path))
                img = image.load_img(img_path)
    return img, img_path

# FUNCTION FOR DOWNLOADING STORY fMRI IMAGES THROUGH DATALAD

def downloadStory( task ):
    print( "starting download of " + task )
    if task=="prettymouthaffair" or task=="prettymouthparanoia":
        task="prettymouth"
    elif task=="milkywayoriginal" or task=="milkywaysynonyms" or task=="milkywayvodka":
        task="milkyway"
    else:
        task=task
    for root, dirs, files in os.walk(alias_dir):
        for file in files:
            if str(task) in file and file.endswith("space-MNI152NLin2009cAsym_res-native_desc-preproc_bold.nii.gz"):
                epi = os.path.join(root,file)
                #print("Downloading data for %s" % (epi))
                subprocess.call(["datalad","get", epi], shell=True)
            if str(task) in file and file.endswith("desc-confounds_regressors.tsv"):
                epi = os.path.join(root, file)
                #print("Downloading data for %s" % (epi))
                subprocess.call(["datalad", "get", epi], shell=True)

# HELPER FUNCTION FOR EXCLUDING PARTICIPANTS USING excluded.xlsx in root directory

def exclude_participants(story, participants):
    df = pd.read_excel('excluded.xlsx')
    removed = []
    if not df.loc[df['story'] == story, 'ids'].isnull().all():
        cell = df.loc[df['story'] == story, 'ids'].values[0]
        excluded = cell.split(',')
        for bye in excluded:
            if bye in participants:
                participants.remove(bye)
                removed.append(bye)
    if len(removed)>0:
        print("Excluding " + ",".join(removed) + " from " + story)
    return participants

# HELPER FUNCTION FOR MERGING FIRST LEVEL MODELS FOR MILKYWAY VARIANTS
import shutil

def mergeMilkyway():
    print( "Merging milkyways...")
    shutil.copytree(processed_dir + "/milkywayoriginal/", processed_dir + "/milkyway/", dirs_exist_ok=True)
    shutil.copytree(processed_dir + "/milkywayvodka/", processed_dir + "/milkyway/", dirs_exist_ok=True)
    shutil.copytree(processed_dir + "/milkywaysynonyms/", processed_dir + "/milkyway/", dirs_exist_ok=True)

# LOAD ALL PARTICIPANT NUMBERS, EXCLUDING THE ONES FROM excluded.xlsx
def load_participants(task):
    #return a list of all participant numbers for a story
    participants = []
    story = ""
    if task == "milkyway": mergeMilkyway() #in case this is a milkyway second-level model this will me called
    if task=="prettymouthaffair" or task=="prettymouthparanoia": story="prettymouth"
    elif task=="milkywayoriginal" or task=="milkywaysynonyms" or task=="milkywayvodka":
        story="milkyway"
    else: story=task
    story = story + "_"
    for root, dirs, files in os.walk(alias_dir):
        for file in files:
            if story in file and file.endswith("space-MNI152NLin2009cAsym_res-native_desc-preproc_bold.nii.gz"):
                participant_number = file[4:7]
                #print("Getting participant number %03s for %04s" % (participant_number, story[:-1]))
                if participant_number not in participants:
                    participants.append(participant_number)
    return exclude_participants(task, participants)

# Load MRI file (in Nifti format)
def load_epi_data(sub,story):
    #print( "getting epi data for %s" % (sub))
    if story =='prettymouthaffair' or story == 'prettymouthparanoia': story='prettymouth'
    elif story=="milkywayoriginal" or story=="milkywaysynonyms" or story=="milkywayvodka":
        story="milkyway"
    else : story=story
    story = story + "_"
    for root, dirs, files in os.walk(alias_dir):
        for file in files:
            if story in file and file.endswith("space-MNI152NLin2009cAsym_res-native_desc-preproc_bold.nii.gz") and "sub-"+str(sub) in file:
                #print( "Downloading data for %s" % (sub))
                epi_in = os.path.join(root,file)
                if "run-1" in file:
                    epi_in = os.path.join(alias_dir,"sub-%03s/func/sub-%03s_task-%srun-1_space-MNI152NLin2009cAsym_res-native_desc-preproc_bold.nii.gz" % (sub, sub, story))
                elif "run-2" in file:
                    print( "Ignoring data for run 2")
                else:
                    epi_in = os.path.join(alias_dir,"sub-%03s/func/sub-%03s_task-%sspace-MNI152NLin2009cAsym_res-native_desc-preproc_bold.nii.gz" % (sub, sub, story))
                epi_data = nib.load(epi_in)
                #print("Loading data from %s" % (epi_in))
    return epi_data, epi_in

# Load tsv file with regressors
def load_regressor(sub,story):
    if story =='prettymouthaffair' or story == 'prettymouthparanoia': story='prettymouth'
    elif story=="milkywayoriginal" or story=="milkywaysynonyms" or story=="milkywayvodka":
        story="milkyway"
    else: story = story
    story = story + "_"
    for root, dirs, files in os.walk(alias_dir):
        for file in files:
            if story in file and file.endswith("desc-confounds_regressors.tsv") and "sub-"+sub in file:
                if "run-1" in file:
                    regressor_location = os.path.join(alias_dir,"sub-%03s/func/sub-%03s_task-%04srun-1_desc-confounds_regressors.tsv" % (sub, sub, story))
                elif "run-2" in file:
                    print( "Ignoring regressors for run 2")
                else:
                    regressor_location = os.path.join(alias_dir,"sub-%03s/func/sub-%03s_task-%04sdesc-confounds_regressors.tsv" % (sub, sub, story))

                #print("Loading regressors from %s" % (regressor_location))
                regressor = pd.read_csv(regressor_location, sep='\t')
    return regressor, regressor_location

#return a list of tuples (index, foundation name, foundation score, onset, duration)
def get_top_foundation_per_sentence(story, foundations):
    story = story + "_"
    sentence_tuples = []
    for root, dirs, files in os.walk(foundationScores_dir):
        for file in files:
            if story in file:
                print("Getting top foundations per sentence in %03s " % (story[:-1]))
                df = pd.read_excel(os.path.join(root, file))
                for index, row in df.iterrows():
                    if ( row[foundations].max() > 0 ):
                        tuple = row[foundations].idxmax(), row['start'], row['end'] - row['start']
                    else:
                        tuple = 'baseline', row['start'], row['end'] - row['start']
                    sentence_tuples.append(tuple)
    return sentence_tuples

#return a list of timestamps for onset
def load_onsets(story):
    story = story + "_"
    onsets = []
    for root, dirs, files in os.walk(foundationScores_dir):
        for file in files:
            if story in file:
                print("Getting sentence onsets for %03s " % (story[:-1]))
                onsets = pd.read_excel(os.path.join(root, file), usecols=['start'] )
    return onsets

#return a list of durations for a story, given timestamps
def load_durations(story):
    story = story + "_"
    durations = []
    for root, dirs, files in os.walk(foundationScores_dir):
        for file in files:
            if story in file:
                print("Getting sentence durations for %03s " % (story[:-1]))
                df = pd.read_excel(os.path.join(root, file), usecols=['start', 'end'] )
                durations.append( df['end'] - df['start'] )
    return durations

#
# MODELLING FUNCTIONS FOLLOW
#

def firstLevelMacVices(story, processed_dir, scoring):
    os.makedirs(processed_dir + story + "\\7_MAC_V\\", mode=0o777, exist_ok=True)  # this checks if the directory for dropping .nii files exists and creates it, if not
    # this creates a dataframe with per sentence and per segment scores for all foundations and column names that match them, plus segment file name as first element
    if (story=='prettymouthaffair') or (story=='prettymouthparanoia'):
        segmentFileDF = pd.read_excel(foundationScores_dir + "prettymouth_"+scoring+"_MFT_MAC.xlsx")
    else:
        segmentFileDF = pd.read_excel(foundationScores_dir + story + "_MFT_MAC.xlsx")
    sentenceValues = segmentFileDF[['MAC_a_fairness_vice',
                                    'MAC_a_group_vice',
                                    'MAC_a_deference_vice',
                                    'MAC_a_heroism_vice',
                                    'MAC_a_reciprocity_vice',
                                    'MAC_a_family_vice',
                                    'MAC_a_property_vice']]
    foundations_per_sentence = get_top_foundation_per_sentence(story,
                                                               ['MAC_a_fairness_vice',
                                                                'MAC_a_group_vice',
                                                                'MAC_a_deference_vice',
                                                                'MAC_a_heroism_vice',
                                                                'MAC_a_reciprocity_vice',
                                                                'MAC_a_family_vice',
                                                                'MAC_a_property_vice'])

    events = pd.DataFrame(foundations_per_sentence, columns=["trial_type", "onset", "duration"])

    #
    #   Data transform
    #

    for participant in load_participants(story):
        #print ("Building first-level models for participant %s" % (participant))
        epi_data_NIFTI, epi_path = load_epi_data(participant, story)
        df, regressor_path = load_regressor(participant, story)
        confound_file1 = df[['csf', 'white_matter', 'trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']].to_numpy()

        # this weird thing handles cases in which there were multiple runs for a single participant
        if "run-1" in epi_path:
            participant = participant + "_run-1_"

        if "run-2" in epi_path:
            participant = participant + "_run-2_"

        # Make an average
        mean_img = image.mean_img(epi_data_NIFTI, copy_header=True)
        mask = masking.compute_epi_mask(mean_img, lower_cutoff=0.2, upper_cutoff=0.85, opening=3, connected=True)

        # Clean and smooth data
        epi_data_NIFTI = image.clean_img(epi_data_NIFTI, standardize=False)
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
        os.makedirs(processed_dir + story + "\\7_MAC_V\\F_contrast\\", mode=0o777, exist_ok=True)  # this checks if the directory exists and creates it, if not
        z_map_masked.to_filename(processed_dir+story+"\\7_MAC_V\\F_contrast\\"+participant+"_"+story+"_F_contrast_7_MAC_V_perSentence.nii.gz")

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

        os.makedirs(processed_dir + story + "\\7_MAC_V\\VsOther6\\", mode=0o777, exist_ok=True)  # this checks if the directory exists and creates it, if not
        z_map_foundation1.to_filename(processed_dir+story+"\\7_MAC_V\\VsOther6\\"+ participant + "_"+story+"_7_MAC_V_perSentence_z_map_foundation1_vsOther6.nii.gz")
        z_map_foundation2.to_filename(processed_dir+story+"\\7_MAC_V\\VsOther6\\"+ participant + "_"+story+"_7_MAC_V_perSentence_z_map_foundation2_vsOther6.nii.gz")
        z_map_foundation3.to_filename(processed_dir+story+"\\7_MAC_V\\VsOther6\\"+ participant + "_"+story+"_7_MAC_V_perSentence_z_map_foundation3_vsOther6.nii.gz")
        z_map_foundation4.to_filename(processed_dir+story+"\\7_MAC_V\\VsOther6\\"+ participant + "_"+story+"_7_MAC_V_perSentence_z_map_foundation4_vsOther6.nii.gz")
        z_map_foundation5.to_filename(processed_dir+story+"\\7_MAC_V\\VsOther6\\"+ participant + "_"+story+"_7_MAC_V_perSentence_z_map_foundation5_vsOther6.nii.gz")
        z_map_foundation6.to_filename(processed_dir+story+"\\7_MAC_V\\VsOther6\\"+ participant + "_"+story+"_7_MAC_V_perSentence_z_map_foundation6_vsOther6.nii.gz")
        z_map_foundation7.to_filename(processed_dir+story+"\\7_MAC_V\\VsOther6\\"+ participant + "_"+story+"_7_MAC_V_perSentence_z_map_foundation7_vsOther6.nii.gz")

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

        os.makedirs(processed_dir + story + "\\7_MAC_V\\VsBaseline\\", mode=0o777, exist_ok=True)  # this checks if the directory exists and creates it, if not
        z_map_foundation1vsbase.to_filename(processed_dir+story+"\\7_MAC_V\\VsBaseline\\"+ participant + "_"+story+"_7_MAC_V_perSentence_z_map_foundation1_vsBaseline.nii.gz")
        z_map_foundation2vsbase.to_filename(processed_dir+story+"\\7_MAC_V\\VsBaseline\\"+ participant + "_"+story+"_7_MAC_V_perSentence_z_map_foundation2_vsBaseline.nii.gz")
        z_map_foundation3vsbase.to_filename(processed_dir+story+"\\7_MAC_V\\VsBaseline\\"+ participant + "_"+story+"_7_MAC_V_perSentence_z_map_foundation3_vsBaseline.nii.gz")
        z_map_foundation4vsbase.to_filename(processed_dir+story+"\\7_MAC_V\\VsBaseline\\"+ participant + "_"+story+"_7_MAC_V_perSentence_z_map_foundation4_vsBaseline.nii.gz")
        z_map_foundation5vsbase.to_filename(processed_dir+story+"\\7_MAC_V\\VsBaseline\\"+ participant + "_"+story+"_7_MAC_V_perSentence_z_map_foundation5_vsBaseline.nii.gz")
        z_map_foundation6vsbase.to_filename(processed_dir+story+"\\7_MAC_V\\VsBaseline\\"+ participant + "_"+story+"_7_MAC_V_perSentence_z_map_foundation6_vsBaseline.nii.gz")
        z_map_foundation7vsbase.to_filename(processed_dir+story+"\\7_MAC_V\\VsBaseline\\"+ participant + "_"+story+"_7_MAC_V_perSentence_z_map_foundation7_vsBaseline.nii.gz")

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
        os.makedirs(processed_dir + story + "\\7_MAC_V\\Conjunction\\", mode=0o777, exist_ok=True)  # this checks if the directory exists and creates it, if not
        min_stat_map.to_filename(processed_dir+story+"\\7_MAC_V\\Conjunction\\"+ participant + "_"+story+"_7_MAC_V_perSentence_minimum_stat_conjunction.nii.gz")


        plotting.plot_stat_map(z_map_masked, bg_img=mean_img, title="Masked z-map")

def firstLevelMacVirtues(story, processed_dir):
    os.makedirs(processed_dir + story + "\\7_MAC\\", mode=0o777, exist_ok=True)  # this checks if the directory for dropping .nii files exists and creates it, if not
    # this creates a dataframe with per sentence and per segment scores for all foundations and column names that match them, plus segment file name as first element
    if (story=='prettymouthaffair') or (story=='prettymouthparanoia'):
        segmentFileDF = pd.read_excel(foundationScores_dir + "prettymouth_MFT_MAC.xlsx")
    else:
        segmentFileDF = pd.read_excel(foundationScores_dir + story + "_MFT_MAC.xlsx")
    sentenceValues = segmentFileDF[[
                                    'MAC_a_fairness_virtue',
                                    'MAC_a_group_virtue',
                                    'MAC_a_deference_virtue',
                                    'MAC_a_heroism_virtue',
                                    'MAC_a_reciprocity_virtue',
                                    'MAC_a_family_virtue',
                                    'MAC_a_property_virtue']]

    # This creates the events and durations FOR DESIGN MATRIX
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
    #
    #   Data transform
    #

    for participant in load_participants(story):
        #print ("Building first-level models for participant %s" % (participant))
        epi_data_NIFTI, epi_path = load_epi_data(participant, story)
        df, regressor_path = load_regressor(participant, story)
        confound_file1 = df[
            ['csf', 'white_matter', 'trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']].to_numpy()

        # this weird thing handles cases in which there were multiple runs for a single participant
        if "run-1" in epi_path:
            participant = participant + "_run-1_"
        if "run-2" in epi_path:
            participant = participant + "_run-2_"

        # Make an average
        mean_img = image.mean_img(epi_data_NIFTI, copy_header=True)
        mask = masking.compute_epi_mask(mean_img, lower_cutoff=0.2, upper_cutoff=0.85, opening=3, connected=True)

        # Clean and smooth data
        epi_data_NIFTI = image.clean_img(epi_data_NIFTI, standardize=False)
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

        # foundation 2
        c2 = np.array([0, 1, 0, 0, 0, 0, 0, -1])  # exact -1/6
        z_map_foundation2vsbase = FM1.compute_contrast(c2, stat_type='t', output_type='z_score')

        # foundation 3
        c3 = np.array([0, 0, 1, 0, 0, 0, 0, -1])  # exact -1/6
        z_map_foundation3vsbase = FM1.compute_contrast(c3, stat_type='t', output_type='z_score')

        # foundation 4
        c4 = np.array([0, 0, 0, 1, 0, 0, 0, -1])  # exact -1/6
        z_map_foundation4vsbase = FM1.compute_contrast(c4, stat_type='t', output_type='z_score')

        # foundation 5
        c5 = np.array([0, 0, 0, 0, 1, 0, 0, -1])  # exact -1/6
        z_map_foundation5vsbase = FM1.compute_contrast(c5, stat_type='t', output_type='z_score')

        # foundation 6
        c6 = np.array([0, 0, 0, 0, 0, 1, 0, -1])  # exact -1/6
        z_map_foundation6vsbase = FM1.compute_contrast(c6, stat_type='t', output_type='z_score')

        # foundation 7
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

def univariateWithMask(story, mask, processed_dir):

    # ----------------------------
    # Load prep data structures
    # ----------------------------
    participants = load_participants(story)
    n_participants = len(participants)
    print( "Using " + mask + " for " + story )
    mask_orig = image.load_img(mask_dir + mask)

    # ----------------------------
    # Reference image (for resampling)
    # ----------------------------
    ref_img = load_image_data(participants[0], story)[0]
    # Resample masks to match the first image’s space
    mask_res = image.resample_to_img(mask_orig, ref_img.slicer[..., 0], interpolation='nearest', force_resample=True,copy_header=True,clip=True, fill_value=0)

    mask_data = mask_res.get_fdata().astype(bool)

    n_voxels = np.sum(mask_data)
    n_timepoints = ref_img.shape[-1]
    #print(f"Each participant has {n_timepoints} timepoints")

    # ----------------------------
    # Preallocate arrays (voxels × time × participants)
    # ----------------------------
    array_3d = np.zeros((n_voxels, n_timepoints, n_participants))

    # ----------------------------
    # Loop through participants
    # ----------------------------
    i = 0
    for participant in participants:
        # Load image and confounds
        img = load_image_data(participant, story)[0]
        confounds_df = load_regressor(participant, story)[0]

        # Select confounds
        confound_cols = [c for c in confounds_df.columns if any(x in c for x in [
            "trans_x", "trans_y", "trans_z",
            "rot_x", "rot_y", "rot_z"
        ])]
        confounds = confounds_df[confound_cols].fillna(0).values

        # Clean the image (regress confounds, detrend, standardize)
        clean_data = image.clean_img(
            img,
            confounds=confounds,
            detrend=True,
            standardize=True
        )

        clean_img_data = clean_data.get_fdata()

        # Resample masks to participant space (if needed)
        mask_res = image.resample_to_img(mask_orig, clean_data.slicer[..., 0], interpolation='nearest', force_resample=True,copy_header=True, fill_value=0)

        mask_data = mask_res.get_fdata().astype(bool)

        # Extract voxel time series (voxels × time)
        ts = clean_img_data[mask_data, :]
        # reshape in case the reference recording is shorter or longer than what you do here
        if ts.shape[1] > n_timepoints:
            ts = ts[:, :n_timepoints]
        if ts.shape[1] < n_timepoints:
            padding = n_timepoints - clean_img_data.shape[-1]
            ts.resize(ts.shape[0], ts[1].shape[0]+padding)

        # Store in 3D array
        array_3d[:, :, i] = ts
        i += 1

    # ----------------------------
    # Save outputs
    # ----------------------------
    os.makedirs(processed_dir + '/' + story + '/', mode=0o777,
                exist_ok=True)  # this checks if the directory exists and creates it, if not
    np.save(processed_dir + '/' + story + "/" + story + "_" + mask.split('.')[0] + "_3D_clean.npy", array_3d)
    #print("Mask array shape:", array_3d.shape, "(voxels × time × subjects)")

def secondLevelMacVices(task, processed_dir):
    # ## second level model directories for PER SENTENCE
    contrastImg_dir = processed_dir + task + "/7_MAC_V/Conjunction/"  # Or /F_contrast/
    processed_dir_local = processed_dir+task+"/7_MAC_V/SecondLevel_contrast/"
    os.makedirs(processed_dir + "/", mode=0o777,
                exist_ok=True)  # this checks if the directory exists and creates it, if not
    os.makedirs(contrastImg_dir + "/", mode=0o777,
                exist_ok=True)  # this checks if the directory exists and creates it, if not

    #main loop over foundations
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
    os.makedirs(processed_dir_local + "/", mode=0o777,exist_ok=True)  # this checks if the directory exists and creates it, if not
    (z_map.to_filename
     (processed_dir_local + "/" + "SecondLevel_CONJUNCTION_"+task+"_per_sentence_MACVices_zscore.nii.gz"))

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
    thresholded_map.to_filename(processed_dir_local + "/threshold_"+f"{threshold:.3g}"+"_"+
                                "SecondLevel_CONJUNCTION_"+task+"_fdrcorrect_per_sentence_MACVices.nii.gz")
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
    thresholded_map2.to_filename(processed_dir_local + "/threshold_"+f"{threshold2:.3g}"+"_"+
                                 "SecondLevel_CONJUNCTION_"+task+"_bonfcorrect_per_Sentence_MACVices.nii.gz")

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

def secondLevelMacVirtues(task, processed_dir):
    # ## second level model directories for PER SENTENCE
    contrastImg_dir = processed_dir +task+"/7_MAC/Conjunction/"  # Or /F_contrast/
    contrastImg_Testdir = ""
    processed_dir_local = processed_dir+task+"/7_MAC/SecondLevel_contrast/"
    os.makedirs(processed_dir + "/", mode=0o777,
                exist_ok=True)  # this checks if the directory exists and creates it, if not
    os.makedirs(contrastImg_dir + "/", mode=0o777,
                exist_ok=True)  # this checks if the directory exists and creates it, if not
    #main loop over foundations
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
    os.makedirs(processed_dir_local + "/", mode=0o777,exist_ok=True)  # this checks if the directory exists and creates it, if not
    (z_map.to_filename
     (processed_dir_local + "/" + "SecondLevel_CONJUNCTION_"+task+"_per_sentence_MACVirtues_zscore.nii.gz"))

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
    thresholded_map.to_filename(processed_dir_local + "/threshold_"+f"{threshold:.3g}"+"_"+
                                "SecondLevel_CONJUNCTION_"+task+"_fdrcorrect_per_sentence_MACVirtues.nii.gz")
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
    thresholded_map2.to_filename(processed_dir_local + "/threshold_"+f"{threshold2:.3g}"+"_"+
                                 "SecondLevel_CONJUNCTION_"+task+"_bonfcorrect_per_Sentence_MACVirtues.nii.gz")

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

def secondLevelMacVices_1v6(task, processed_dir):
    for foundation in range(1,8):
        # ## second level model directories for PER SENTENCE
        contrastImg_dir = processed_dir + task + "/7_MAC_V/VsOther6/"  # Or /F_contrast/
        contrastImg_Testdir = ""
        processed_dir_local = processed_dir+task+"/7_MAC_V/SecondLevel_contrast/"
        os.makedirs(processed_dir + "/", mode=0o777,
                    exist_ok=True)  # this checks if the directory exists and creates it, if not
        os.makedirs(contrastImg_dir + "/", mode=0o777,
                    exist_ok=True)  # this checks if the directory exists and creates it, if not

        #main loop over foundations
        all_imgs = [
            os.path.join(contrastImg_dir, name)
            for name in os.listdir(contrastImg_dir)
                if name.endswith(f"foundation{foundation}_vsOther6.nii.gz")
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
        os.makedirs(processed_dir_local + "/", mode=0o777,exist_ok=True)  # this checks if the directory exists and creates it, if not
        (z_map.to_filename
         (processed_dir_local + "/" + "SecondLevel_"+task+'_foundation'+str(foundation)+"_VsOther6_per_sentence_MACVirtues_zscore.nii.gz"))

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
        thresholded_map.to_filename(processed_dir_local + "/threshold_"+f"{threshold:.3g}"+"_"+
                                    "SecondLevel_"+task+'_foundation'+str(foundation)+"_VsOther6_fdrcorrect_per_sentence_MACVirtues.nii.gz")

def secondLevelMacVirtues_1v6(task, processed_dir):
    for foundation in range(1, 8):
        # ## second level model directories for PER SENTENCE
        contrastImg_dir = processed_dir + task + "/7_MAC/VsOther6/"  # Or /F_contrast/
        contrastImg_Testdir = ""
        processed_dir_local = processed_dir+task+"/7_MAC/SecondLevel_contrast/"
        os.makedirs(processed_dir + "/", mode=0o777,
                    exist_ok=True)  # this checks if the directory exists and creates it, if not
        os.makedirs(contrastImg_dir + "/", mode=0o777,
                    exist_ok=True)  # this checks if the directory exists and creates it, if not

        #main loop over foundations
        all_imgs = [
            os.path.join(contrastImg_dir, name)
            for name in os.listdir(contrastImg_dir)
                if name.endswith(f"foundation{foundation}_vsOther6.nii.gz")
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
        os.makedirs(processed_dir_local + "/", mode=0o777,exist_ok=True)  # this checks if the directory exists and creates it, if not
        (z_map.to_filename
         (processed_dir_local + "/" + "SecondLevel_"+task+'_foundation'+str(foundation)+"_VsOther6_per_sentence_MACVirtues_zscore.nii.gz"))

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
        thresholded_map.to_filename(processed_dir_local + "/threshold_"+f"{threshold:.3g}"+"_"+
                                    "SecondLevel_"+task+'_foundation'+str(foundation)+"_VsOther6_fdrcorrect_per_sentence_MACVirtues.nii.gz")

def secondLevelMacVices_1vB(task, processed_dir):
    for foundation in range(1, 8):
        # ## second level model directories for PER SENTENCE
        contrastImg_dir = processed_dir + task + "/7_MAC_V/VsBaseline/"  # Or /F_contrast/
        contrastImg_Testdir = ""
        processed_dir_local = processed_dir +task+"/7_MAC_V/SecondLevel_contrast/"
        os.makedirs(processed_dir + "/", mode=0o777,
                    exist_ok=True)  # this checks if the directory exists and creates it, if not
        os.makedirs(contrastImg_dir + "/", mode=0o777,
                    exist_ok=True)  # this checks if the directory exists and creates it, if not

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
        os.makedirs(processed_dir_local + "/", mode=0o777,exist_ok=True)  # this checks if the directory exists and creates it, if not
        (z_map.to_filename
         (processed_dir_local + "/" + "SecondLevel_"+task+'_foundation'+str(foundation)+"_VsBaseline_per_sentence_MACVices_zscore.nii.gz"))

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
        thresholded_map.to_filename(processed_dir_local + "/threshold_"+f"{threshold:.3g}"+"_"+
                                    "SecondLevel_"+task+'_foundation'+str(foundation)+"_VsBaseline_fdrcorrect_per_sentence_MACVices.nii.gz")

def secondLevelMacVirtues_1vB(task, processed_dir):
    # ## second level model directories for PER SENTENCE
    for foundation in range(1, 8):
        contrastImg_dir = processed_dir + task + "/7_MAC/VsBaseline/"  # Or /F_contrast/
        contrastImg_Testdir = ""
        processed_dir_local = processed_dir+task+"/7_MAC/SecondLevel_contrast/"
        os.makedirs(processed_dir + "/", mode=0o777,
                    exist_ok=True)  # this checks if the directory exists and creates it, if not
        os.makedirs(contrastImg_dir + "/", mode=0o777,
                    exist_ok=True)  # this checks if the directory exists and creates it, if not

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
        os.makedirs(processed_dir_local + "/", mode=0o777,exist_ok=True)  # this checks if the directory exists and creates it, if not
        (z_map.to_filename
         (processed_dir_local + "/" + "SecondLevel_"+task+'_foundation'+str(foundation)+"_VsBaseline_per_sentence_MACVirtues_zscore.nii.gz"))

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
        thresholded_map.to_filename(processed_dir_local + "/threshold_"+f"{threshold:.3g}"+"_"+
                                    "SecondLevel_"+task+'_foundation'+str(foundation)+"_VsBaseline_fdrcorrect_per_sentence_MACVirtues.nii.gz")