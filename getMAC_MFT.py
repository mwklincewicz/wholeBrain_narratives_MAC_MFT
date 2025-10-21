import os
from thefuzz import fuzz
import Levenshtein as levenshtein
import spacy
print("spaCy version:", spacy.__version__)
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from deep_translator import GoogleTranslator
import pandas as pd
print("pandas version:", pd.__version__)
from emfdscore.scoring import score_docs as emfd_score_docs
from emacscore.scoring import score_docs as emac_score_docs
from pathlib import Path

#   Use Python 3.8, with a batch file that installs
#   spacy 3.4
#   typing-extensions has to be 4.4
#   has to be pandas 1.5.3
#   also install scikit-learn 1.3
#   also install openpyxl
#   emfdscore from git
#   emacscore best to download zip and install with pip

#stories = ['shapesphysical','shapessocial']
stories = ['black']#,'21styear','tunnel','pieman','piemanpni'] #names of narrative files in 'text_to_be_segmented' subdir

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 400)

my_translator = GoogleTranslator(source='auto', target='en')
sentimentAnalysisVader = SentimentIntensityAnalyzer()
spacy.load('en_core_web_sm') #may work better with a bigger model

#
#   Main loop over narrative files for processing
#
index = 0

def run(story):
    directory_input=story
    if story == 'shapesphysical' or story == 'shapessocial':
        directory_text = "timestamps/" + story
    else:
        directory_text = "text_to_be_segmented/" + story
    directory=story
    print(f"{bcolors.OKCYAN}==========================================================================={bcolors.END}")
    print(f"{bcolors.OKBLUE} Processing "+story+ f" narrarative for sentence foundations and timestamps{bcolors.END}")
    print(f"{bcolors.OKCYAN}==========================================================================={bcolors.END}")
    for filename in os.listdir("./text/"+directory_text):
        if filename.endswith('.txt'):
            with open("./text/"+directory_text + "/" + filename, encoding='utf-8') as f1:
                lines = f1.read()
                f2 = open(os.path.join('my_temp_file'), 'w', encoding='utf-8')
                f2.write(lines)
                f2.close()
            file = open(os.path.join('my_temp_file'), 'r', encoding='utf-8', errors='ignore')

            exp2_text = file.read()
            sentences = exp2_text.split('. ') # do this so you can keep track of the original length of text for later

            index = 0
            for sentence in sentences:
                sentiment_dict = sentimentAnalysisVader.polarity_scores(sentence)
                print(f"{bcolors.OKCYAN}Sentiment for {bcolors.END}" + sentence)
                neg = sentiment_dict['neg'] * 100
                pos = sentiment_dict['pos'] * 100
                neu = sentiment_dict['neu'] * 100
                com = sentiment_dict['compound'] * 100

                # Parse with MFT dictionary and MAC dictionary
                tempDf = pd.DataFrame([sentence])
                #tempDf.to_csv('emfdTemp.csv', index=False, header=False)
                #tempDf = pd.read_csv('emfdTemp.csv', header=None)
                length = len(tempDf)
                print(f"{bcolors.OKCYAN}MAC and MFT scores for {bcolors.END}" + sentence)
                eMFD_df_all = emfd_score_docs(tempDf, 'emfd', 'all', 'bow', 'vice-virtue', length)
                eMAC_df_all = emac_score_docs(tempDf, 'emac', 'all', 'bow', 'vice-virtue', length)

                # MFT virtue-vice bow all
                mft_virtue_vice_all = {
                    "MFT_a_care_virtue" : eMFD_df_all['care.virtue'].values[0],
                    "MFT_a_fairness_virtue" : eMFD_df_all['fairness.virtue'].values[0],
                    "MFT_a_loyalty_virtue" : eMFD_df_all['loyalty.virtue'].values[0],
                    "MFT_a_authority_virtue" : eMFD_df_all['authority.virtue'].values[0],
                    "MFT_a_sanctity_virtue" : eMFD_df_all['sanctity.virtue'].values[0],
                    "MFT_a_care_vice" : eMFD_df_all['care.vice'].values[0],
                    "MFT_a_fairness_vice" : eMFD_df_all['fairness.vice'].values[0],
                    "MFT_a_loyalty_vice" : eMFD_df_all['loyalty.vice'].values[0],
                    "MFT_a_authority_vice" : eMFD_df_all['authority.vice'].values[0],
                    "MFT_a_sanctity_vice" : eMFD_df_all['sanctity.vice'].values[0],
                    "MFT_a_moral_nonmoral": eMFD_df_all['moral_nonmoral_ratio'].values[0]
                }

                # MAC virtue-vice bow all
                mac_virtue_vice_all = {
                    "MAC_a_fairness_virtue" : eMAC_df_all['fairness.virtue'].values[0],
                    "MAC_a_group_virtue" : eMAC_df_all['group.virtue'].values[0],
                    "MAC_a_deference_virtue" : eMAC_df_all['deference.virtue'].values[0],
                    "MAC_a_heroism_virtue" : eMAC_df_all['heroism.virtue'].values[0],
                    "MAC_a_reciprocity_virtue" : eMAC_df_all['reciprocity.virtue'].values[0],
                    "MAC_a_family_virtue" : eMAC_df_all['family.virtue'].values[0],
                    "MAC_a_property_virtue" : eMAC_df_all['property.virtue'].values[0],
                    "MAC_a_fairness_vice" : eMAC_df_all['fairness.vice'].values[0],
                    "MAC_a_group_vice" : eMAC_df_all['group.vice'].values[0],
                    "MAC_a_deference_vice" : eMAC_df_all['deference.vice'].values[0],
                    "MAC_a_heroism_vice" : eMAC_df_all['heroism.vice'].values[0],
                    "MAC_a_reciprocity_vice" : eMAC_df_all['reciprocity.vice'].values[0],
                    "MAC_a_family_vice" : eMAC_df_all['family.vice'].values[0],
                    "MAC_a_property_vice" : eMAC_df_all['property.vice'].values[0],
                    "MAC_a_moral_nonmoral" : eMAC_df_all['moral_nonmoral_ratio'].values[0]
                }

                merged = { **mft_virtue_vice_all , **mac_virtue_vice_all }

                if index == 0:
                    dataFrameForSaving = pd.DataFrame.from_dict(merged, orient='index').transpose()
                    dataFrameForSaving.insert(0, "full_text", filename)
                    dataFrameForSaving.insert(0, "sentence", sentence)
                elif index > 0 and index < len(sentences):
                    newIndex = len(dataFrameForSaving)
                    dataFrameForSaving.loc[newIndex] = merged
                    dataFrameForSaving.loc[newIndex, "full_text"] = filename
                    dataFrameForSaving.loc[newIndex, 'sentence'] = sentence
                index += 1
#
#   This will add timestamps (per sentence)
#
    sentenceStart = 0.0
    sentenceEnd = 0.0
    # load file with word timestamps for the text being analyzed, if it exists
    if os.path.exists('./text/timestamps/' +directory_input +"/"+ directory_input + '_transcription_per_word_x.csv'):
        print(f"{bcolors.OKCYAN}Reading in audio transcription with time stamps csv file (./text/timestamps/" +directory_input +"/" + directory_input + f"_transcription_per_word_x.csv)...{bcolors.END}")
        wordTimestamps_df = pd.read_csv('./text/timestamps/' +directory_input +"/"+ directory_input + '_transcription_per_word_x.csv', header=None)
    else:
        print(f"{bcolors.FAIL}Per word timestamps in{bcolors.END} ./text/timestamps/"+directory_input +"/" + directory_input + f"_transcription_per_word_x.csv {bcolors.FAIL}FAILED TO LOAD!{bcolors.END}")

    t_index = 1
    t_sentence = ""
    t = 0
    counter = 0
    for s in sentences:
        while levenshtein.ratio(t_sentence, s) < 100:
            print("Initial fuzz: " + str(levenshtein.ratio(t_sentence, s)))
            t_sentence = t_sentence + " " + wordTimestamps_df.loc[t_index + t, 1]
            if len(wordTimestamps_df) == t_index + t or len(wordTimestamps_df) == t_index + t + 1:
                print(f"!Match between {bcolors.WARNING}<" + s + f">{bcolors.END} and speech-to-text transcription: {bcolors.OKGREEN}" + t_sentence + f"{bcolors.END} using " + str(t) + " transcription words.")
                dataFrameForSaving.loc[counter, 'start'] = wordTimestamps_df.loc[t_index][2]
                dataFrameForSaving.loc[counter, 'end'] = wordTimestamps_df.loc[t_index + t][3]
                t_sentence = ""
                t += 1
                break
            if levenshtein.ratio(t_sentence, s) >= levenshtein.ratio(t_sentence + " " + wordTimestamps_df.loc[t_index + t + 1, 1], s) \
                and levenshtein.ratio(t_sentence, s) >= levenshtein.ratio(t_sentence + " " + wordTimestamps_df.loc[t_index + t + 2, 1], s):
                print("Lookahead fuzz 1: " + str(levenshtein.ratio(s, t_sentence + " " + wordTimestamps_df.loc[t_index + t + 1, 1])))
                print("Lookahead fuzz 2: " + str(levenshtein.ratio(s, t_sentence + " " + wordTimestamps_df.loc[t_index + t + 2, 1])))
                print(f"Match between {bcolors.WARNING}<" + s + f">{bcolors.END} and speech-to-text transcription: {bcolors.OKGREEN}" + t_sentence + f"{bcolors.END} using " + str(t) + " transcription words.")
                dataFrameForSaving.loc[counter, 'start'] = wordTimestamps_df.loc[t_index][2]
                dataFrameForSaving.loc[counter, 'end'] = wordTimestamps_df.loc[t_index + t][3]
                t_sentence = ""
                t += 1
                break
            else:
                t += 1
        t_index = t_index + t
        counter += 1
        t = 0
        t_sentence = ""

    # # Save dataframe with both sentence and segment scores to xlsx
    #dataFrameForSaving.to_excel('sentence_segment_MFT_MAC.xlsx')
    dataFrameForSaving.to_excel("./text/foundationScores/" + directory_input + '_MFT_MAC.xlsx')

    #Cleaning up temp files
    file.close()
    os.remove('my_temp_file')

    file2 = Path('emfdTemp.csv')
    try:
        my_abs_path = file2.resolve(strict=True)
    except FileNotFoundError:
        print( "Don't have to clean up empfdTemp.csv")
    else:
        os.remove('emfdTemp.csv')

# UNCOMMENT THIS TO RUN RUN MANUALLY AND ADD NAME OF STORY THROUGH PROMPT
#

# story = input("Enter the story you wish to analyze: ")
# run(story)