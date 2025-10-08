import glob
from contextlib import nullcontext
from pathlib import Path
from thefuzz import fuzz
import spacy
import csv
import sys
import os

from openpyxl.reader.excel import load_workbook
from scipy.optimize import direct

#has to be spacy 3.4
#typing-extensions has to be 4.4
print("spaCy version:", spacy.__version__)
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from deep_translator import GoogleTranslator
import pandas as pd
#has to be pandas 1.5.3
#also install scikit-learn 1.3
#also install openpyxl
print("pandas version:", pd.__version__)
from emfdscore.scoring import score_docs as emfd_score_docs
from emacscore.scoring import score_docs as emac_score_docs

pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 400)

my_translator = GoogleTranslator(source='auto', target='en')
my_nl_translator = GoogleTranslator(source='auto', target='nl')

sentimentAnalysisVader = SentimentIntensityAnalyzer()

print("---------------------------------------")
print("WILL YOU BE CREATING SEGMENTS FROM AN EXCEL FILE?  y/n ")
excelToProcess = str(input())
if excelToProcess == 'y':
    print("---------------------------------------")
    print("DIRECTORY WHERE EXCEL IS (root: /excel, press Enter):  ")
    directory_excel = str(input())
    if directory_excel == '':
        directory_excel = 'excel'
    else:
        directory_excel = 'excel/' + directory_excel
    print("---------------------------------------")
    print("COLUMN NAME WHERE SEGMENTS ARE:  ")
    column = str(input())
    for filenameExcel in os.listdir("./" + directory_excel):
        if filenameExcel.endswith('.xlsx'):
            df = pd.read_excel('./'+directory_excel+'/'+filenameExcel)
            df.dropna(subset=[column], inplace=True)
            #df = pd.read_excel(str(filenameExcel))
            segments = df[column].tolist()
            # print( segments )

print("---------------------------------------")
print("DIRECTORY NAME FOR TEXT TO PROCESS (root: /text_to_be_segmented, press Enter):  ")
directory_input = str(input())
if directory_input == '':
    directory_text = 'text_to_be_segmented'
else:
    directory_text = 'text_to_be_segmented/' + directory_input
print("---------------------------------------")
print("DIRECTORY NAME FOR SEGMENTS TO PROCESS (root: /segments, press Enter):  ")
directory = str(input())
if directory == '':
    directory = 'segments'
else:
    subDirectory = directory
    directory = 'segments/' + directory
    index = 0
    if excelToProcess == 'y':
        for segment in segments:
            index += 1
            f2 = open(os.path.join(directory+'/'+filenameExcel.split('.')[0]+'_'+subDirectory+str(index)+'.txt'), 'w', encoding='utf-8')
            f2.write(str(segment))
            f2.close()
print("---------------------------------------")
print("MAC or MFT FOUNDATION TO RANK WITH OR 'ALL' FOR MFT/MAC SCORES FOR HIGHEST FOUNDATION PER UNIT (e.g., mac_reciprocity_vice):  ")
print("MFT: mft_harm_(care/vice), mft_fairness_(care/vice), mft_loyalty_(care/vice), mft_authority_(care/vice), mft_sanctity_(care/vice)", )
print("MAC: mac_fairness_(care/vice), mac_group_(care/vice), mac_deference_(care/vice), mac_heroism_(care/vice), mac_reciprocity_(care/vice), mac_family_(care/vice), mac_property_(care/vice)")
foundation = str(input())
print("---------------------------------------")
print("PRECISION (for default: 4, press Enter):  ")
decimal = str(input())
if decimal == '':
    decimal = 4
decimal = int(decimal)
print("---------------------------------------")
print("OUTPUT STYLE: 1 - console; 2 -  filename_PRECISION.txt (default 2, press Enter):")
outputStyle = str(input())
if outputStyle == '':
    outputStyle = 2
outputStyle = int(outputStyle)
originalSysOutput = sys.stdout

# Turn file into individual sentences and score them with MAC and MFT; whatever dictionary you put into 'merge' will be in .xslx
exp2_text = ""

for filename in os.listdir("./"+directory_text):
    if filename.endswith('.txt'):
        with open("./"+directory_text + "/" + filename, encoding='utf-8') as f1:
            lines = f1.read()
            f2 = open(os.path.join('my_temp_file'), 'w', encoding='utf-8')
            f2.write(lines)
            f2.close()
        file = open(os.path.join('my_temp_file'), 'r', encoding='utf-8', errors='ignore')
        exp2_text = file.read()
        sys.stdout = originalSysOutput
        #print( exp2_text )
        if outputStyle == 1:
            print("NO OUTPUT FILES!")
        elif outputStyle == 2:
            if not os.path.exists('./_processed/'+directory_text+"/sentences"):
                os.makedirs('./_processed/'+directory_text+"/sentences")
            if not os.path.exists('./_processed/' + directory_text + "/segments"):
                os.makedirs('./_processed/' + directory_text + "/segments")

        sentences = exp2_text.split('. ')

        index = 0
    #     for sentence in sentences:
    #         #sys.stdout = open('./sentences/' + filename.split('.')[0] + "_" + str(index) + '_' + str(decimal) + "_" + foundation + '.txt', "w")
    #         #sys.stdout = open('./_processed/'+directory_text+'/sentences/'+ "scored"+filename.split('.')[0] + "_" + str(index) + '_' + str(decimal) + "_" + foundation + '.txt', "w")
    #         exp2_text = sentence
    #         #print( exp2_text )
    #         sentiment_dict = sentimentAnalysisVader.polarity_scores(exp2_text)
    #
    #         neg = sentiment_dict['neg'] * 100
    #         pos = sentiment_dict['pos'] * 100
    #         neu = sentiment_dict['neu'] * 100
    #         com = sentiment_dict['compound'] * 100
    #
    #         # Parse with MFT dictionary and MAC dictionary
    #         tempDf = pd.DataFrame([exp2_text])
    #         #tempDf.to_csv('emfdTemp.csv', index=False, header=False)
    #         #tempDf = pd.read_csv('emfdTemp.csv', header=None)
    #         length = len(tempDf)
    #
    #         eMFD_df = emfd_score_docs(tempDf, 'emfd', 'single', 'bow', 'vice-virtue', length)
    #         eMAC_df = emac_score_docs(tempDf, 'emac', 'single', 'bow', 'vice-virtue', length)
    #         eMFD_df_all = emfd_score_docs(tempDf, 'emfd', 'all', 'bow', 'vice-virtue', length)
    #         eMAC_df_all = emac_score_docs(tempDf, 'emac', 'all', 'bow', 'vice-virtue', length)
    #
    #         eMFD_df.drop( columns=['moral_nonmoral_ratio', 'f_var'],  inplace=True)
    #         eMFD_df_all.drop( columns=['moral_nonmoral_ratio', 'f_var'], inplace=True)
    #
    #         eMAC_df.drop( columns=['moral_nonmoral_ratio', 'f_var'], inplace=True)
    #         eMAC_df_all.drop( columns=['moral_nonmoral_ratio', 'f_var'], inplace=True)
    #
    #         # MFT virtue-vice bow per word
    #         mft_virtue_vice_word = {
    #             "MFT_w_care_virtue" : eMFD_df['care.virtue'].values[0] ,
    #             "MFT_w_fairness_virtue" : eMFD_df['fairness.virtue'].values[0] ,
    #             "MFT_w_loyalty_virtue" : eMFD_df['loyalty.virtue'].values[0] ,
    #             "MFT_w_authority_virtue" : eMFD_df['authority.virtue'].values[0] ,
    #             "MFT_w_sanctity_virtue" : eMFD_df['sanctity.virtue'].values[0] ,
    #             "MFT_w_care_vice" : eMFD_df['care.vice'].values[0] ,
    #             "MFT_w_fairness_vice" : eMFD_df['fairness.vice'].values[0] ,
    #             "MFT_w_loyalty_vice" : eMFD_df['loyalty.vice'].values[0] ,
    #             "MFT_w_authority_vice" : eMFD_df['authority.vice'].values[0] ,
    #             "MFT_w_sanctity_vice" : eMFD_df['sanctity.vice'].values[0]
    #         }
    #
    #         # MAC virtue-vice bow per word
    #         mac_virtue_vice_word ={
    #             "MAC_w_fairness_virtue" : eMAC_df['fairness.virtue'].values[0],
    #             "MAC_w_group_virtue" : eMAC_df['group.virtue'].values[0],
    #             "MAC_w_deference" : eMAC_df['deference.virtue'].values[0],
    #             "MAC_w_heroism" : eMAC_df['heroism.virtue'].values[0],
    #             "MAC_w_reciprocity" : eMAC_df['reciprocity.virtue'].values[0],
    #             "MAC_w_family_virtue" : eMAC_df['family.virtue'].values[0],
    #             "MAC_w_property_virtue" : eMAC_df['property.virtue'].values[0],
    #             "MAC_w_fairness_vice" : eMAC_df['fairness.vice'].values[0],
    #             "MAC_w_group_vice" : eMAC_df['group.vice'].values[0],
    #             "MAC_w_deference_vice" : eMAC_df['deference.vice'].values[0],
    #             "MAC_w_heroism_vice" : eMAC_df['heroism.vice'].values[0],
    #             "MAC_w_reciprocity_vice" : eMAC_df['reciprocity.vice'].values[0],
    #             "MAC_w_family_vice" : eMAC_df['family.vice'].values[0],
    #             "MAC_w_property_vice" : eMAC_df['property.vice'].values[0]
    #         }
    #
    #         # MFT virtue-vice bow all
    #         mft_virtue_vice_all = {
    #             "MFT_a_care_virtue" : eMFD_df_all['care.virtue'].values[0],
    #             "MFT_a_fairness_virtue" : eMFD_df_all['fairness.virtue'].values[0],
    #             "MFT_a_loyalty_virtue" : eMFD_df_all['loyalty.virtue'].values[0],
    #             "MFT_a_authority_virtue" : eMFD_df_all['authority.virtue'].values[0],
    #             "MFT_a_sanctity_virtue" : eMFD_df_all['sanctity.virtue'].values[0],
    #             "MFT_a_care_vice" : eMFD_df_all['care.vice'].values[0],
    #             "MFT_a_fairness_vice" : eMFD_df_all['fairness.vice'].values[0],
    #             "MFT_a_loyalty_vice" : eMFD_df_all['loyalty.vice'].values[0],
    #             "MFT_a_authority_vice" : eMFD_df_all['authority.vice'].values[0],
    #             "MFT_a_sanctity_vice" : eMFD_df_all['sanctity.vice'].values[0]
    #         }
    #
    #         # MAC virtue-vice bow all
    #         mac_virtue_vice_all = {
    #             "MAC_a_fairness_virtue" : eMAC_df_all['fairness.virtue'].values[0],
    #             "MAC_a_group_virtue" : eMAC_df_all['group.virtue'].values[0],
    #             "MAC_a_deference_virtue" : eMAC_df_all['deference.virtue'].values[0],
    #             "MAC_a_heroism_virtue" : eMAC_df_all['heroism.virtue'].values[0],
    #             "MAC_a_reciprocity_virtue" : eMAC_df_all['reciprocity.virtue'].values[0],
    #             "MAC_a_family_virtue" : eMAC_df_all['family.virtue'].values[0],
    #             "MAC_a_property_virtue" : eMAC_df_all['property.virtue'].values[0],
    #             "MAC_a_fairness_vice" : eMAC_df_all['fairness.vice'].values[0],
    #             "MAC_a_group_vice" : eMAC_df_all['group.vice'].values[0],
    #             "MAC_a_deference_vice" : eMAC_df_all['deference.vice'].values[0],
    #             "MAC_a_heroism_vice" : eMAC_df_all['heroism.vice'].values[0],
    #             "MAC_a_reciprocity_vice" : eMAC_df_all['reciprocity.vice'].values[0],
    #             "MAC_a_family_vice" : eMAC_df_all['family.vice'].values[0],
    #             "MAC_a_property_vice" : eMAC_df_all['property.vice'].values[0]
    #         }
    #
    #         merged = { **mft_virtue_vice_all , **mac_virtue_vice_all }
    #
    #         if index == 0:
    #             dataFrameForSaving = pd.DataFrame.from_dict(merged, orient='index').transpose()
    #             dataFrameForSaving.insert(0, "full_text", filename)
    #             dataFrameForSaving.insert(0, "sentence", sentence)
    #         elif index > 0 and index < len(sentences):
    #             newIndex = len(dataFrameForSaving)
    #             dataFrameForSaving.loc[newIndex] = merged
    #             dataFrameForSaving.loc[newIndex, "full_text"] = filename
    #             dataFrameForSaving.loc[newIndex, 'sentence'] = sentence
    #         index += 1
    #
    # # Look through segments in <segments> directory and match them with sentences in dataframe that contains sentence scores
    # index = 0 #this is for keeping track of sentence index as we go through the segments
    # fragmentFiles = os.listdir("./" + directory)
    # fragmentFiles.sort(reverse=True)
    # fragmentFiles.sort(key=str.lower)
    # fragmentFiles.sort(key=len)
    # print("Reading fragment files: " + str(fragmentFiles))
    #
    # segmentIndex = 0
    #
    # for segments in fragmentFiles:
    #     if segments.endswith('.txt'):
    #         segmentIndex = segmentIndex+1
    #         print( "Reading text from segment file: " + str(segments) + " which is " + str(segmentIndex) + " of " + str(len(fragmentFiles)))
    #         with open("./" + directory + "/" + segments, encoding='utf-8') as f1:
    #             segment = f1.read()
    #             f2 = open(os.path.join('my_temp_file'), 'w', encoding='utf-8')
    #             f2.write(segment)
    #             f2.close()
    #         file = open(os.path.join('my_temp_file'), 'r', encoding='utf-8')
    #         exp2_text = file.read()
    #
    #         # Parse with MFT dictionary and MAC dictionary
    #         tempDf = pd.DataFrame([exp2_text])
    #         print( "Reading fragment from temp file: " + str(tempDf.values) )
    #         tempDf.to_csv('emfdTemp.csv', index=False, header=False)
    #         tempDf = pd.read_csv('emfdTemp.csv', header=None)
    #         length = len(tempDf)
    #
    #         eMFD_df = emfd_score_docs(tempDf, 'emfd', 'single', 'bow', 'vice-virtue', length)
    #         eMAC_df = emac_score_docs(tempDf, 'emac', 'single', 'bow', 'vice-virtue', length)
    #         eMFD_df_all = emfd_score_docs(tempDf, 'emfd', 'all', 'bow', 'vice-virtue', length)
    #         eMAC_df_all = emac_score_docs(tempDf, 'emac', 'all', 'bow', 'vice-virtue', length)
    #
    #         eMFD_df.drop(columns=['moral_nonmoral_ratio', 'f_var'], inplace=True)
    #         eMFD_df_all.drop(columns=['moral_nonmoral_ratio', 'f_var'], inplace=True)
    #
    #         eMAC_df.drop(columns=['moral_nonmoral_ratio', 'f_var'], inplace=True)
    #         eMAC_df_all.drop(columns=['moral_nonmoral_ratio', 'f_var'], inplace=True)
    #
    #         # MFT virtue-vice bow per word
    #         mft_virtue_vice_word = {
    #             "seg_MFT_w_care_virtue": eMFD_df['care.virtue'].values[0],
    #             "seg_MFT_w_fairness_virtue": eMFD_df['fairness.virtue'].values[0],
    #             "seg_MFT_w_loyalty_virtue": eMFD_df['loyalty.virtue'].values[0],
    #             "seg_MFT_w_authority_virtue": eMFD_df['authority.virtue'].values[0],
    #             "seg_MFT_w_sanctity_virtue": eMFD_df['sanctity.virtue'].values[0],
    #             "seg_MFT_w_care_vice": eMFD_df['care.vice'].values[0],
    #             "seg_MFT_w_fairness_vice": eMFD_df['fairness.vice'].values[0],
    #             "seg_MFT_w_loyalty_vice": eMFD_df['loyalty.vice'].values[0],
    #             "seg_MFT_w_authority_vice": eMFD_df['authority.vice'].values[0],
    #             "seg_MFT_w_sanctity_vice": eMFD_df['sanctity.vice'].values[0]
    #         }
    #
    #         # MAC virtue-vice bow per word
    #         mac_virtue_vice_word = {
    #             "seg_MAC_w_fairness_virtue": eMAC_df['fairness.virtue'].values[0],
    #             "seg_MAC_w_group_virtue": eMAC_df['group.virtue'].values[0],
    #             "seg_MAC_w_deference": eMAC_df['deference.virtue'].values[0],
    #             "seg_MAC_w_heroism": eMAC_df['heroism.virtue'].values[0],
    #             "seg_MAC_w_reciprocity": eMAC_df['reciprocity.virtue'].values[0],
    #             "seg_MAC_w_family_virtue": eMAC_df['family.virtue'].values[0],
    #             "seg_MAC_w_property_virtue": eMAC_df['property.virtue'].values[0],
    #             "seg_MAC_w_fairness_vice": eMAC_df['fairness.vice'].values[0],
    #             "seg_MAC_w_group_vice": eMAC_df['group.vice'].values[0],
    #             "seg_MAC_w_deference_vice": eMAC_df['deference.vice'].values[0],
    #             "seg_MAC_w_heroism_vice": eMAC_df['heroism.vice'].values[0],
    #             "seg_MAC_w_reciprocity_vice": eMAC_df['reciprocity.vice'].values[0],
    #             "seg_MAC_w_family_vice": eMAC_df['family.vice'].values[0],
    #             "seg_MAC_w_property_vice": eMAC_df['property.vice'].values[0]
    #         }
    #
    #         # MFT virtue-vice bow all
    #         mft_virtue_vice_all = {
    #             "seg_MFT_a_care_virtue": eMFD_df_all['care.virtue'].values[0],
    #             "seg_MFT_a_fairness_virtue": eMFD_df_all['fairness.virtue'].values[0],
    #             "seg_MFT_a_loyalty_virtue": eMFD_df_all['loyalty.virtue'].values[0],
    #             "seg_MFT_a_authority_virtue": eMFD_df_all['authority.virtue'].values[0],
    #             "seg_MFT_a_sanctity_virtue": eMFD_df_all['sanctity.virtue'].values[0],
    #             "seg_MFT_a_care_vice": eMFD_df_all['care.vice'].values[0],
    #             "seg_MFT_a_fairness_vice": eMFD_df_all['fairness.vice'].values[0],
    #             "seg_MFT_a_loyalty_vice": eMFD_df_all['loyalty.vice'].values[0],
    #             "seg_MFT_a_authority_vice": eMFD_df_all['authority.vice'].values[0],
    #             "seg_MFT_a_sanctity_vice": eMFD_df_all['sanctity.vice'].values[0]
    #         }
    #
    #         # MAC virtue-vice bow all
    #         mac_virtue_vice_all = {
    #             "seg_MAC_a_fairness_virtue": eMAC_df_all['fairness.virtue'].values[0],
    #             "seg_MAC_a_group_virtue": eMAC_df_all['group.virtue'].values[0],
    #             "seg_MAC_a_deference_virtue": eMAC_df_all['deference.virtue'].values[0],
    #             "seg_MAC_a_heroism_virtue": eMAC_df_all['heroism.virtue'].values[0],
    #             "seg_MAC_a_reciprocity_virtue": eMAC_df_all['reciprocity.virtue'].values[0],
    #             "seg_MAC_a_family_virtue": eMAC_df_all['family.virtue'].values[0],
    #             "seg_MAC_a_property_virtue": eMAC_df_all['property.virtue'].values[0],
    #             "seg_MAC_a_fairness_vice": eMAC_df_all['fairness.vice'].values[0],
    #             "seg_MAC_a_group_vice": eMAC_df_all['group.vice'].values[0],
    #             "seg_MAC_a_deference_vice": eMAC_df_all['deference.vice'].values[0],
    #             "seg_MAC_a_heroism_vice": eMAC_df_all['heroism.vice'].values[0],
    #             "seg_MAC_a_reciprocity_vice": eMAC_df_all['reciprocity.vice'].values[0],
    #             "seg_MAC_a_family_vice": eMAC_df_all['family.vice'].values[0],
    #             "seg_MAC_a_property_vice": eMAC_df_all['property.vice'].values[0]
    #         }
    #
    #         for sentence in (exp2_text.split(". ")):
    #             print( "Segment sentence: " + sentence )
    #             if  fuzz.ratio(dataFrameForSaving.iloc[index]['sentence'], sentence) > 65:
    #                 print(" -- match between segment and sentence")
    #                 #print(dataFrameForSaving.iloc[index]['sentence'])
    #                 dataFrameForSaving.loc[index, 'segment'] = str(segments)
    #                 dataFrameForSaving.loc[index, 'seg_sentence'] = str(sentence)
    #                 dataFrameForSaving.loc[index, 'trans_sentence'] = str(  )
    #                 dataFrameForSaving.loc[index, 'start_sentence'] = str(  )
    #                 dataFrameForSaving.loc[index, 'end_sentence'] = str( )
    #                 dataFrameForSaving.loc[index, "seg_MFT_a_care_virtue"] = mft_virtue_vice_all.get("seg_MFT_a_care_virtue")
    #                 dataFrameForSaving.loc[index, "seg_MFT_a_fairness_virtue"] = mft_virtue_vice_all.get("seg_MFT_a_fairness_virtue")
    #                 dataFrameForSaving.loc[index, "seg_MFT_a_loyalty_virtue"] = mft_virtue_vice_all.get("seg_MFT_a_loyalty_virtue")
    #                 dataFrameForSaving.loc[index, "seg_MFT_a_authority_virtue"] = mft_virtue_vice_all.get("seg_MFT_a_authority_virtue")
    #                 dataFrameForSaving.loc[index, "seg_MFT_a_sanctity_virtue"] = mft_virtue_vice_all.get("seg_MFT_a_sanctity_virtue")
    #                 dataFrameForSaving.loc[index, "seg_MFT_a_care_vice"] = mft_virtue_vice_all.get("seg_MFT_a_care_vice")
    #                 dataFrameForSaving.loc[index, "seg_MFT_a_fairness_vice"] = mft_virtue_vice_all.get("seg_MFT_a_fairness_vice")
    #                 dataFrameForSaving.loc[index, "seg_MFT_a_loyalty_vice"] = mft_virtue_vice_all.get("seg_MFT_a_loyalty_vice")
    #                 dataFrameForSaving.loc[index, "seg_MFT_a_authority_vice"] = mft_virtue_vice_all.get("seg_MFT_a_authority_vice")
    #                 dataFrameForSaving.loc[index, "seg_MFT_a_sanctity_vice"] = mft_virtue_vice_all.get("seg_MFT_a_sanctity_vice")
    #                 dataFrameForSaving.loc[index, "seg_MAC_a_fairness_virtue"] = mac_virtue_vice_all.get("seg_MAC_a_fairness_virtue")
    #                 dataFrameForSaving.loc[index, "seg_MAC_a_group_virtue"] = mac_virtue_vice_all.get("seg_MAC_a_group_virtue")
    #                 dataFrameForSaving.loc[index, "seg_MAC_a_deference_virtue"] = mac_virtue_vice_all.get("seg_MAC_a_deference_virtue")
    #                 dataFrameForSaving.loc[index, "seg_MAC_a_heroism_virtue"] = mac_virtue_vice_all.get("seg_MAC_a_heroism_virtue")
    #                 dataFrameForSaving.loc[index, "seg_MAC_a_reciprocity_virtue"] = mac_virtue_vice_all.get("seg_MAC_a_reciprocity_virtue")
    #                 dataFrameForSaving.loc[index, "seg_MAC_a_family_virtue"] = mac_virtue_vice_all.get("seg_MAC_a_family_virtue")
    #                 dataFrameForSaving.loc[index, "seg_MAC_a_property_virtue"] = mac_virtue_vice_all.get("seg_MAC_a_property_virtue")
    #                 dataFrameForSaving.loc[index, "seg_MAC_a_fairness_vice"] = mac_virtue_vice_all.get("seg_MAC_a_fairness_vice")
    #                 dataFrameForSaving.loc[index, "seg_MAC_a_group_vice"] = mac_virtue_vice_all.get("seg_MAC_a_group_vice")
    #                 dataFrameForSaving.loc[index, "seg_MAC_a_deference_vice"] = mac_virtue_vice_all.get("seg_MAC_a_deference_vice")
    #                 dataFrameForSaving.loc[index, "seg_MAC_a_heroism_vice"] = mac_virtue_vice_all.get("seg_MAC_a_heroism_vice")
    #                 dataFrameForSaving.loc[index, "seg_MAC_a_reciprocity_vice"] = mac_virtue_vice_all.get("seg_MAC_a_reciprocity_vice")
    #                 dataFrameForSaving.loc[index, "seg_MAC_a_family_vice"] = mac_virtue_vice_all.get("seg_MAC_a_family_vice")
    #                 dataFrameForSaving.loc[index, "seg_MAC_a_property_vice"] = mac_virtue_vice_all.get("seg_MAC_a_property_vice")
    #
    #                 index += 1


#load file with word timestamps for the text being analyzed, if it exists
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

transcriptionSentences = []
transcriptionSentence = ""
sentenceStart = 0.0
sentenceEnd = 0.0
if os.path.exists('./word_timestamps/'+directory_input+'_transcription.csv'):
   print("Reading in audio transcription with time stamps csv file (./word_timestamps/"+directory_input+"_transcription.csv)...")
   wordTimestamps_df = pd.read_csv('./word_timestamps/' + directory_input + '_transcription.csv', header=None)
    #creates a list of tuples
   for word in wordTimestamps_df[1:].itertuples():
       if transcriptionSentence == "": sentenceStart = float(word[3])
       transcriptionSentence = transcriptionSentence + str(word[2]) + " "
       if transcriptionSentence.endswith('. '):
           sentenceEnd = float(word[4])
           sentenceTimestamps = ((transcriptionSentence, sentenceStart, sentenceEnd))
           transcriptionSentences.append( (sentenceTimestamps)  )
           transcriptionSentence = ""
else:
   print(f"The audio transcription{bcolors.WARNING}"+ directory_input+f"{bcolors.END}_transcription.csv file does not exist in /word_timestamps/")

t_index = 1
t_sentence = ""
t = 0
for sentence in (exp2_text.split(". ")):
    print("Transcript sentence: " + sentence)
    #print( t_index )
    for tuple in wordTimestamps_df[t_index:].itertuples():
        print( tuple[2] )
        t_sentence = t_sentence + str(tuple[2]) + " "
        if fuzz.token_sort_ratio(t_sentence, sentence) > 96:
            t += 1
            t_sentence2 = t_sentence + wordTimestamps_df.loc[t_index+t][2]
            t_sentence3 = t_sentence2 + wordTimestamps_df.loc[t_index + t+1][2]

            if fuzz.token_sort_ratio(t_sentence, sentence) < fuzz.token_sort_ratio(t_sentence2, sentence):
                t += 1
                print(f"Match between {bcolors.WARNING}<" + sentence + f">{bcolors.END} and speech-to-text transcription: {bcolors.OKGREEN}" + t_sentence2 + f"{bcolors.END} using " + str(t) + " transcription words.")
                t_sentence = ""
                if fuzz.token_sort_ratio(t_sentence2, sentence) < fuzz.token_sort_ratio(t_sentence3, sentence):
                    t += 1
                    print(
                        f"Match between {bcolors.WARNING}<" + sentence + f">{bcolors.END} and speech-to-text transcription: {bcolors.OKGREEN}" + t_sentence3 + f"{bcolors.END} using " + str(t) + " transcription words.")
                    t_sentence = ""
                    break
                break
            else:
                #t+=1
                print(f"Match between {bcolors.WARNING}<" + sentence + f">{bcolors.END} and speech-to-text transcription: {bcolors.OKGREEN}" + t_sentence + f"{bcolors.END} using " + str(t) + " transcription words.")
                t_sentence = ""
                break
        else:
            t=t+1
    t_index = t_index + t
    print( t_index )
    t = 0
    t_sentence = ""

        # print(dataFrameForSaving.iloc[index]['sentence'])
        # dataFrameForSaving.loc[index, 'trans_sentence'] = str()
        # dataFrameForSaving.loc[index, 'start_sentence'] = str()
        # dataFrameForSaving.loc[index, 'end_sentence'] = str()

# transcriptionSentences[index][0]
# transcriptionSentences[index][1]
# transcriptionSentences[index][2]
#
# # Save dataframe with both sentence and segment scores to xlsx
# #dataFrameForSaving.to_excel('sentence_segment_MFT_MAC.xlsx')
# dataFrameForSaving.to_excel(directory.split('.')[0] + '_segment_MFT_MAC.xlsx')