import os
import warnings
import pandas as pd
warnings.filterwarnings("ignore")

########################################################################################################################
#   This creates text files from sentences that are used to populate design matrices for each foundation, which are
#   then used to find corresponding brain regions.  Use this to find patterns in language for those stimuli.
#
########################################################################################################################
foundationScores_dir = "./text/foundationScores/"
foundationBins_dir = "./text/foundationBins/"

#return a list of tuples (foundation name, sentence, value) with top foundation score
def get_top_foundation_per_sentence(story, foundations, scoring):
    story = story + "_"
    sentence_tuples = []
    for root, dirs, files in os.walk(foundationScores_dir):
        for file in files:
            if story in file and str(scoring) in file:
                print(f"{bcolors.OKBLUE}Getting top foundations per sentence in {bcolors.WARNING}%03s {bcolors.OKBLUE}for {bcolors.END}%03s" % (story[:-1], foundations) )
                df = pd.read_excel(os.path.join(root, file))
                for index, row in df.iterrows():
                    if ( row[foundations].max() > 0 ):
                        tuple = row[foundations].idxmax(), row['sentence'], row[foundations].max()
                        sentence_tuples.append(tuple)
                    # else:
                    #     tuple = 'baseline', row['sentence'] #, row[foundations].max()
                    #     sentence_tuples.append(tuple)
    return sentence_tuples

stories =                   ['21styear','tunnel']
foundationGroups =          [['MAC_a_fairness_virtue',
                            'MAC_a_group_virtue',
                            'MAC_a_deference_virtue',
                            'MAC_a_heroism_virtue',
                            'MAC_a_reciprocity_virtue',
                            'MAC_a_family_virtue',
                            'MAC_a_property_virtue'],
                             ['MAC_a_fairness_vice',
                            'MAC_a_group_vice',
                            'MAC_a_deference_vice',
                            'MAC_a_heroism_vice',
                            'MAC_a_reciprocity_vice',
                            'MAC_a_family_vice',
                            'MAC_a_property_vice']]
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

dct = {}

for story in stories:
    for group in foundationGroups:
        foundationBins = get_top_foundation_per_sentence(story, group, 0)
        for tuple in foundationBins:
            foundation = tuple[0]
            sentence = dct.get(foundation, [])
            sentence.append(str(round(tuple[2], 3))+'; '+str(tuple[1]))
            dct[foundation] = sentence

        for bin in dct.keys():
            print( f'\n{bcolors.OKBLUE}'+bin+f'{bcolors.END} in ' + bcolors.OKCYAN+story+bcolors.END )
            file = open(foundationBins_dir + str(story) + '_' + bin + '.csv', 'w')
            for sentence in dct.get(bin):
                file.write(sentence+'\n')
                print( sentence )
        dct = {}
        file.close()