from contextlib import nullcontext

from mn import analysis as a
import warnings
warnings.filterwarnings("ignore")

########################################################################################################################
#   This creates text files from sentences that are used to populate design matrices for each foundation, which are
#   then used to find corresponding brain regions.  Use this to find patterns in language for those stimuli.
#
########################################################################################################################
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
        foundationBins = a.get_top_foundation_per_sentence(story, group, 0)
        for tpl in foundationBins:
            idx = tpl[0]  # idx has the 'dependent' variable
            temp = dct.get(idx, [])
            temp.append(tpl[1])
            dct[idx] = temp

        for bin in dct.keys():
            print( f'\n{bcolors.OKBLUE}'+bin+f'{bcolors.END} in ' + bcolors.OKCYAN+story+bcolors.END )
            for sentence in dct.get(bin):
                print( sentence )
        dct = {}
    # for foundation in vices_per_sentence:
    #     file = open('./text/sentenceBins/'+str(story)+'_'+foundation+'.csv', 'w')
    #
    #     for values in foundations_per_sentence:
    #         file.write(values[1]+'\n')
    #     file.close()