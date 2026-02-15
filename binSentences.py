from mn import analysis as a
import warnings
warnings.filterwarnings("ignore")

########################################################################################################################
#   This creates text files from sentences that are used to populate design matrices for each foundation, which are
#   then used to find corresponding brain regions.  Use this to find patterns in language for those stimuli.
#
########################################################################################################################
stories =                   ['21styear','tunnel']
foundations =               ['MAC_a_fairness_virtue',
                            'MAC_a_group_virtue',
                            'MAC_a_deference_virtue',
                            'MAC_a_heroism_virtue',
                            'MAC_a_reciprocity_virtue',
                            'MAC_a_family_virtue',
                            'MAC_a_property_virtue',
                            'MAC_a_fairness_vice',
                            'MAC_a_group_vice',
                            'MAC_a_deference_vice',
                            'MAC_a_heroism_vice',
                            'MAC_a_reciprocity_vice',
                            'MAC_a_family_vice',
                            'MAC_a_property_vice']

for story in stories:
    for foundation in foundations:
        file = open('./text/sentenceBins/'+str(story)+'_'+str(foundation)+'.csv', 'a')
        foundations_per_sentence = a.get_top_foundation_per_sentence(story, [foundation], 0)
        for values in foundations_per_sentence:
            file.write(values[1]+'\n')
        file.close()