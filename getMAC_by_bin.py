import os
import warnings
import pandas as pd
warnings.filterwarnings("ignore")
import pandas as pd
print("pandas version:", pd.__version__)
from emfdscore.scoring import score_docs as emfd_score_docs
from emacscore.scoring import score_docs as emac_score_docs

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

########################################################################################################################
#   This takes text files from sentences that are used to populate design matrices for each foundation, which are
#   and then scores them using MAC dictionary
#
########################################################################################################################
foundationScores_dir = "./text/foundationScores/"
foundationBins_dir = "./text/foundationBins/"
sentences = ''
for filename in os.listdir(foundationBins_dir):
    if filename.endswith('.csv'):
        print( filename )
        sentences = pd.read_csv(foundationBins_dir + "/" + filename, sep=';', usecols=[1], header=None)
    foundation = filename.split('_')[3]
    vice_or_virtue = filename.split('_')[4].split('.')[0]
    # Parse with MFT dictionary and MAC dictionary
    tempDf = pd.DataFrame(sentences)
    tempDf.to_csv('macTemp.csv', index=False, header=False)
    tempDf = pd.read_csv('macTemp.csv', header=None)
    length = len(tempDf)
    #print(f"{bcolors.OKCYAN}MAC "+ foundation + " " + vice_or_virtue + f" for {bcolors.END}" )
    #print(sentences.to_string())
    #eMFD_df_all = emfd_score_docs(tempDf, 'emfd', 'all', 'bow', 'vice-virtue', length)
    eMAC_df_all = emac_score_docs(tempDf, 'emac', 'all', 'bow', 'vice-virtue', length)
    print( eMAC_df_all[foundation+'.'+vice_or_virtue].values[0] )
    file = open(foundationBins_dir + '/' + filename, 'a')
    file.write(str( eMAC_df_all[foundation + '.' + vice_or_virtue].values[0] ) )