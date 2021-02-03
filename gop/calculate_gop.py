import pandas as pd
import pickle
import os
from utils import *
from gop import *
from scipy.special import softmax
from kaldiio import ReadHelper

if not os.path.exists('phones_pure_epa.pickle'):
    generate_df_phones_pure()

if not os.path.exists('alignments.pickle'):
    generate_alignments_df()

df_phones_pure = pd.read_pickle('phones_pure_epa.pickle')
df_phones_pure = df_phones_pure.reset_index()

alignments = pd.read_pickle('alignments.pickle')
alignments = pd.DataFrame.from_dict(alignments)

gop = {}
with ReadHelper('ark:loglikes.ark') as reader:
    for key, loglikes in reader:
        
        loglikes = softmax(np.array(loglikes))
        df_scores = pd.DataFrame(alignments.loc[:,key]).transpose()
        df_scores['p'] = [loglikes]
        gop[key] = gop_robust_with_matrix(df_scores, df_phones_pure, 6024, 1, [], [])        
        #Aca le estoy pasando los utt de a uno al gop, no estoy aprovechando que es matricial, despues deberia cambiarlo

with open('gop_epa.pickle', 'wb') as handle:
    pickle.dump(gop, handle, protocol=pickle.HIGHEST_PROTOCOL)

gop_epa = pd.read_pickle('gop_epa.pickle')

gop_output_file = open('gop_epa.txt', 'w+')

for logid, score in gop_epa.items():
    score = score[0]
    phones = score['phones_pure']
    gop = score['gop']
    
    if len(phones) != len(gop):
        raise Exception("Phones and gop list lengths do not match.")
    
    gop_output_file.write(logid)
    for i in range(len(phones)):
        gop_output_file.write(' [ ' + phones[i] + ' ' + str(gop[i]) + ' ] ')
    gop_output_file.write('\n')