import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib
from pathlib import Path
import os
from scipy.special import logsumexp

def get_pdfs_for_pure_phone(df_phones_pure, phone):
    pdfs = list(df_phones_pure.loc[(df_phones_pure['phone_pure'] == str(phone+1) )].forward_pdf)
    pdfs = pdfs + list(df_phones_pure.loc[(df_phones_pure['phone_pure'] == str(phone+1) )].self_pdf)
    pdfs = set(pdfs)
    return pdfs


def matrix_gop_robust(df_phones_pure, number_senones, batch_size):            
    pdfs_to_phone_pure_mask = []

    for phone_pure in range(0, len(list(df_phones_pure.phone_pure.unique()))):
        pdfs = get_pdfs_for_pure_phone(df_phones_pure, phone_pure)
        pdfs_to_phone_pure_file = np.zeros(number_senones)
                            
        for pdf in pdfs:
            pdfs_to_phone_pure_file[int(pdf)] = 1.0 
        
        pdfs_to_phone_pure_mask.append(pdfs_to_phone_pure_file)
                                
    pdfs_to_phone_pure_mask_3D = []

    for i in range(0, batch_size):                
        pdfs_to_phone_pure_mask_3D.append(pdfs_to_phone_pure_mask)
    
    return pdfs_to_phone_pure_mask_3D



def gop_robust_with_matrix(df_scores, df_phones_pure, number_senones, batch_size, output_gop_r, output_gop_r_max):
    
    mask_score = matrix_gop_robust(df_phones_pure, number_senones, batch_size)

    mask_score = np.array(mask_score)
    mask_score = mask_score.transpose(0, 2, 1)

    scores = df_scores.p
    scores = np.array(list(scores))

    scores_phone_pure = np.matmul(scores,mask_score)
    #return scores_phone_pure[0]
    
    logids = df_scores.index

    for j in range(0, len(df_scores)):
        phones = df_scores.phones[j]
        transitions = df_scores.transitions[j]
        logid = logids[j]

        ti = 0
        gops_r_max = []
        gops_r = []
        phones_pure = []
        
        #return scores_phone_pure[j]
        #frame_level_lpp = scores_phone_pure[j]
        #print(np.log(frame_level_lpp[0]))
        
        for i in range(0, len(transitions)):
            transitions_by_phone = transitions[i]
            tf = ti + len(transitions_by_phone) - 1     
            
            lpp = (sum(np.log(scores_phone_pure[j])[ti:tf+1]))/(tf-ti+1)
                        
            phone_pure = df_phones_pure.loc[(df_phones_pure['phone_name'] == str(phones[i]) )].phone_pure.unique()[0]
            gop_r_max = lpp[int(phone_pure)-1] - max(lpp)
            gop_r = lpp[int(phone_pure)-1]

            phones_pure.append(phone_pure)
            gops_r_max.append(gop_r_max)
            gops_r.append(gop_r)
            
            ti = tf + 1
        
        #output_gop_r_max.append({'gop': gops_r,
        #                        'phones_pure': phones_pure,
        #                        'logid': logid})

        output_gop_r.append({'gop': gops_r,'phones_pure': phones_pure, 'logid': logid})

    return output_gop_r