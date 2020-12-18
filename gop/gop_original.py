from IPython import embed
from IPython import embed
from paiplain.pipeline import Task
import numpy as np
import pandas as pd
from batch_data_generator import batch_data_generator
from tqdm import tqdm
import joblib
from pathlib import Path
import os
from scipy.special import logsumexp


class gop(Task):
    
    def process(self, model, df_partitions, df_features, df_trans, df_phones_pure = None, df_target=None, label_names=None, target='target', task='classification'):


        if df_target is None:
            with_target = False
        else:
            with_target = True

        if with_target:
            if label_names is not None:
                if hasattr(model, 'classes_'):
                    if not (label_names == model.classes_).all():
                        raise Exception('Model label_names order not assured')
            else:
                if hasattr(model, 'classes_'):
                    label_names = model.classes_
                else:
                    raise Exception('Missing label names')

        scores = {}

        if isinstance(df_features.columns, pd.core.indexes.multi.MultiIndex):
            df_features.columns = df_features.columns.droplevel()

        df_partitions = df_partitions.fillna(False)
        

        for partition_name in df_partitions.columns:
            if partition_name != 'train' and  partition_name != 'test':
                part_idx = df_partitions[df_partitions[partition_name]].index
                if len(part_idx) > 0:
                    X = df_features.reindex(part_idx)
                    print('Evaluating', partition_name)

                    if task == 'classification_by_frame':
                        align_transitions = df_target.align_transitions                    
                        phones = df_target.phones    
                        phones_names = df_target.phones_names

                    if task == 'classification_by_frame_libri':
                        align_transitions = df_trans.align_transitions
                        phones = df_trans.phones    
                        phones_names = df_trans.phones_names

                    y = df_target[target]
                    
                    if align_transitions is None:
                        df_gop = self.predict_proba(model, X, y, df_trans, phones)
                        df_gops = self.gops(df_gop, phones, phones_names, df_trans)
                    else:
                        #df_gop, df_gop2 = self.predict_proba(model, X, y, df_trans, phones, align_transitions, df_phones_pure)
                        #df_gops = self.gops1(df_gop, phones, phones_names, df_trans, align_transitions, df_gop2, task)
                        df_gop, df_gop_robust_max, df_gop_robust, df_gop2  = self.predict_proba(model, X, y, df_trans, phones, align_transitions, df_phones_pure)
                        #df_gops = self.gops(df_gop_robust, phones, phones_names, df_trans, align_transitions, task)
        embed()
        return df_gop, df_gop_robust_max, df_gop_robust, df_gop2
    
    def log_sum_exp(self, x):
        res = []
        log_sum_exp = logsumexp(x)
        for m in range(0, len(x)):
            res.append(x[m] - log_sum_exp)
        return res

    def select_target(self, x, i_target):
        return x[i_target]

    def min(self, val1, val2):
        minimo = val2
        if(val1 < val2):
            minimo = val1

        return minimo

    def predict_proba(self, model, df_X, df_Y, df_trans, phones, align_transitions = None, df_phones_pure = None):
        bdg = batch_data_generator(df_X, df_y=None, max_length=model.max_length, batch_size=model.batch_size,
                                   in_memory=model.load_batch_data_in_memory, mask_value=model.mask_value, 
                                   padding_value=model.padding_value)
        output_gop_r_max = []
        output_gop_r = []

        output_ = []
        output_gop = []
        output_gop2 = []
        speakers = []
        waveforms_largos = []
        from scipy.special import logsumexp
         
        for i, batch_data in enumerate(tqdm(bdg)):
                         
            logid = batch_data['logids']
            fetches = model.get_fetches(batch_data, training_phase=False)

            pre_act = model.sess.run([model.tf_variables_dict['pre_activations']], fetches)[0]

            pre_act1 =  pre_act
            log_sum_exp = logsumexp(pre_act, axis=2, keepdims=True) 
            pre_act -= log_sum_exp

            output_ = []
            output_spkr = []
            aux = 0
            for i, logid_ in enumerate(logid):

                flatten = lambda l: [item for sublist in l for item in sublist]
                df_align_transitions = align_transitions[logid_]
                df_phones = phones[logid_]
                                                                        
                if len(df_Y.loc[logid_]) > model.max_length:
                    print("Cut Y " +logid_)
                    df_align_transitions = self.cut_align_transitions(df_align_transitions, model.max_length)
                    
                    #p y log p no los corto porque necesito poder multiplicarlos como matriz
                    output_.append({'p':np.exp(pre_act[i]),
                                    'log_p': pre_act[i],
                                    'pre_act': pre_act1[i],
                                    'y': df_Y.loc[logid_][0:len(flatten(df_align_transitions))],
                                    'transitions' :df_align_transitions,
                                    'phones': df_phones.loc[logid_][0:len(flatten(df_align_transitions))],
                                    'logid': logid_})
                else:
                    output_.append({'p':np.exp(pre_act[i]),
                                    'log_p': pre_act[i],
                                    'pre_act': pre_act1[i],
                                    'y': df_Y.loc[logid_],
                                    'transitions' :df_align_transitions,
                                    'phones': df_phones,
                                    'logid': logid_})
                
                if len(df_Y.loc[logid_]) != len(phones[logid_]):
                    print("no coinciden las longitudes de senones y phones")
                    embed()

                if len(df_Y.loc[logid_]) > model.max_length and logid_ not in waveforms_largos:
                    waveforms_largos.append(logid_)
                    print("corto longitudes de scores: predict_proba" + logid_)

            df_scores = pd.DataFrame(output_).set_index('logid')


            if align_transitions is None:
                self.gop(df_scores, phones, output_gop)
            else:
                number_senones = 3100
                batch_size = 16

                self.gop_transitions(df_scores, output_gop)
                self.gop_robust_with_matrix(df_scores, df_phones_pure, number_senones, batch_size, output_gop_r, output_gop_r_max)
                self.gop2(df_scores, df_phones_pure,  output_gop2)            


        df_gop = []
        if align_transitions is None:
            return df_gop
        else:

            df_gop = pd.DataFrame(output_gop).set_index('logid')
            df_gop = df_gop[~df_gop.index.duplicated(keep='first')]
            
            df_gop_r = pd.DataFrame(output_gop_r).set_index('logid')
            df_gop_r = df_gop_r[~df_gop_r.index.duplicated(keep='first')]

            df_gop_r_max = pd.DataFrame(output_gop_r_max).set_index('logid')
            df_gop_r_max = df_gop_r_max[~df_gop_r_max.index.duplicated(keep='first')]

            df_gop2 = pd.DataFrame(output_gop2).set_index('logid')
            df_gop2 = df_gop2[~df_gop2.index.duplicated(keep='first')]
        
            return df_gop, df_gop_r_max, df_gop_r, df_gop2

    def cut_align_transitions(self, align_transitions, tam):
        
        i = 0
        tam_parcial = 0
        res = []
        while i < len(align_transitions):
           tam_actual = len(align_transitions[i])
           if tam_parcial + tam_actual <= tam:
                res.append(align_transitions[i])
                tam_parcial = tam_parcial + tam_actual
                i = i + 1 
           else:
                return res
        
        return res    


    def matrix_gop_robust(self, df_phones_pure, number_senones, batch_size):            
        pdfs_to_phone_pure_mask = []

        for phone_pure in range(0, len(list(df_phones_pure.phone_pure.unique()))):
            pdfs = df_phones_pure.loc[(df_phones_pure['phone_pure'] == str(phone_pure+1) )].pdf.unique()
            pdfs_to_phone_pure_file = np.zeros(number_senones)
                                
            for pdf in pdfs:
                pdfs_to_phone_pure_file[int(pdf)] = 1.0
            
            pdfs_to_phone_pure_mask.append(pdfs_to_phone_pure_file)
                                    
        pdfs_to_phone_pure_mask_3D = []

        for i in range(0, batch_size):                
            pdfs_to_phone_pure_mask_3D.append(pdfs_to_phone_pure_mask)
        
        return pdfs_to_phone_pure_mask_3D

    def gop_robust_with_matrix(self, df_scores, df_phones_pure, number_senones, batch_size, output_gop_r, output_gop_r_max):
        
        mask_score = self.matrix_gop_robust(df_phones_pure, number_senones, batch_size)

        mask_score = np.array(mask_score)
        mask_score = mask_score.transpose(0, 2, 1)

        scores = df_scores.p
        scores = np.array(list(scores))
        scores_phone_pure = np.matmul(scores,mask_score)
        
        logids = df_scores.index
        for j in range(0, len(df_scores)):
            phones = df_scores.phones[j]
            transitions = df_scores.transitions[j] 
            logid = logids[j]

            ti = 0
            gops_r_max = []
            gops_r = []
            phones_pure = []
            
            for i in range(0, len(transitions)):
                transitions_by_phone = transitions[i]
                tf = ti + len(transitions_by_phone) - 1

                lpp = (sum(np.log(scores_phone_pure[j][ti:tf+1])))/(tf-ti+1)
                
                phone_pure = df_phones_pure.loc[(df_phones_pure['phone'] == str(phones[ti]) )].phone_pure.unique()[0]
                gop_r_max = lpp[int(phone_pure)-1] - max(lpp)
                gop_r = lpp[int(phone_pure)-1]

                phones_pure.append(phone_pure)
                gops_r_max.append(gop_r_max)
                gops_r.append(gop_r)
                
                ti = tf + 1
            
            output_gop_r_max.append({'gop': gops_r_max,
                                    'phones_pure': phones_pure,
                                    'logid': logid})

            output_gop_r.append({'gop': gops_r,
                                'phones_pure': phones_pure,
                                'logid': logid})




    def calcular_gop_transitions(self, scores, ti, tf, y):
        scores_by_phone = []
        for k in range(ti, tf+1):
            target = y[k]
            scores_by_phone.append(scores[k][target])

        suma = sum(scores_by_phone)
        suma /= (tf+1)-ti 

        return suma

        
    def gop_transitions(self, df_scores, output_gop):
        gops = []

        #por cada waveforms
        for j in range(0, len(df_scores)):
            df_phones = df_scores.phones[j]
            align_transitions = df_scores.transitions[j]             
            logid = df_scores.iloc()[j].name
            scores = df_scores.iloc()[j].log_p
            y = df_scores.iloc()[j].y

            ti = 0
            gops = []
            phones = []

            for i in range(0, len(align_transitions)):
                transitions = align_transitions[i]
                tf = ti + len(transitions) - 1                
                
                gop = self.calcular_gop_transitions(scores, ti, tf, y)

                phone = df_phones[ti]
                phones.append(phone)
                gops.append(gop)
                
                ti = tf + 1
          
            
            output_gop.append({'gop': gops,
                            'phones' : phones,
                            'logid': logid})
            



    def get_senones1(self, align_transitions, df_transitions):
        senones_by_phone = []
        current_state = df_transitions.loc()[align_transitions[0]].hmm_state
        for i in range(1, len(align_transitions)):
            state = df_transitions.loc()[align_transitions[i]].hmm_state
            
            if current_state != state :
                current_state = state    
                senone = df_transitions.loc()[align_transitions[i-1]].pdf
                senones_by_phone.append(senone)
                
        senone = df_transitions.loc()[align_transitions[len(align_transitions)-1]].pdf
        
        senones_by_phone.append(senone)
        phone = df_transitions.loc[align_transitions[len(align_transitions)-1]].phone_name

        if len(senones_by_phone) != 3 and phone not in ('sil', 'sil_B', 'sil_E', 'sil_I', 'sil_S', 'sp', 'sp_B', 'sp_E', 'sp_I', 'sp_S', 'spn_B', 'spn_E', 'spn_I', 'spn_S'):
            print("cantidad de senones por fono incorrecto")
            embed()
        
        return senones_by_phone

    def calcular_gop_robusto(self, scores, ti, tf, senones_by_phone):
        gop = 0
        for k in range(ti, tf+1):
            ot = scores[k]
            ot_temp = [] 
            for i in range(0, len(senones_by_phone)):
                s = int(senones_by_phone[i])
                ot_temp.append(ot[s])
           
            gop = gop + np.log(sum(ot_temp))

            #gop = gop + logsumexp(ot_temp)
            #gop = gop + (logsumexp(ot_temp) - np.log(3))

        return gop/(tf-ti+1)

    def gop2(self, df_scores, df_phones_pure, output):
        
        gops = []
        output_ = []                               

        for j in range(0, len(df_scores)):
            logid = df_scores.iloc()[j].name
            scores = df_scores.iloc()[j].p
            df_phones = df_scores.phones[j]
            align_transitions = df_scores.transitions[j]             
            y = df_scores.iloc()[j].y

            ti = 0
            gops = []
            phones = []

            for i in range(0, len(align_transitions)):
                tf = ti + len(align_transitions[i]) - 1                

                if(len(align_transitions[i]) < 3 and not self.todos_unos(align_transitions[i])):
                    print("MENOR QUE 3: " + logid)
                    embed()
                
                senones_by_phone = self.get_senones1(align_transitions[i], df_phones_pure)
                gop = self.calcular_gop_robusto(scores, ti, tf, senones_by_phone)
                phone = df_phones[ti]
                phones.append(phone)
                gops.append(gop)

                ti = tf + 1

            output.append({'gop': gops,
                            'phones': phones,
                            'logid': logid})


    def todos_unos(self, align_transitions):
        for i in range(0, len(align_transitions)):
            if align_transitions[i] != '1':
                return False
        return True  

