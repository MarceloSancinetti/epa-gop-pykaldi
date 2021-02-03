from pathlib import Path 
import numpy as np 
import pandas as pd 
import tqdm
import pickle
import os



#Generates transitions file
def show_transitions(dir_show_transitions, show_transitions_output_filename):
    
    path_show_transitions = dir_show_transitions + "/" + show_transitions_output_filename

    if not os.path.isdir(dir_show_transitions):
        os.mkdir(dir_show_transitions)

    os.system("show-transitions " + "phones/phones.txt " + "../exp/chain_cleaned/tdnn_1d_sp/final.mdl > " + path_show_transitions)  





#Builds dictionary using transitions file
def get_transitions(path_show_transitions, path_output_transitions):

    transitions_dict = {}

    f = open(path_show_transitions, "r")
    phone = -1
    data = []
    for line in f:
            line_array = line.split(' ')
            if line_array[0] == 'Transition-state':
                    data = []
                    transition_state = line_array[1].split(":")[0]
                    phone = line_array[4]
                    hmm_state = line_array[7]
                    pdf = line_array[10]
                    pdf = pdf.split('\n')
                    pdf = pdf[0]

                    data.append(transition_state)
                    data.append(phone)
                    data.append(hmm_state)
                    data.append(pdf)
            if line_array[1] == 'Transition-id':
                    transition_id = line_array[3]
                    transitions_dict[transition_id] = data + [transition_id]


    df_transitions = pd.DataFrame.from_dict(transitions_dict, orient='index', columns=['transition_state', 'phone_name', 'hmm_state', 'pdf', 'transition_id'])

    return df_transitions


#Generates phones_pure_epa.pickle
def generate_df_phones_pure():

    phones = open("phones/phones.txt", "r")
    phones_dict = {}
    for line in tqdm.tqdm(phones):
            l = line.split()
            phones_dict[l[1]] = l[0]

    f = open("phones/phones-pure.txt", "r")
    phones_pure_dict = {}
    for line in tqdm.tqdm(f):
            l = line.split()
            phones_pure_dict[l[1]] = l[0]

    output = []
    phone_to_pure_file = open("phones/phone-to-pure-phone.int", "r")
    for line in tqdm.tqdm(phone_to_pure_file):
            l = line.split()
            output.append({'phone':l[0],
                            'phone_name':phones_dict[l[0]],
                            'phone_pure': l[1],
                            'phone_pure_name': phones_pure_dict[l[1]]})


    df_phone_pure = pd.DataFrame(output)

    dir_show_transitions = "transitions/"
    show_transitions_output_filename = "show-transitions.txt"
    
    path_show_transitions = dir_show_transitions + show_transitions_output_filename

    show_transitions(dir_show_transitions, show_transitions_output_filename)

    path_output_transitions = "transitions.pickle"


    df_transitions = get_transitions(path_show_transitions, path_output_transitions)

    df = df_transitions.set_index('phone_name').join(df_phone_pure.set_index('phone_name'))
    df = df.reset_index().set_index("phone")

    df.to_pickle('phones_pure_epa.pickle')




def get_phone_from_transition_id(id, df_phones_pure):
    return df_phones_pure.loc[df_phones_pure['transition_id'] == str(id),'phone_pure'].values[0]

def removeSymbols(str, symbols):
    for symbol in symbols:
        str = str.replace(symbol,'')
    return str

def get_alignments(path_alignements, df_phones_pure):
    
    alignments_dict = {}

    for l in open(path_alignements+"align_output", encoding="utf8", errors='ignore').readlines():
        l=l.split()
        #Get transitions alignments
        if len(l) > 3 and l[1] == 'transitions':
            waveform_name = l[0]
            alignment_array = []
            current_phone_transition = int(removeSymbols(l[2],['[',']',',']))
            current_phone = get_phone_from_transition_id(current_phone_transition, df_phones_pure)
            transitions = []
            #alignments_dict[waveform_name] = {}
            for i in range(2, len(l)):
                transition_id = int(removeSymbols(l[i],['[',']',',']))
                phone = get_phone_from_transition_id(transition_id, df_phones_pure)
                if phone != current_phone:
                    alignment_array.append(transitions)
                    transitions = []
                    current_phone_transition = transition_id
                    current_phone = get_phone_from_transition_id(current_phone_transition, df_phones_pure)
                transitions.append(transition_id)
            alignments_dict[waveform_name]['transitions'] = alignment_array

        #Get phones alignments
        if len(l) > 3 and l[1] == 'phones':
            waveform_name = l[0]
            phones = []
            alignments_dict[waveform_name] = {}
            for i in range(2, len(l),3):
                current_phone = removeSymbols(l[i],['[',']',',',')','(','\''])
                phones.append(current_phone)
            alignments_dict[waveform_name]['phones'] = phones
    return alignments_dict





def generate_alignments_df(data_path="."):
    df_phones_pure = pd.read_pickle('phones_pure_epa.pickle')

    #path_show_alignments = "alignments/"
    path_show_alignments = ""

    alignments_dict = get_alignments(path_show_alignments, df_phones_pure)

    with open('alignments.pickle', 'wb') as handle:
        pickle.dump(alignments_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

