import os
import sys
import glob
from IPython import embed
sys.path.append("../..")
from src.utils.finetuning_utils import *

#Returns the target phone and labels columns from a labels file
def get_target_and_labels_column(labels_file):
    labels_fh = open(labels_file)
    target_column = []
    labels = []
    for line in labels_fh.readlines():
        target_phone = line.split()[1]
        label = line.split()[3]
        target_column.append(target_phone)
        labels.append(label)
    return target_column, labels

#Updates the count in a phone count dictionary with the occurences in a given transcription
def add_phone_count(phone_count_dict, transcription, phone_sym2int, as_int=False, separate_pos_neg=False, labels=None):

    for i, phone in enumerate(transcription):
        if as_int:
            phone = phone_sym2int[phone]
        if separate_pos_neg:
            phone = phone + labels[i]
        
        if phone in phone_count_dict:
            phone_count_dict[phone] = phone_count_dict[phone] + 1
        else:
            phone_count_dict[phone] = 1
    return phone_count_dict

#Creates a file with the amount of occurences of each phone-class in the dataset
def create_phone_class_count_list(class_count_dict, phone_class_counts_path):
    class_counts_fh = open(phone_class_counts_path, "w+")
    class_counts_fh.write("---\n")
    for phone in sorted(class_count_dict.keys()):
        occurrences = class_count_dict[phone]
        class_counts_fh.write("  " + str(phone) + ":    " + str(occurrences) + "\n")

#Creates a file with the weight for each phone based on the amount of occurences for each phone-class in the dataset
def create_phone_weight_yaml(phone_weights_path, phone_count_dict, class_count_dict, phone_sym2int, phone_int2sym, phone_thereshold, class_thereshold):    

    phone_weights_fh = open(phone_weights_path, "w+")
    phone_weights_fh.write("---\n")

    for phone in sorted(phone_count_dict.keys()):
        phone_sym = phone_int2sym[phone]
        minority_occurences = min(class_count_dict[phone_sym + '-'], class_count_dict[phone_sym + '+'])
        occurrences = phone_count_dict[phone]

        if occurrences < phone_thereshold or minority_occurences < class_thereshold:
            phone_weights_fh.write("  " + str(phone_sym) + ":    " + str(0) + "\n")
        else:
            total_occurrences = sum(phone_count_dict.values())
            phone_count = len(phone_count_dict.keys())
            weight = occurrences / total_occurrences * phone_count * 2
            phone_weights_fh.write("  " + str(phone_sym) + ":    " + str(weight) + "\n")

if __name__ == "__main__":
    labels_dir = "../../data/kaldi_labels/"

    phone_list_path               = "../../phones/kaldi_phones_list.txt"
    phone_weights_path            = "phone_weights.yaml"
    phone_class_counts_path       = "phone_class_counts.yaml"
    phone_count                   = 39
    phone_thereshold              = 30
    class_thereshold              = 15

    phone_sym2int, phone_int2sym, _ = get_phone_dictionaries(phone_list_path)

    phone_int_count_dict      = {phone:0 for phone in range(phone_count)}
    phone_sym_count_dict      = {phone_int2sym[phone_int]+'+':0 for phone_int in range(phone_count)}
    phone_sym_count_dict.update({phone_int2sym[phone_int]+'-':0 for phone_int in range(phone_count)})


    phone_int_count_dict = {}
    for labels_file in glob.glob(labels_dir + "*/labels/*.txt"):
        target_column, labels = get_target_and_labels_column(labels_file)
        phone_int_count_dict = add_phone_count(phone_int_count_dict, target_column, phone_sym2int, as_int=True)
        phone_sym_count_dict = add_phone_count(phone_sym_count_dict, target_column, phone_sym2int, separate_pos_neg=True, labels=labels)


    create_phone_class_count_list(phone_sym_count_dict, phone_class_counts_path)
    create_phone_weight_yaml(phone_weights_path, phone_int_count_dict, phone_sym_count_dict, phone_sym2int, phone_int2sym, phone_thereshold, class_thereshold)    
