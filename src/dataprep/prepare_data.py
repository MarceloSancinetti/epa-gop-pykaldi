import glob
import os
import argparse
from src.utils.FeatureManager import FeatureManager
from src.utils.utils import makedirs_for_file
import src.dataprep.convert_chain_to_pytorch as convert_chain_to_pytorch
from IPython import embed

def generate_arguments(args_dict):
    res = ""
    for arg_name, value in args_dict.items():
        res = res + "--" + arg_name + " " + str(value) + " "
    return res

def download_librispeech_models(librispeech_models_path):
    if not os.path.exists(librispeech_models_path):
        os.makedirs("librispeech_models/")
        os.system("wget https://kaldi-asr.org/models/13/0013_librispeech_v1_chain.tar.gz")
        os.system("wget https://kaldi-asr.org/models/13/0013_librispeech_v1_lm.tar.gz")
        os.system("wget https://kaldi-asr.org/models/13/0013_librispeech_v1_extractor.tar.gz")
        os.system("tar -xf 0013_librispeech_v1_chain.tar.gz -C " + librispeech_models_path)
        os.system("tar -xf 0013_librispeech_v1_lm.tar.gz -C " + librispeech_models_path)
        os.system("tar -xf 0013_librispeech_v1_extractor.tar.gz -C " + librispeech_models_path)
        os.system("rm -f 0013_librispeech_v1_chain.tar.gz")
        os.system("rm -f 0013_librispeech_v1_lm.tar.gz")
        os.system("rm -f 0013_librispeech_v1_extractor.tar.gz")

def prepare_pytorch_models(pytorch_models_path, libri_chain_mdl_path, libri_chain_txt_path, acoustic_model_path):
    #Convert librispeech acoustic model .mdl to .txt
    if not os.path.exists(libri_chain_txt_path):
        os.system("nnet3-copy --binary=false " + libri_chain_mdl_path + " " + libri_chain_txt_path)

    #Create directory for pytorch models
    if not os.path.exists(pytorch_models_path):
        os.makedirs(pytorch_models_path)

    #Convert final.txt to pytorch acoustic model used in alginments stage
    if not os.path.exists(acoustic_model_path):
        makedirs_for_file(acoustic_model_path)
        config_dict = {"libri-chain-txt-path": libri_chain_txt_path,
                       "acoustic-model-path":  acoustic_model_path}
        convert_chain_to_pytorch.main(config_dict)

def create_epadb_full_sample_list(data_root_path, utterance_list_path):
    #Skip if utterance list already exists
    if os.path.exists(utterance_list_path):
        return
    
    #Create sample_lists directory if necessary
    utterance_list_dir_path = os.path.dirname(utterance_list_path) 
    if not os.path.exists(utterance_list_dir_path):
       os.makedirs(utterance_list_dir_path)

    utt_list_fh = open(utterance_list_path, 'w+')
    for file in sorted(glob.glob(data_root_path + '/*/waveforms/*.wav')):
        basename = os.path.basename(file)
        utt_list_fh.write(basename.split('.')[0] + ' ' + file + '\n')

def copy_sample_lists_and_reference(train_list_path, test_list_path, reference_path, dataprep_output_path):
    new_train_list_path = dataprep_output_path + "epadb_full_path_list.txt"
    new_test_list_path = dataprep_output_path  + "heldout_full_path_list.txt"
    new_reference_path = dataprep_output_path  + "reference_transcriptions.txt"

    if not os.path.exists(new_train_list_path):
        os.system('cp ' + train_list_path + ' ' + new_train_list_path)

    if not os.path.exists(new_test_list_path):
        os.system('cp ' + test_list_path + ' ' + new_test_list_path)

    if not os.path.exists(new_reference_path):
        os.system('cp ' + reference_path + ' ' + new_reference_path)

def create_ref_labels_symlinks(data_root_path, labels_path):
    #Create symbolic links to epa reference labels
    for file in sorted(glob.glob(data_root_path + '*/labels/*')):
        fullpath = os.path.abspath(file)
        basename = os.path.basename(file)
        #Get spkr id
        spkr = fullpath.split('/')[-3]
        labels_dir_for_spkr = labels_path + spkr + '/labels/' 
        #Create directory for speaker's labels
        if not os.path.exists(labels_dir_for_spkr):
            os.system('mkdir -p ' + labels_dir_for_spkr)
        #Make symbolic link to speaker labels from EpaDB directory
        if not os.path.exists(labels_dir_for_spkr + '/' + basename):
            os.system('ln -s ' + fullpath + ' ' + labels_dir_for_spkr + '/')

    #Handle symbolic links for EpaDB reference transcriptions
    if not os.path.exists(labels_path + 'reference_transcriptions.txt'):
        current_path = os.getcwd()
        cmd = 'ln -s ' + current_path + "/" + data_root_path + '/reference_transcriptions.txt ' + current_path + "/" + labels_path + '/reference_transcriptions.txt'
        os.system(cmd)

def main(config_dict):
    data_root_path          = config_dict['data-root-path']
    features_path           = config_dict['features-path']
    conf_path               = config_dict['features-conf-path']
    librispeech_models_path = config_dict['librispeech-models-path']
    pytorch_models_path     = config_dict['pytorch-models-path']
    libri_chain_mdl_path    = config_dict['libri-chain-mdl-path']
    libri_chain_txt_path    = config_dict['libri-chain-txt-path']
    acoustic_model_path     = config_dict['acoustic-model-path']
    train_list_path         = config_dict['train-list-path']
    test_list_path          = config_dict['test-list-path']
    reference_path          = config_dict['reference-trans-path']
    dataprep_output_path    = config_dict['output-dir']

    #Download librispeech models and extract them into librispeech-models-path
    download_librispeech_models(librispeech_models_path)

    #Prepare pytorch models
    prepare_pytorch_models(pytorch_models_path, libri_chain_mdl_path, libri_chain_txt_path, acoustic_model_path)

    #Extract features
    feature_manager = FeatureManager(data_root_path, features_path, conf_path)
    feature_manager.extract_features_using_kaldi()

    #Create links to utterance lists in data folder
    copy_sample_lists_and_reference(train_list_path, test_list_path, reference_path, dataprep_output_path)



