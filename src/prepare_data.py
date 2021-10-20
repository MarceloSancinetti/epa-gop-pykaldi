import glob
import os
import argparse
from FeatureManager import FeatureManager
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

def prepare_pytorch_models(pytorch_models_path, libri_chain_mdl_path, libri_chain_txt_path, acoustic_model_path,  setup, batchnorm, seed,finetune_model_path = None, phone_count = None):
    #Convert librispeech acoustic model .mdl to .txt
    if not os.path.exists(libri_chain_txt_path):
        os.system("nnet3-copy --binary=false " + libri_chain_mdl_path + " " + libri_chain_txt_path)

    #Create directory for pytorch models
    if not os.path.exists(pytorch_models_path):
        os.makedirs(pytorch_models_path)

    #Convert final.txt to pytorch acoustic model used in alginments stage
    if not os.path.exists(acoustic_model_path):
        args_dict = {"chain-model-path": libri_chain_txt_path,
                     "output-path":      acoustic_model_path}
        arguments = generate_arguments(args_dict)
        os.system("python src/convert_chain_to_pytorch.py " + arguments)

    #Generate model to finetune in training stage
    if  setup == "exp" and not os.path.exists(finetune_model_path):
        args_dict = {"chain-model-path": libri_chain_txt_path,
                     "output-path":      finetune_model_path,
                     "phone-count":      phone_count,
                     "batchnorm":        batchnorm,
                     "seed":             seed}
        arguments = generate_arguments(args_dict)
        os.system("python src/convert_chain_to_pytorch_for_finetuning.py " + arguments)

def create_epadb_full_sample_list(epadb_root_path, utterance_list_path):
    #Skip if utterance list already exists
    if os.path.exists(utterance_list_path):
        return
    
    #Create sample_lists directory if necessary
    utterance_list_dir_path = os.path.dirname(utterance_list_path) 
    if not os.path.exists(utterance_list_dir_path):
       os.makedirs(utterance_list_dir_path)

    utt_list_fh = open(utterance_list_path, 'w+')
    for file in sorted(glob.glob(epadb_root_path + '/*/waveforms/*.wav')):
        basename = os.path.basename(file)
        utt_list_fh.write(basename.split('.')[0] + ' ' + file + '\n')

def create_ref_labels_symlinks(epadb_root_path, labels_path):
    #Create symbolic links to epa reference labels
    for file in sorted(glob.glob(epadb_root_path + '*/labels/*')):
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
        cmd = 'ln -s ' + current_path + "/" + epadb_root_path + '/reference_transcriptions.txt ' + current_path + "/" + labels_path + '/reference_transcriptions.txt'
        os.system(cmd)

def make_experiment_directory(experiment_dir_path, setup):
    #This will create the experiment directory and the test sample list, 
    #state dict, and gop scores directories inside of it
    test_sample_lists_dir  = experiment_dir_path + "/test_sample_lists/"
    train_sample_lists_dir = experiment_dir_path + "/train_sample_lists/"
    state_dicts_dir        = experiment_dir_path + "/state_dicts/"
    gop_scores_dir         = experiment_dir_path + "/gop_scores/"
    eval_dir               = experiment_dir_path + "/eval/"
    
    if not os.path.exists(test_sample_lists_dir) and setup == "exp":
        os.makedirs(test_sample_lists_dir)
    if not os.path.exists(train_sample_lists_dir) and setup == "exp":
        os.makedirs(train_sample_lists_dir)
    if not os.path.exists(state_dicts_dir) and setup == "exp":
        os.makedirs(state_dicts_dir)
    if not os.path.exists(gop_scores_dir):
        os.makedirs(gop_scores_dir)
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)


def main(config_dict):
    epadb_root_path         = config_dict['epadb-root-path']
    features_path           = config_dict['features-path']
    conf_path               = config_dict['features-conf-path']
    labels_path             = config_dict['epa-ref-labels-dir-path']
    librispeech_models_path = config_dict['librispeech-models-path']
    pytorch_models_path     = config_dict['pytorch-models-path']
    libri_chain_mdl_path    = config_dict['libri-chain-mdl-path']
    libri_chain_txt_path    = config_dict['libri-chain-txt-path']
    acoustic_model_path     = config_dict['acoustic-model-path']
    utterance_list_path     = config_dict['utterance-list-path']
    phone_count             = config_dict.get('phone-count', None)
    experiment_dir_path     = config_dict['experiment-dir-path']
    setup                   = config_dict['setup']
    use_heldout             = config_dict['held-out']
    heldout_root_path       = config_dict['heldout-root-path']
    heldout_list_path       = config_dict['test-list-path']
    finetune_model_path     = config_dict['finetune-model-path']
    batchnorm               = config_dict.get('batchnorm', None)
    seed                    = config_dict['seed']

    if setup != "exp" and setup != "gop":
        raise Exception("Error: setup argument must be either gop or exp")
        exit()

    make_experiment_directory(experiment_dir_path, setup)

    #Download librispeech models and extract them into librispeech-models-path
    download_librispeech_models(librispeech_models_path)

    #Prepare pytorch models
    prepare_pytorch_models(pytorch_models_path, libri_chain_mdl_path, libri_chain_txt_path, 
                           acoustic_model_path, setup, batchnorm, seed, finetune_model_path, phone_count=phone_count)

    #Extract features
    feature_manager = FeatureManager(epadb_root_path, features_path, conf_path, heldout_root_path=heldout_root_path)
    feature_manager.extract_features_using_kaldi()

    #Create symlinks
    create_ref_labels_symlinks(epadb_root_path, labels_path)

    
    #Create full EpaDB sample list
    create_epadb_full_sample_list(epadb_root_path, utterance_list_path)
    
    if use_heldout:
        create_epadb_full_sample_list(heldout_root_path, heldout_list_path)




