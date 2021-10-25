import os

import generate_kfold_utt_lists
import train
import generate_score_txt
import generate_data_for_eval
import convert_chain_to_pytorch_for_finetuning
from Stages import *
from run_utils import *

def check_if_config_exists(config_path):
    if not os.path.exists(config_path):
        raise Exception('Config with path %s not found.' %(config_path) )


def get_model_name(config_dict, fold, epoch=None, use_heldout=False, swa=False):
    if epoch == None:
        epoch  = config_dict["epochs"]
    run_name   = config_dict["run-name"]

    swa_id  = swa_identifier(swa)
    fold_id = fold_identifier(use_heldout, fold)

    return run_name +  fold_id + '-epoch-' + str(epoch) + swa_id #Aca hay codigo repetido entre el PATH de train y esto


def get_test_sample_list_path_for_fold(test_sample_list_dir, fold):
    return test_sample_list_dir + "/test_sample_list_fold_" + str(fold) #Aca tmb codigo repetido


def swa_identifier(is_swa):
	if is_swa:
		swa_identifier = '_swa'
	else:
		swa_identifier = ''

	return swa_identifier


def get_eval_filename(epoch, is_swa):
	swa_id = swa_identifier(is_swa)
	return "data_for_eval_epoch" + str(epoch) + swa_id + ".pickle"

def fold_identifier(use_heldout, fold_number):
	if use_heldout:
		fold_identifier = ''
	else:
		fold_identifier = '-fold-' + str(fold_number)
		
	return fold_identifier

class CreateExperimentDirectoryStage(AtomicStage):
    _name = "prepdir"
    def run(self):
        make_experiment_directory(self._config_dict["experiment-dir-path"], "exp")

class CreateFinetuneModelStage(AtomicStage):
    _name = "prepmodel"
    def run(self):
        #Generate model to finetune in training stage
        convert_chain_to_pytorch_for_finetuning.main(self._config_dict)

class TrainCrossValStage(AtomicStage):
    _name = "train"
    def run(self):
        config_dict = self._config_dict
        generate_kfold_utt_lists.main(config_dict)

        fold_amount = config_dict["folds"]

        for fold in range(fold_amount):
            config_dict["fold"]            = fold
            config_dict["train-list-path"] = config_dict["train-sample-list-dir"] + 'train_sample_list_fold_' + str(fold)
            config_dict["test-list-path"]  = config_dict["test-sample-list-dir"]  + 'test_sample_list_fold_'  + str(fold)
            config_dict["train-root-path"] = config_dict["epadb-root-path"]
            config_dict["test-root-path"]  = config_dict["epadb-root-path"]				 
            train.main(config_dict)

class TrainHeldoutStage(AtomicStage):
    _name = "train"
    def run(self):
        config_dict = self._config_dict

        config_dict["fold"]            = 0
        config_dict["train-root-path"] = config_dict["epadb-root-path"]
        config_dict["test-root-path"]  = config_dict["heldout-root-path"]
        train.main(config_dict)

class GenerateScoresCrossValStage(AtomicStage):
    _name = "scores"
    def __init__(self, config_dict, epoch=None, is_swa=False):

        #self._name        = name
        self._config_dict = config_dict.copy()
        self._epoch       = epoch
        self._is_swa      = is_swa

    def run(self):
        config_dict = self._config_dict
        cat_file_names = ""
        for fold in range(config_dict["folds"]):
            model_name = get_model_name(config_dict, fold, epoch=self._epoch, use_heldout=False, swa=self._is_swa)
            utterance_list_path = get_test_sample_list_path_for_fold(config_dict["test-sample-list-dir"], fold)

            config_dict["model-name"]           = model_name
            config_dict["utterance-list-path"]  = utterance_list_path
            config_dict["gop-txt-name"]         = 'gop-'+model_name+'.txt'
            generate_score_txt.main(config_dict)
            cat_file_names += config_dict['gop-scores-dir'] + '/' +'gop-'+config_dict['model-name']+'.txt ' #Codigo repetido con generate_score_txt
        #Concatenate gop scores for all folds
        os.system("cat " + cat_file_names + " > " + config_dict["full-gop-score-path"])

class GenerateScoresHeldoutStage(AtomicStage):
    _name = "scores"
    def __init__(self, config_dict, epoch=None, is_swa=False):

        #self._name        = name
        self._config_dict = config_dict.copy()
        self._epoch       = epoch
        self._is_swa      = is_swa

    def run(self):
        config_dict = self._config_dict
        model_name = get_model_name(config_dict, 0, epoch=self._epoch, use_heldout=True, swa=self._is_swa)
        config_dict["model-name"]           = model_name
        config_dict["utterance-list-path"]  = config_dict["test-list-path"]
        config_dict["gop-txt-name"]         = 'gop-'+model_name+'.txt'
        config_dict["epadb-root-path"]      = config_dict["heldout-root-path"]

        generate_score_txt.main(config_dict)

class EvaluateScoresCrossValStage(AtomicStage):
    _name = "evaluate"
    def __init__(self, config_dict, epoch=None, is_swa=False):

        #self._name        = name
        self._config_dict = config_dict.copy()
        self._epoch       = epoch
        self._is_swa      = is_swa

    def run(self):
        config_dict = self._config_dict
        config_dict["eval-filename"] = get_eval_filename(self._epoch, self._is_swa)
        generate_data_for_eval.main(config_dict)

class EvaluateScoresHeldoutStage(AtomicStage):
    _name = "evalaute"
    def __init__(self, config_dict, epoch=None, is_swa=False):

        #self._name        = name
        self._config_dict = config_dict.copy()
        self._epoch       = epoch
        self._is_swa      = is_swa
    
    def run(self):
        config_dict = self._config_dict

        model_name = get_model_name(config_dict, 0, epoch=self._epoch, use_heldout=True, swa=self._is_swa)
        
        config_dict["eval-filename"]        = get_eval_filename(self._epoch, self._is_swa)
        config_dict["utterance-list-path"]  = config_dict["test-list-path"]
        config_dict["full-gop-score-path"] 	= config_dict["gop-scores-dir"] + 'gop-'+model_name+'.txt'
        
        generate_data_for_eval.main(config_dict)