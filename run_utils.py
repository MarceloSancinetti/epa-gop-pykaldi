import sys
sys.path.append("src")
sys.path.append("src/gop")
sys.path.append("src/evaluate")
import yaml
import argparse
import os
from IPython import embed

import prepare_data
import align
import generate_data_for_eval
import create_kaldi_labels
import generate_score_txt

def generate_arguments(args_dict):
	res = ""
	for arg_name, value in args_dict.items():
		res = res + "--" + arg_name + " " + str(value) + " "
	return res

def run_script(script, args_dict):
	arguments = generate_arguments(args_dict)
	return os.system("python " + script + " " + arguments)

def get_phone_count(phone_list_path):
	return len(open(phone_list_path).readlines())

def get_run_name(config_yaml, use_heldout=False):
	heldout_suffix = ''
	if use_heldout:
		heldout_suffix = '_heldout'
	return os.path.basename(config_yaml).split('.')[0] + heldout_suffix

def get_experiment_directory(config_yaml, use_heldout=False):
	return "experiments/" + get_run_name(config_yaml, use_heldout) + '/'

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


def get_model_name(config_dict, fold, epoch=None, use_heldout=False, swa=False):
	if epoch == None:
		epoch  = config_dict["epochs"]
	run_name   = config_dict["run-name"]
	
	swa_id  = swa_identifier(swa)
	fold_id = fold_identifier(use_heldout, fold)

	return run_name +  fold_id + '-epoch-' + str(epoch) + swa_id #Aca hay codigo repetido entre el PATH de train y esto

def get_test_sample_list_path_for_fold(test_sample_list_dir, fold):
	return test_sample_list_dir + "/test_sample_list_fold_" + str(fold) #Aca tmb codigo repetido

def run_data_prep(config_dict):
	prepare_data.main(config_dict)

def run_align(config_dict):
	align.main(config_dict)

	if config_dict.get("held-out"):
		config_dict['epadb-root-path']     = config_dict['heldout-root-path']
		config_dict['utterance-list-path'] = config_dict['test-list-path']
		config_dict['alignments-path']     = config_dict['heldout-align-path']
		align.main(config_dict)
	
def run_evaluate(config_dict, epoch='', swa=False):
	if config_dict.get("held-out"):
		run_evaluate_heldout(config_dict, epoch=epoch, swa=swa)
	else:
		run_evaluate_kfold(config_dict, epoch=epoch, swa=swa)

def run_evaluate_kfold(config_dict, epoch='', swa=False):
	config_dict["eval-filename"] = get_eval_filename(epoch, swa)
	
	generate_data_for_eval.main(config_dict)

def run_evaluate_heldout(config_dict, epoch='', swa=False):
	model_name = get_model_name(config_dict, 0, epoch=epoch, use_heldout=True, swa=swa)
	
	config_dict["eval-filename"]        = get_eval_filename(epoch, swa)
	config_dict["full-gop-score-path"] 	= config_dict["gop-scores-dir"] + 'gop-'+model_name+'.txt'
				
	generate_data_for_eval.main(config_dict)

def run_create_kaldi_labels(config_dict, setup):
	
	config_dict['labels-dir-path']               = config_dict['kaldi-labels-path']
	create_kaldi_labels.main(config_dict)
	
	if config_dict.get("held-out"):
		config_dict['reference-trans-path']           = config_dict["heldout-root-path"] + "/reference_transcriptions.txt"
		config_dict['utterance-list-path']           = config_dict['test-list-path']
		config_dict['labels-dir-path']               = config_dict['heldout-root-path']
		config_dict['alignments-path']               = config_dict['heldout-align-path']
		create_kaldi_labels.main(config_dict)

def run_generate_scores(config_dict, epoch=None, swa=False):
	if config_dict.get("held-out"):
		run_generate_scores_heldout(config_dict, epoch=epoch, swa=swa)
	else:
		run_generate_scores_kfold(config_dict, epoch=epoch, swa=swa)

def run_generate_scores_kfold(config_dict, epoch=None, swa=False, device='cpu'):
	cat_file_names = ""
	for fold in range(config_dict["folds"]):
		model_name = get_model_name(config_dict, fold, epoch=epoch, use_heldout=False, swa=swa)
		utterance_list_path = get_test_sample_list_path_for_fold(config_dict["test-sample-list-dir"], fold)

		config_dict["model-name"]           = model_name
		config_dict["utterance-list-path"]  = utterance_list_path
		config_dict["gop-txt-name"]         = 'gop-'+model_name+'.txt'
		config_dict["device-name"]          = device
		generate_score_txt.main(config_dict)
		cat_file_names += config_dict['gop-scores-dir'] + '/' +'gop-'+config_dict['model-name']+'.txt ' #Codigo repetido con generate_score_txt
	#Concatenate gop scores for all folds
	os.system("cat " + cat_file_names + " > " + config_dict["full-gop-score-path"])

def run_generate_scores_heldout(config_dict, epoch=None, swa=False, device='cpu'):
	model_name = get_model_name(config_dict, 0, epoch=epoch, use_heldout=True, swa=swa)
	config_dict["model-name"]           = model_name
	config_dict["utterance-list-path"]  = config_dict["test-list-path"]
	config_dict["gop-txt-name"]         = 'gop-'+model_name+'.txt'
	config_dict["device-name"]          = device
	config_dict["epadb-root-path"]      = config_dict["heldout-root-path"]

	generate_score_txt.main(config_dict)

def extend_config_dict(config_yaml, config_dict, setup, use_heldout):
	config_dict["experiment-dir-path"] 	 = get_experiment_directory(config_yaml, use_heldout=use_heldout)
	config_dict["run-name"] 			 = get_run_name(config_yaml, use_heldout=use_heldout)
	config_dict["test-sample-list-dir"]  = config_dict["experiment-dir-path"] 	  + "test_sample_lists/"
	config_dict["train-sample-list-dir"] = config_dict["experiment-dir-path"] 	  + "train_sample_lists/"
	config_dict["state-dict-dir"] 		 = config_dict["experiment-dir-path"] 	  + "state_dicts/"
	config_dict["gop-scores-dir"] 		 = config_dict["experiment-dir-path"] 	  + "gop_scores/"
	config_dict["eval-dir"] 			 = config_dict["experiment-dir-path"] 	  + "eval/"
	config_dict["alignments-path"]       = config_dict["experiment-dir-path"] 	  + "align_output"
	config_dict["heldout-align-path"]    = config_dict["experiment-dir-path"]     + "align_output_heldout"	
	config_dict["loglikes-path"]         = config_dict["experiment-dir-path"] 	  + "loglikes.ark"
	config_dict["reference-trans-path"]  = config_dict["epa-ref-labels-dir-path"] + "reference_transcriptions.txt"
	config_dict["held-out"]              = use_heldout
	config_dict["libri-chain-mdl-path"]  = config_dict["libri-chain-mdl-path"]
	config_dict["libri-chain-txt-path"]  = config_dict["libri-chain-txt-path"]
	config_dict["setup"]                 = setup
	config_dict["phone-count"]           = get_phone_count(config_dict["phones-list-path"])
	config_dict["seed"]                  = 42

	#Choose labels dir
	if config_dict["use-kaldi-labels"]:
		config_dict["labels-dir-path"] = config_dict["kaldi-labels-path"]
	else:
		config_dict["labels-dir-path"] = config_dict["epa-ref-labels-dir-path"]

	if "utterance-list-path" not in config_dict:
		config_dict["utterance-list-path"] = config_dict["train-list-path"]
	if "train-list-path" not in config_dict:
		config_dict["train-list-path"] = config_dict["utterance-list-path"]

	if  use_heldout:
		config_dict["eval-utt-list-path"]  = config_dict["test-list-path"]
	else:
		config_dict["eval-utt-list-path"]  = config_dict["utterance-list-path"]  
		config_dict["full-gop-score-path"] = config_dict["gop-scores-dir"] + "gop-all-folds.txt"


	#If only one layer will be trained or finetune model path is not defined, make finetune model path relative to experiment dir
	if config_dict["layers"] == 1 or "finetune-model-path" not in config_dict:
		config_dict["finetune-model-path"]   = config_dict["experiment-dir-path"] + "/model_finetuning_kaldi.pt"


	return config_dict
