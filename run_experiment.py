import sys
sys.path.append("src")
sys.path.append("src/gop")
sys.path.append("src/evaluate")
import yaml
import argparse
import os
from run_utils import *
from IPython import embed

import train
import prepare_data
import generate_kfold_utt_lists

def generate_scores_and_evaluate_epochs(config_dict, step):
	epochs = config_dict['epochs']

	swa_epochs = config_dict.get('swa-epochs', 0)
	swa_start  = epochs - swa_epochs
	
	for epoch in range(0, epochs+1, step):
		print("Evaluating epoch %d/%d" % (int(epoch/step), int(epochs/step)))
		run_generate_scores(config_dict, epoch=epoch)
		run_evaluate(config_dict, epoch=epoch)

		if epoch >= swa_start and swa_epochs != 0:
			run_generate_scores(config_dict, epoch=epoch, swa=True)
			run_evaluate(config_dict, epoch=epoch, swa=True)

def run_train(config_dict, device_name):
	if "held-out" in config_dict and config_dict["held-out"]:
		run_train_heldout(config_dict, device_name)
	else:
		run_train_kfold(config_dict, device_name)

def run_train_kfold(config_dict, device_name):
	generate_kfold_utt_list.main()

	for fold in range(fold_amount):
		config_dict["fold"]            = fold
		config_dict["train-list-path"] = config_dict["train-sample-list-dir"] + 'train_sample_list_fold_' + str(fold)
		config_dict["test-list-path"]  = config_dict["test-sample-list-dir"]  + 'test_sample_list_fold_'  + str(fold)
		config_dict["train-root-path"] = config_dict["epadb-root-path"]
		config_dict["test-root-path"]  = config_dict["epadb-root-path"]
		config_dict["device"]          = device_name				 
		train.main(config_dict)

def run_train_heldout(config_dict, device_name):

	config_dict["fold"]            = 0
	config_dict["train-root-path"] = config_dict["epadb-root-path"]
	config_dict["test-root-path"]  = config_dict["heldout-root-path"]
	config_dict["device"]          = device_name
	train.main(config_dict)

def run_all(config_yaml, stage, device_name, use_heldout):
	config_fh = open(config_yaml, "r")
	config_dict = yaml.safe_load(config_fh)	

	config_dict = extend_config_dict(config_yaml, config_dict, "exp", use_heldout)

	if stage in ["dataprep", "all"]:
		print("Running data preparation")
		run_data_prep(config_dict)

	if stage in ["align", "all"]:
		print("Running aligner")
		run_align(config_dict)
	
	if stage in ["labels", "all"] and config_dict['use-kaldi-labels']:
		print("Creating Kaldi labels")
		run_create_kaldi_labels(config_dict, 'exp')

	if stage in ["train+", "train", "all"]:
		print("Running training")
		run_train(config_dict, device_name)

	if stage in ["train+","evaluate", "all"]:
		print("Evaluating results")
		generate_scores_and_evaluate_epochs(config_dict, 25)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--config', dest='config_yaml',  help='Path .yaml config file for experiment', default=None)
	parser.add_argument('--stage', dest='stage',  help='Stage to run (dataprep, align, train, scores, evaluate), or \'all\' to run all stages', default=None)
	parser.add_argument('--device', dest='device_name', help='Device name to use, such as cpu or cuda', default=None)
	parser.add_argument('--heldout', action='store_true', help='Use this option to test on heldout set', default=False)

	args = parser.parse_args()
	use_heldout = args.heldout

	run_all(args.config_yaml, args.stage, args.device_name, use_heldout)
