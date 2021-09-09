import yaml
import argparse
import os
from utils import *

def extend_config_dict(config_yaml, config_dict):
	config_dict["experiment-dir-path"] 	= get_experiment_directory(config_yaml)
	config_dict["run-name"] 			= get_run_name(config_yaml)
	config_dict["test-sample-list-dir"] = config_dict["experiment-dir-path"] 	 + "test_sample_lists/"
	config_dict["state-dict-dir"] 		= config_dict["experiment-dir-path"] 	 + "state_dicts/"
	config_dict["gop-scores-dir"] 		= config_dict["experiment-dir-path"] 	 + "gop_scores/"
	config_dict["full-gop-score-path"] 	= config_dict["gop-scores-dir"] 	 	 + "gop-all-folds.txt"
	config_dict["eval-dir"] 			= config_dict["experiment-dir-path"] 	 + "eval/"
	config_dict["alignments-path"]      = config_dict["experiment-dir-path"] 	 + "align_output"
	config_dict["loglikes-path"]        = config_dict["experiment-dir-path"] 	 + "loglikes.ark"
	config_dict["transcription-file"]   = config_dict["epa-ref-labels-dir-path"] + "reference_transcriptions.txt"
	config_dict["finetune-model-path"]  = config_dict["experiment-dir-path"]     + "/model_finetuning_kaldi.pt"

	#Choose labels dir
	if config_dict["use-kaldi-labels"]:
		config_dict["labels-dir"] = config_dict["kaldi-labels-path"]
	else:
		config_dict["labels-dir"] = config_dict["epa-ref-labels-dir-path"]

	return config_dict

def run_all(config_yaml, step):
	config_fh = open(config_yaml, "r")
	config_dict = yaml.safe_load(config_fh)	

	config_dict = extend_config_dict(config_yaml, config_dict)

	epochs = config_dict['epochs']
	
	for epoch in range(0, epochs, step):
		print("Evaluating epoch %d/%d" % (int(epoch/step), int(epochs/step)))
		run_generate_scores(config_dict, epoch=epoch)
		run_evaluate(config_dict, epoch=epoch)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--config', dest='config_yaml',  help='Path .yaml config file for experiment', default=None)
	parser.add_argument('--step', dest='step',  help='How many epochs between each comparison', type=int, default=None)

	args = parser.parse_args()

	run_all(args.config_yaml, args.step)
