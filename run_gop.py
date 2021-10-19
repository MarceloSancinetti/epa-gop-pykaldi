import yaml
import argparse
import os
from run_utils import *

def extend_config_dict(config_yaml, config_dict):
	config_dict["experiment-dir-path"] 	= get_experiment_directory(config_yaml)
	#config_dict["gop-dir"] 			= config_dict["experiment-dir-path"] 	 	 + "gop_scores/"
	config_dict["eval-dir"] 			= config_dict["experiment-dir-path"] 	 + "eval/"
	config_dict["alignments-path"]      = config_dict["experiment-dir-path"] 	 + "align_output"
	config_dict["loglikes-path"]        = config_dict["experiment-dir-path"] 	 + "loglikes.ark"
	config_dict["transcription-file"]   = config_dict["epa-ref-labels-dir-path"] + "reference_transcriptions.txt"
	config_dict["full-gop-score-path"]  = config_dict["experiment-dir-path"] 	 + "gop.txt"

	#Choose labels dir
	if config_dict["use-kaldi-labels"]:
		config_dict["labels-dir"] = config_dict["kaldi-labels-path"]
	else:
		config_dict["labels-dir"] = config_dict["epa-ref-labels-dir-path"]

	return config_dict

def run_gop(config_dict):
	args_dict = {'libri-phones-path': 			  config_dict["librispeech-models-path"] 
												  + config_dict['libri-phones-path'],
												  
				 'utterance-list-path': 		  config_dict["utterance-list-path"],

				 'libri-phones-to-pure-int-path': config_dict['libri-phones-to-pure-int-path'],

				 'libri-phones-pure-path': 		  config_dict['libri-phones-pure-path'],

				 'transition-model-path': 		  config_dict["librispeech-models-path"] 
												  + config_dict['transition-model-path'],

				 'gop-dir': 					  config_dict['experiment-dir-path'],
				 'loglikes-path': 				  config_dict['loglikes-path'],
				 'alignments-dir-path':			  config_dict['experiment-dir-path']
				}
	run_script("src/gop/calculate_gop.py", args_dict)


def run_all(config_yaml, stage, use_heldout):
	config_fh = open(config_yaml, "r")
	config_dict = yaml.safe_load(config_fh)	

	config_dict = extend_config_dict(config_yaml, config_dict, "gop", use_heldout)

	if stage in ["dataprep", "all"]:
		print("Running data preparation")
		run_data_prep(config_dict)

	if stage in ["align", "all"]:
		print("Running aligner")
		run_align(config_dict)
	
	if stage in ["gop", "all"]:
		if config_dict['use-kaldi-labels']:
			print("Creating Kaldi labels")
			run_create_kaldi_labels(config_dict, 'gop')
		print("Running GOP")
		run_gop(config_dict)
			
	if stage in ["evaluate", "all"]:
		print("Evaluating results")
		run_evaluate(config_dict)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--config', dest='config_yaml',  help='Path .yaml config file for experiment', default=None)
	parser.add_argument('--stage', dest='stage',  help='Stage to run (dataprep, align, train, scores, evaluate), or \'all\' to run all stages', default=None)
	parser.add_argument('--heldout', action='store_true', help='Use this option to test on heldout set', default=False)

	args = parser.parse_args()

	run_all(args.config_yaml, args.stage, args.heldout)