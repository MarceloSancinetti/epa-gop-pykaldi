import yaml
import argparse
import os

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

def get_run_name(config_yaml):
	return os.path.basename(config_yaml).split('.')[0]

def get_experiment_directory(config_yaml):
	return "experiments/" + get_run_name(config_yaml) + '/'

def get_test_sample_list_path_for_fold(test_sample_list_dir, fold):
	return test_sample_list_dir + "/test_sample_list_fold_" + str(fold) #Aca tmb codigo repetido

def extend_config_dict(config_yaml, config_dict):
	config_dict["experiment-dir-path"] 	= get_experiment_directory(config_yaml)
	#config_dict["gop-dir"] 			= config_dict["experiment-dir-path"] 	 	 + "gop_scores/"
	config_dict["eval-dir"] 			= config_dict["experiment-dir-path"] 	 + "eval/"
	config_dict["alignments-path"]      = config_dict["experiment-dir-path"] 	 + "align_output"
	config_dict["loglikes-path"]        = config_dict["experiment-dir-path"] 	 + "loglikes.ark"
	config_dict["transcription-file"]   = config_dict["epa-ref-labels-dir-path"] + "reference_transcriptions.txt"
	config_dict["gop-score-path"]       = config_dict["experiment-dir-path"] 	 + "gop.txt"

	#Choose labels dir
	if config_dict["use-kaldi-labels"]:
		config_dict["labels-dir"] = config_dict["kaldi-labels-path"]
	else:
		config_dict["labels-dir"] = config_dict["epa-ref-labels-dir-path"]

	return config_dict

def run_data_prep(config_dict):
	args_dict = {"epa-root-path": 			config_dict["epadb-root-path"],
				 "features-path": 			config_dict["features-path"],
				 "conf-path":               config_dict["features-conf-path"],
				 "labels-path":             config_dict["epa-ref-labels-dir-path"],
				 "librispeech-models-path": config_dict["librispeech-models-path"],
				 
				 "libri-chain-mdl-path":    config_dict["librispeech-models-path"] + 
				 							config_dict["libri-chain-mdl-path"],

				 "libri-chain-txt-path":    config_dict["librispeech-models-path"] +
				 							config_dict["libri-chain-txt-path"],
				 							
				 "acoustic-model-path":     config_dict["acoustic-model-path"],
				 "utterance-list-path":     config_dict["utterance-list-path"],
				 "experiment-dir-path":     config_dict["experiment-dir-path"],
				 "setup":                   "gop"}
	run_script("src/prepare_data.py", args_dict)

def run_align(config_dict):
	args_dict = {"utterance-list": 		  config_dict["utterance-list-path"],
				 "acoustic-model-path":   config_dict["acoustic-model-path"],

				 "transition-model-path": config_dict["librispeech-models-path"] 
				 						  + config_dict["transition-model-path"],
				 
				 "tree-path": 			  config_dict["librispeech-models-path"] 
				 						  + config_dict["tree-path"],
				 
				 "disam-path": 			  config_dict["librispeech-models-path"] 
				 						  + config_dict["disam-path"],
				 
				 "word-boundary-path": 	  config_dict["librispeech-models-path"] 
				 						  + config_dict["word-boundary-path"],
				 
				 "lang-graph-path": 	  config_dict["librispeech-models-path"] 
				 						  + config_dict["lang-graph-path"],

				 "words-path": 			  config_dict["librispeech-models-path"] 
				 						  + config_dict["words-path"],

				 "phones-path": 		  config_dict["librispeech-models-path"] 
				 						  + config_dict["libri-phones-path"],

				 "features-path":         config_dict["features-path"],
				 "conf-path":     		  config_dict["features-conf-path"],	 
				 "loglikes-path": 		  config_dict["loglikes-path"],
				 "align-path": 			  config_dict["alignments-path"],
				 "epadb-root-path": 	  config_dict["epadb-root-path"]
				 }

	run_script("src/align_using_pytorch_am.py", args_dict)

def run_create_kaldi_labels(config_dict):
	args_dict = {'reference-transcriptions-path': config_dict["epa-ref-labels-dir-path"] + "/reference_transcriptions.txt",
				 'utterance-list-path': 		  config_dict["utterance-list-path"],
				 'labels-dir-path': 			  config_dict["epa-ref-labels-dir-path"],
				 'alignments-path': 			  config_dict["alignments-path"],
				 'output-dir-path': 			  config_dict["kaldi-labels-path"]
				}
	run_script("src/create_kaldi_labels.py", args_dict)

def run_gop(config_dict):
	args_dict = {'libri-phones-path': 			  config_dict["librispeech-models-path"] 
												  + config_dict['libri-phones-path'],

				 'libri-phones-to-pure-int-path': config_dict['libri-phones-to-pure-int-path'],

				 'libri-phones-pure-path': 		  config_dict['libri-phones-pure-path'],

				 'transition-model-path': 		  config_dict["librispeech-models-path"] 
												  + config_dict['transition-model-path'],

				 'gop-dir': 					  config_dict['experiment-dir-path'],
				 'loglikes-path': 				  config_dict['loglikes-path'],
				 'alignments-dir-path':			  config_dict['experiment-dir-path']
				}
	run_script("src/gop/calculate_gop.py", args_dict)

def run_evaluate(config_dict):
	args_dict = {"transcription-file": config_dict["transcription-file"],
				 "utterance-list": 	   config_dict["utterance-list-path"],
				 "output-dir": 		   config_dict["eval-dir"],
				 "gop-file": 		   config_dict["gop-score-path"],
				 "phones-pure-file":   config_dict["libri-phones-pure-path"],
				 "labels": 	   		   config_dict["labels-dir"]
				}
	run_script("src/evaluate/generate_data_for_eval.py", args_dict)


def run_all(config_yaml, stage):
	config_fh = open(config_yaml, "r")
	config_dict = yaml.safe_load(config_fh)	

	config_dict = extend_config_dict(config_yaml, config_dict)

	if stage in ["dataprep", "all"]:
		print("Running data preparation")
		run_data_prep(config_dict)

	if stage in ["align", "all"]:
		print("Running aligner")
		run_align(config_dict)
	
	if stage in ["gop", "all"]:
		if config_dict['use-kaldi-labels']:
			print("Creating Kaldi labels")
			run_create_kaldi_labels(config_dict)
		run_gop(config_dict)
			
	if stage in ["evaluate", "all"]:
		print("Evaluating results")
		run_evaluate(config_dict)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--config', dest='config_yaml',  help='Path .yaml config file for experiment', default=None)
	parser.add_argument('--stage', dest='stage',  help='Stage to run (dataprep, align, train, scores, evaluate), or \'all\' to run all stages', default=None)

	args = parser.parse_args()

	run_all(args.config_yaml, args.stage)