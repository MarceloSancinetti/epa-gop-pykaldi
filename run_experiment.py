import yaml
import argparse
import os

def generate_arguments(args_dict):
	res = ""
	for arg_name, value in args_dict.items():
		res = res + "--" + arg_name + " " + str(value) + " "
	return res

def run_data_prep(config_dict):
	args_dict = {"epa-root-path": config_dict["epadb-root-path"],
				 "features-path": config_dict["features-path"],
				 "conf-path":     config_dict["features-conf-path"],
				 "labels-path":   config_dict["epa-ref-labels-dir-path"]}
	arguments = generate_arguments(args_dict)
	os.system("python src/prepare_data.py " + arguments)

def run_align(config_dict):
	args_dict = {"utterance-list": 		  config_dict["utterance-list-path"],
				 "acoustic-model-path":   config_dict["acoustic-model-path"],
				 "transition-model-path": config_dict["transition-model-path"],
				 "tree-path": 			  config_dict["tree-path"],
				 "disam-path": 			  config_dict["disam-path"],
				 "word-boundary-path": 	  config_dict["word-boundary-path"],
				 "lang-graph-path": 	  config_dict["lang-graph-path"],
				 "words-path": 			  config_dict["words-path"],
				 "phones-path": 		  config_dict["libri-phones-path"],
				 "features-path":         config_dict["features-path"],
				 "conf-path":     		  config_dict["features-conf-path"],
				 "loglikes-path": 		  config_dict["loglikes-path"],
				 "align-path": 			  config_dict["alignments-path"],
				 "epadb-root-path": 	  config_dict["epadb-root-path"]
				 }

	arguments = generate_arguments(args_dict)
	os.system("python src/align_using_pytorch_am.py " + arguments)

def run_create_kaldi_labels(config_dict):
	pass

def run_train(config_dict):
	pass

def run_generate_scores(config_dict):
	pass

def run_evaluate(config_dict):
	pass

def run_experiment(config_yaml):
	config_fh = open(config_yaml, "r")
	config_dict = yaml.safe_load(config_fh)
	
	print("Running data preparation")
	run_data_prep(config_dict)
	print("Running aligner")
	run_align(config_dict)
	if config_dict['use-kaldi-labels']:
		print("Creating Kaldi labels")
		run_create_kaldi_labels(config_dict)
	print("Running training")
	run_train(config_dict)
	run_generate_scores(config_dict)
	run_evaluate(config_dict)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--config', dest='config_yaml',  help='Path .yaml config file for experiment', default=None)

	args = parser.parse_args()

	run_experiment(args.config_yaml)