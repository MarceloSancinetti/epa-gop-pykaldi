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

def get_model_name(config_dict, fold, epoch=config_dict["epochs"] - 1):
	run_name   = config_dict["run-name"]
	return run_name + '-fold-' + str(fold) + '-epoch-' + str(epoch) #Aca hay codigo repetido entre el PATH de train y esto

def get_test_sample_list_path_for_fold(test_sample_list_dir, fold):
	return test_sample_list_dir + "/test_sample_list_fold_" + str(fold) #Aca tmb codigo repetido

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

	#Choose labels dir
	if config_dict["use-kaldi-labels"]:
		config_dict["labels-dir"] = config_dict["kaldi-labels-path"]
	else:
		config_dict["labels-dir"] = config_dict["epa-ref-labels-dir-path"]

	return config_dict

def run_create_kaldi_labels(config_dict):
	args_dict = {'reference-transcriptions-path': config_dict["epa-ref-labels-dir-path"] + "/reference_transcriptions.txt",
				 'utterance-list-path': 		  config_dict["utterance-list-path"],
				 'labels-dir-path': 			  config_dict["epa-ref-labels-dir-path"],
				 'alignments-path': 			  config_dict["alignments-path"],
				 'output-dir-path': 			  config_dict["kaldi-labels-path"]
				}
	run_script("src/create_kaldi_labels.py", args_dict)

def run_generate_scores(config_dict):
	cat_file_names = ""
	for fold in range(config_dict["folds"]):
		args_dict = {"state-dict-dir":  config_dict["state-dict-dir"],
					 "model-name": 	    get_model_name(config_dict, fold),
					 "epa-root": 	    config_dict["epadb-root-path"],
					 "sample-list":     get_test_sample_list_path_for_fold(config_dict["test-sample-list-dir"], fold),
					 "phone-list":      config_dict["phones-list-path"],
					 "labels-dir":      config_dict["labels-dir"],
					 "gop-txt-dir":     config_dict["gop-scores-dir"],
					 "features-path":   config_dict["features-path"],
					 "conf-path":       config_dict["features-conf-path"],
					 "alignments-path": config_dict["alignments-path"]
					}
		run_script("src/generate_score_txt.py", args_dict)
		cat_file_names += args_dict['gop-txt-dir'] + '/' +'gop-'+args_dict['model-name']+'.txt ' #Codigo repetido con generate_score_txt
	#Concatenate gop scores for all folds
	os.system("cat " + cat_file_names + " > " + config_dict["full-gop-score-path"])

def run_evaluate(config_dict, epoch=''):

	args_dict = {"transcription-file": config_dict["transcription-file"],
				 "utterance-list": 	   config_dict["utterance-list-path"],
				 "output-dir": 		   config_dict["eval-dir"],
				 "output-filename":    "data_for_eval_epoch" + str(epoch),
				 "gop-file": 		   config_dict["full-gop-score-path"],
				 "phones-pure-file":   config_dict["kaldi-phones-pure-path"],
				 "labels": 	   		   config_dict["labels-dir"]
				}
	run_script("src/evaluate/generate_data_for_eval.py", args_dict)


def run_all(config_yaml, step):
	config_fh = open(config_yaml, "r")
	config_dict = yaml.safe_load(config_fh)	

	config_dict = extend_config_dict(config_yaml, config_dict)

	epochs = config_dict['epochs']

	if config_dict['use-kaldi-labels']:
		print("Creating Kaldi labels")
		run_create_kaldi_labels(config_dict)
	
	for epoch in range(0, epochs, step):
		print("Evaluating epoch %s/%s", str(epoch/step), str(epochs/step))
		run_generate_scores(config_dict, epoch=epoch)
		run_evaluate(config_dict, epoch=epoch)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--config', dest='config_yaml',  help='Path .yaml config file for experiment', default=None)
	parser.add_argument('--step', dest='step',  help='How many epochs between each comparison', default=None)

	args = parser.parse_args()

	run_all(args.config_yaml, args.step)