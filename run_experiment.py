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

def get_model_name(config_dict, fold):
	last_epoch = config_dict["epochs"] - 1
	run_name   = config_dict["run-name"]
	return run_name + '-fold-' + str(fold) + '-epoch-' + str(last_epoch) #Aca hay codigo repetido entre el PATH de train y esto

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

def run_data_prep(config_dict):
	phone_count = get_phone_count(config_dict["phones-list-path"])
	args_dict = {"epa-root-path": 			config_dict["epadb-root-path"],
				 "features-path": 			config_dict["features-path"],
				 "conf-path":               config_dict["features-conf-path"],
				 "labels-path":             config_dict["epa-ref-labels-dir-path"],
				 "librispeech-models-path": config_dict["librispeech-models-path"],
				 "pytorch-models-path": 	config_dict["pytorch-models-path"],
				 
				 "libri-chain-mdl-path":    config_dict["librispeech-models-path"] + 
				 							config_dict["libri-chain-mdl-path"],

				 "libri-chain-txt-path":    config_dict["librispeech-models-path"] +
				 							config_dict["libri-chain-txt-path"],
				 							
				 "acoustic-model-path":     config_dict["acoustic-model-path"],
				 "finetune-model-path":     config_dict["finetune-model-path"],
				 "utterance-list-path":     config_dict["utterance-list-path"],
				 "phone-count":             phone_count,
				 "experiment-dir-path":     config_dict["experiment-dir-path"],
				 "setup":                   "exp"}
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
	phone_count = get_phone_count(config_dict["phones-list-path"])
	args_dict = {'reference-transcriptions-path': config_dict["epa-ref-labels-dir-path"] + "/reference_transcriptions.txt",
				 'utterance-list-path': 		  config_dict["utterance-list-path"],
				 'labels-dir-path': 			  config_dict["epa-ref-labels-dir-path"],
				 'alignments-path': 			  config_dict["alignments-path"],
				 'output-dir-path': 			  config_dict["kaldi-labels-path"],
				 'phones-list-path': 			  config_dict["phones-list-path"],
				 'phone-weights-path': 		      config_dict["phone-weights-path"],
				 'phone-count':                   phone_count
				 }
	run_script("src/create_kaldi_labels.py", args_dict)

def run_train(config_dict, device_name):
	args_dict = {"run-name": 			 config_dict["run-name"],
				 "utterance-list": 		 config_dict["utterance-list-path"],
				 "folds": 				 config_dict["folds"],
 				 "epochs": 				 config_dict["epochs"],
				 "layers": 		 		 config_dict["layers"],
				 "use-dropout": 		 config_dict["use-dropout"],
				 "dropout-p": 		     config_dict["dropout-p"],
				 "learning-rate":        config_dict["learning-rate"],
				 "batch-size":           config_dict["batch-size"],
				 "use-clipping":         config_dict["use-clipping"],
                 "use-first-batchnorm":  config_dict["use-first-batchnorm"],
				 "use-final-batchnorm":  config_dict["use-final-batchnorm"],
				 "phones-file": 		 config_dict["phones-list-path"],
				 "labels-dir": 			 config_dict["labels-dir"],
				 "model-path": 			 config_dict["finetune-model-path"],
				 "phone-weights-path":   config_dict["phone-weights-path"],
				 "epa-root-path": 		 config_dict["epadb-root-path"],
				 "features-path": 		 config_dict["features-path"],
				 "conf-path": 			 config_dict["features-conf-path"],
				 "test-sample-list-dir": config_dict["test-sample-list-dir"],
				 "state-dict-dir": 		 config_dict["state-dict-dir"],
				 "use-multi-process":    config_dict["use-multi-process"],
				 "device":               config_dict["device"]				 
				}
	run_script("src/train.py", args_dict)

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
					 "alignments-path": config_dict["alignments-path"],
				     "device":          config_dict["device"]				 
					}
		run_script("src/generate_score_txt.py", args_dict)
		cat_file_names += args_dict['gop-txt-dir'] + '/' +'gop-'+args_dict['model-name']+'.txt ' #Codigo repetido con generate_score_txt
	#Concatenate gop scores for all folds
	os.system("cat " + cat_file_names + " > " + config_dict["full-gop-score-path"])

def run_evaluate(config_dict):
	args_dict = {"transcription-file": config_dict["transcription-file"],
				 "utterance-list": 	   config_dict["utterance-list-path"],
				 "output-dir": 		   config_dict["eval-dir"],
				 "output-filename":    "data_for_eval.picke",
				 "gop-file": 		   config_dict["full-gop-score-path"],
				 "phones-pure-file":   config_dict["kaldi-phones-pure-path"],
				 "labels": 	   		   config_dict["labels-dir"]
				}
	run_script("src/evaluate/generate_data_for_eval.py", args_dict)


def run_all(config_yaml, stage, device_name):
	config_fh = open(config_yaml, "r")
	config_dict = yaml.safe_load(config_fh)	

	config_dict = extend_config_dict(config_yaml, config_dict)

	if stage in ["dataprep", "all"]:
		print("Running data preparation")
		run_data_prep(config_dict)

	if stage in ["align", "all"]:
		print("Running aligner")
		run_align(config_dict)
	
	if stage in ["train", "all"]:
		if config_dict['use-kaldi-labels']:
			print("Creating Kaldi labels")
			run_create_kaldi_labels(config_dict)
		print("Running training")
		run_train(config_dict, device_name)
	
	if stage in ["scores", "all"]:
		print("Generating GOP scores")
		run_generate_scores(config_dict, device_name)

	if stage in ["evaluate", "all"]:
		print("Evaluating results")
		run_evaluate(config_dict)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--config', dest='config_yaml',  help='Path .yaml config file for experiment', default=None)
	parser.add_argument('--stage', dest='stage',  help='Stage to run (dataprep, align, train, scores, evaluate), or \'all\' to run all stages', default=None)
    parser.add_argument('--device', dest='device_name', help='Device name to use, such as cpu or cuda', default=None)

	args = parser.parse_args()

	run_all(args.config_yaml, args.stage, args.device_name)
