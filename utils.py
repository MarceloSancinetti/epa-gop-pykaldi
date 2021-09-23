import yaml
import argparse
import os
from IPython import embed

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

def get_model_name(config_dict, fold, epoch=None, use_heldout=False):
	if epoch == None:
		epoch  = config_dict["epochs"]
	run_name   = config_dict["run-name"]
	
	if use_heldout:
		fold_identifier = ''
	else:
		fold_identifier = '-fold-' + str(fold)
	
	return run_name +  fold_identifier + '-epoch-' + str(epoch) #Aca hay codigo repetido entre el PATH de train y esto

def get_test_sample_list_path_for_fold(test_sample_list_dir, fold):
	return test_sample_list_dir + "/test_sample_list_fold_" + str(fold) #Aca tmb codigo repetido

def run_data_prep(config_dict, setup):
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
				 "utterance-list-path":     config_dict["utterance-list-path"],
				 "experiment-dir-path":     config_dict["experiment-dir-path"],
				 "setup":                   setup}
	
	if setup == 'exp':
		phone_count = get_phone_count(config_dict["phones-list-path"])
		args_dict.update({"heldout":                 config_dict["held-out"],
						  "heldout-root-path":       config_dict["heldout-root-path"],
						  "heldout-list-path":       config_dict["test-list-path"],
						  "finetune-model-path":     config_dict["finetune-model-path"],
			   			  "phone-count":             phone_count,
		 				  "batchnorm":     			 config_dict["batchnorm"],
         				  "seed":                    42})

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

	if config_dict.get("held-out"):
		args_dict['epadb-root-path']               = config_dict['heldout-root-path']
		args_dict['utterance-list']                = config_dict['test-list-path']
		args_dict['align-path']                    = config_dict['heldout-align-path']

		run_script("src/align_using_pytorch_am.py", args_dict)

def run_evaluate(config_dict, epoch=''):
	if config_dict.get("held-out"):
		run_evaluate_heldout(config_dict, epoch=epoch)
	else:
		run_evaluate_kfold(config_dict, epoch=epoch)

def run_evaluate_kfold(config_dict, epoch=''):

	args_dict = {"transcription-file": config_dict["transcription-file"],
				 "utterance-list": 	   config_dict["utterance-list-path"],
				 "output-dir": 		   config_dict["eval-dir"],
				 "output-filename":    "data_for_eval_epoch" + str(epoch) + ".pickle",
				 "gop-file": 		   config_dict["full-gop-score-path"],
				 "phones-pure-file":   config_dict["kaldi-phones-pure-path"],
				 "labels": 	   		   config_dict["labels-dir"]
				}
	run_script("src/evaluate/generate_data_for_eval.py", args_dict)

def run_evaluate_heldout(config_dict, epoch=''):
	model_name = get_model_name(config_dict, 0, epoch=epoch, use_heldout=True)

	args_dict = {"transcription-file": config_dict["transcription-file"],
				 "utterance-list": 	   config_dict["test-list-path"],
				 "output-dir": 		   config_dict["eval-dir"],
				 "output-filename":    "data_for_eval_epoch" + str(epoch) + ".pickle",
				 "gop-file": 		   config_dict["gop-scores-dir"] + 'gop-'+model_name+'.txt',
				 "phones-pure-file":   config_dict["kaldi-phones-pure-path"],
				 "labels": 	   		   config_dict["labels-dir"]
				}
	run_script("src/evaluate/generate_data_for_eval.py", args_dict)

def run_create_kaldi_labels(config_dict, setup):
	args_dict = {'reference-transcriptions-path': config_dict["epa-ref-labels-dir-path"] + "/reference_transcriptions.txt",
				 'utterance-list-path': 		  config_dict["utterance-list-path"],
				 'labels-dir-path': 			  config_dict["epa-ref-labels-dir-path"],
				 'alignments-path': 			  config_dict["alignments-path"],
				 'output-dir-path': 			  config_dict["kaldi-labels-path"]
				}
	if setup == 'exp':
		phone_count = get_phone_count(config_dict["phones-list-path"])
		args_dict.update({'phones-list-path': 			  config_dict["phones-list-path"],
				 	  	  'phone-weights-path': 		  config_dict["phone-weights-path"],
				 	      'phone-count':                  phone_count
						 })
	
	run_script("src/create_kaldi_labels.py", args_dict)
	
	if config_dict.get("held-out"):
		args_dict['reference-transcriptions-path'] = config_dict["heldout-root-path"] + "/reference_transcriptions.txt"
		args_dict['utterance-list-path']           = config_dict['test-list-path']
		args_dict['labels-dir-path']               = config_dict['heldout-root-path']
		args_dict['alignments-path']               = config_dict['heldout-align-path']
		run_script("src/create_kaldi_labels.py", args_dict)

def run_generate_scores(config_dict, epoch=None):
	if config_dict.get("held-out"):
		run_generate_scores_heldout(config_dict, epoch=epoch)
	else:
		run_generate_scores_kfold(config_dict, epoch=epoch)

def run_generate_scores_kfold(config_dict, epoch=None):
	cat_file_names = ""
	for fold in range(config_dict["folds"]):
		model_name = get_model_name(config_dict, fold, epoch=epoch, use_heldout=False)
		args_dict = {"state-dict-dir":  config_dict["state-dict-dir"],
					 "model-name": 	    model_name,
					 "epa-root": 	    config_dict["epadb-root-path"],
					 "sample-list":     get_test_sample_list_path_for_fold(config_dict["test-sample-list-dir"], fold),
					 "phone-list":      config_dict["phones-list-path"],
					 "labels-dir":      config_dict["labels-dir"],
					 "gop-txt-dir":     config_dict["gop-scores-dir"],
	 				 "gop-txt-name":    'gop-'+model_name+'.txt',
					 "features-path":   config_dict["features-path"],
					 "conf-path":       config_dict["features-conf-path"],
			         "device":          "cpu",
                     "batchnorm":       config_dict["batchnorm"]
					}
		run_script("src/generate_score_txt.py", args_dict)
		cat_file_names += args_dict['gop-txt-dir'] + '/' +'gop-'+args_dict['model-name']+'.txt ' #Codigo repetido con generate_score_txt
	#Concatenate gop scores for all folds
	os.system("cat " + cat_file_names + " > " + config_dict["full-gop-score-path"])

def run_generate_scores_heldout(config_dict, epoch=None):
	model_name = get_model_name(config_dict, 0, epoch=epoch, use_heldout=True)
	args_dict = {"state-dict-dir":  config_dict["state-dict-dir"],
				 "model-name": 	    model_name,
				 "epa-root": 	    config_dict["heldout-root-path"],
				 "sample-list":     config_dict["test-list-path"],
				 "phone-list":      config_dict["phones-list-path"],
				 "labels-dir":      config_dict["labels-dir"],
				 "gop-txt-dir":     config_dict["gop-scores-dir"],
				 "gop-txt-name":    'gop-'+model_name+'.txt',
				 "features-path":   config_dict["features-path"],
				 "conf-path":       config_dict["features-conf-path"],
		         "device":          "cpu",
                 "batchnorm":       config_dict["batchnorm"]
				}
	run_script("src/generate_score_txt.py", args_dict)

def extend_config_dict(config_yaml, config_dict, use_heldout):
	config_dict["experiment-dir-path"] 	 = get_experiment_directory(config_yaml, use_heldout=use_heldout)
	config_dict["run-name"] 			 = get_run_name(config_yaml, use_heldout=use_heldout)
	config_dict["test-sample-list-dir"]  = config_dict["experiment-dir-path"] 	 + "test_sample_lists/"
	config_dict["train-sample-list-dir"] = config_dict["experiment-dir-path"] 	 + "train_sample_lists/"
	config_dict["state-dict-dir"] 		 = config_dict["experiment-dir-path"] 	 + "state_dicts/"
	config_dict["gop-scores-dir"] 		 = config_dict["experiment-dir-path"] 	 + "gop_scores/"
	config_dict["eval-dir"] 			 = config_dict["experiment-dir-path"] 	 + "eval/"
	config_dict["alignments-path"]       = config_dict["experiment-dir-path"] 	 + "align_output"
	config_dict["heldout-align-path"]    = config_dict["experiment-dir-path"]    + "align_output_heldout"	
	config_dict["loglikes-path"]         = config_dict["experiment-dir-path"] 	 + "loglikes.ark"
	config_dict["transcription-file"]    = config_dict["epa-ref-labels-dir-path"] + "reference_transcriptions.txt"
	config_dict["held-out"]              = use_heldout

	#Choose labels dir
	if config_dict["use-kaldi-labels"]:
		config_dict["labels-dir"] = config_dict["kaldi-labels-path"]
	else:
		config_dict["labels-dir"] = config_dict["epa-ref-labels-dir-path"]

	if "utterance-list-path" not in config_dict:
		config_dict["utterance-list-path"] = config_dict["train-list-path"]
	if "train-list-path" not in config_dict:
		config_dict["train-list-path"] = config_dict["utterance-list-path"]

	if not use_heldout:
		config_dict["full-gop-score-path"] 	 = config_dict["gop-scores-dir"] + "gop-all-folds.txt"

	#If only one layer will be trained or finetune model path is not defined, make finetune model path relative to experiment dir
	if config_dict["layers"] == 1 or "finetune-model-path" not in config_dict:
		config_dict["finetune-model-path"]   = config_dict["experiment-dir-path"] + "/model_finetuning_kaldi.pt"


	return config_dict