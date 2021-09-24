import yaml
import argparse
import os
from utils import *

def generate_scores_and_evaluate_epochs(config_dict, step):
	epochs = config_dict['epochs']

	swa_epochs = config_dict.get('swa_epochs', 0)
	swa_start  = epochs - swa_epochs
	
	for epoch in range(0, epochs+1, step):
		print("Evaluating epoch %d/%d" % (int(epoch/step), int(epochs/step)))
		run_generate_scores(config_dict, epoch=epoch)
		run_evaluate(config_dict, epoch=epoch)

		if epoch >= swa_start:
			run_generate_scores(config_dict, epoch=epoch, swa=True)
			run_evaluate(config_dict, epoch=epoch, swa=True)

def run_all(config_yaml, step):
	config_fh = open(config_yaml, "r")
	config_dict = yaml.safe_load(config_fh)	

	config_dict = extend_config_dict(config_yaml, config_dict)
	generate_scores_and_evaluate_epochs(config_dict, step)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--config', dest='config_yaml',  help='Path .yaml config file for experiment', default=None)
	parser.add_argument('--step', dest='step',  help='How many epochs between each comparison', type=int, default=None)

	args = parser.parse_args()

	run_all(args.config_yaml, args.step)
