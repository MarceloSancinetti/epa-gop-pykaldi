import sys
import yaml
import argparse
from src.DataprepStages import *
from src.Config import DataprepConfig
from IPython import embed

def run_all(config_yaml):

    config_dict = DataprepConfig(config_yaml).config_dict

    prep_stage   = PrepareFeaturesAndModelsStage(config_dict)
    align_stage  = ComplexStage([AlignCrossValStage(config_dict), AlignHeldoutStage(config_dict)], "align")
    labels_stage = ComplexStage([CreateLabelsCrossValStage(config_dict), CreateLabelsHeldoutStage(config_dict)], "labels")
    
    dataprep_stages = [prep_stage, align_stage, labels_stage]

    dataprep = ComplexStage(dataprep_stages, "dataprep")

    dataprep.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config_yaml',  help='Path .yaml config file for experiment', default=None)

    args = parser.parse_args()

    run_all(args.config_yaml)