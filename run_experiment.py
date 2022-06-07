import sys
import yaml
import argparse 
from src.utils.run_utils import get_eval_stage
from src.ExperimentStages import *
from src.Config import ExperimentConfig
from IPython import embed

def get_prep_stage(config_dict):
    prep_dir_stage   = CreateExperimentDirectoryStage(config_dict)
    prep_model_stage = CreateFinetuneModelStage(config_dict)
    return ComplexStage([prep_dir_stage, prep_model_stage], "prep")

def get_train_stage(config_dict):
    if config_dict.get("held-out"):
        return TrainHeldoutStage(config_dict)
    else:
        return TrainCrossValStage(config_dict)

def get_scores_stage(config_dict, epoch, is_swa=False):
    if config_dict.get("held-out"):
        return GenerateScoresHeldoutStage(config_dict, epoch=epoch, is_swa=is_swa)
    else:
        return GenerateScoresCrossValStage(config_dict, epoch=epoch, is_swa=is_swa)

def get_scores_and_eval_stages_for_many_epochs(config_dict, step):
    scores_stages = [] 
    eval_stages   = []

    epochs = config_dict['epochs']

    swa_epochs = config_dict.get('swa-epochs', 0)
    swa_start  = epochs - swa_epochs

    for epoch in range(0, epochs+1, step):
        scores_stages.append(get_scores_stage(config_dict, epoch))
        eval_stages.append(get_eval_stage(config_dict, epoch))

        if epoch >= swa_start and swa_epochs != 0:
            scores_stages.append(get_scores_stage(config_dict, epoch, is_swa=True))
            eval_stages.append(get_eval_stage(config_dict, epoch, is_swa=True))

    scores_stage = ComplexStage(scores_stages, "scores")
    eval_stage   = ComplexStage(eval_stages, "evaluate")

    return scores_stage, eval_stage

def run_all(config_yaml, from_stage, to_stage, device_name, use_heldout):

    config = ExperimentConfig(config_yaml, use_heldout, device_name)
    config_dict = config.config_dict
    checkpoint_step = config_dict['checkpoint-step']

    prep_stage  = get_prep_stage(config_dict)
    train_stage = get_train_stage(config_dict)
    scores_stage, eval_stage = get_scores_and_eval_stages_for_many_epochs(config_dict, checkpoint_step)

    experiment_stages = [prep_stage, train_stage, scores_stage, eval_stage]

    experiment = ComplexStage(experiment_stages, "experiment")

    experiment.run(from_stage, to_stage)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config_yaml',  help='Path .yaml config file for experiment', default=None)
    parser.add_argument('--from', dest='from_stage',  help='First stage to run (prep, train, scores, evaluate)', default=None)
    parser.add_argument('--to', dest='to_stage',  help='Last stage to run (prep, train, scores, evaluate)', default=None)
    parser.add_argument('--device', dest='device_name', help='Device name to use, such as cpu or cuda', default=None)
    parser.add_argument('--heldout', action='store_true', help='Use this option to test on heldout set', default=False)

    args = parser.parse_args()
    use_heldout = args.heldout

    run_all(args.config_yaml, args.from_stage, args.to_stage, args.device_name, use_heldout)