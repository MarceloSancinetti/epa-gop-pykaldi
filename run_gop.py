import sys
import yaml
from src.utils.run_utils import *
from src.GopStages import *
from src.ExperimentStages import EvaluateScoresCrossValStage
from IPython import embed

def run_all(config_yaml, from_stage, to_stage, use_heldout):

    config_dict = load_extended_config_dict(config_yaml, "gop", use_heldout, "cpu")

    prepdir_stage = CreateExperimentDirectoryStage(config_dict)
    gop_stage     = GopStage(config_dict)
    eval_stage    = get_eval_stage(config_dict)

    gop_pipeline  = ComplexStage([prepdir_stage, gop_stage, eval_stage], "gop-pipeline")

    gop_pipeline.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config_yaml',  help='Path .yaml config file for experiment', default=None)
    parser.add_argument('--from', dest='from_stage',  help='First stage to run (prepdir, gop, evaluate)', default=None)
    parser.add_argument('--to', dest='to_stage',  help='Last stage to run (prepdir, gop, evaluate)', default=None)
    parser.add_argument('--heldout', action='store_true', help='Use this option to test on heldout set', default=False)

    args = parser.parse_args()
    use_heldout = args.heldout

    run_all(args.config_yaml, args.from_stage, args.to_stage, use_heldout)