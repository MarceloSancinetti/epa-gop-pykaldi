import os
from IPython import embed

import src.gop.calculate_gop as calculate_gop
import src.evaluate.generate_data_for_eval as generate_data_for_eval
from src.Stages import *


class GopStage(AtomicStage):
    _name = "gop"

    def run(self):
        calculate_gop.main(self._config_dict)

class GopHeldoutStage(AtomicStage):
    _name = "gop"

    def run(self):
        config_dict = self._config_dict

        config_dict["utterance-list-path"] = config_dict["test-list-path"]
        config_dict["loglikes-path"]       = config_dict["loglikes-heldout-path"]
        config_dict["alignments-path"]     = config_dict["heldout-align-path"]
        
        calculate_gop.main(self._config_dict)

class EvaluateGopStage(AtomicStage):
    _name = "evalaute"

    def run(self):
        generate_data_for_eval.main(self._config_dict)