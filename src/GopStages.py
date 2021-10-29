import os

import src.gop.calculate_gop as calculate_gop
from src.Stages import *


class GopStage(AtomicStage):
    _name = "gop"

    def run(self):
        calculate_gop.main(self._config_dict)