import os

import gop.calculate_gop
from Stages import *


class GopStage(AtomicStage):
    _name = "gop"

    def run(self):
        gop.calculate_gop.main(self._config_dict)