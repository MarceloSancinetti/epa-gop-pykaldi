import os

import align
import create_kaldi_labels
import prepare_data
from Stages import *
from run_utils import *

class PrepareFeaturesAndModelsStage(AtomicStage):
    _name = "features"

    def run(self):
        prepare_data.main(self._config_dict)

class AlignCrossValStage(AtomicStage):
    _name = "align"

    def run(self):
        align.main(self._config_dict)

class AlignHeldoutStage(AtomicStage):
    _name = "align"

    def run(self):
        config_dict = self._config_dict

        config_dict['epadb-root-path']     = config_dict['heldout-root-path']
        config_dict['utterance-list-path'] = config_dict['test-list-path']
        config_dict['alignments-path']     = config_dict['heldout-align-path']
        align.main(config_dict)

class CreateLabelsCrossValStage(AtomicStage):
    _name = "labels"

    def run(self):
        config_dict = self._config_dict

        config_dict['labels-dir-path'] = config_dict['kaldi-labels-path']
        create_kaldi_labels.main(config_dict)

class CreateLabelsHeldoutStage(AtomicStage):
    _name = "labels"

    def run(self):
        config_dict = self._config_dict
        config_dict['reference-trans-path'] = config_dict["heldout-root-path"] + "/reference_transcriptions.txt"
        config_dict['utterance-list-path']  = config_dict['test-list-path']
        config_dict['labels-dir-path']      = config_dict['heldout-root-path']
        config_dict['alignments-path']      = config_dict['heldout-align-path']
        create_kaldi_labels.main(config_dict)
