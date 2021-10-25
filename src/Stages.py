import os


class ComplexStage():

    def __init__(self, substages, config_dict, name):

        if len(substages) < 2:
            raise Exception('Must provide at least two substages for a complex stage')
    
        self._substages      = substages
        self._substage_names = [substage.name() for substage in substages]
        self._name           = name
        self._config_dict    = config_dict.copy()   
    
    def run(self, from_stage=None, to_stage=None):
        #Run substages from_stage:to_stage 
        if from_stage == None:
            from_stage = self._substage_names[0]

        if to_stage == None:
            to_stage = self._substage_names[-1]

        first_stage_index = self._substage_names.index(from_stage)
        to_stage_index    = self._substage_names.index(to_stage)
        for stage in self._substages[first_stage_index : to_stage_index + 1]:
            stage.run()
 

    def name(self):
        return self._name

class AtomicStage():

    def __init__(self, config_dict):
        #self._name        = name
        self._config_dict = config_dict.copy()

    def run(self):
        pass

    def name(self):
        return self._name