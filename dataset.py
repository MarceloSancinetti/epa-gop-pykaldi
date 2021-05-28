import os
import glob
from typing import Tuple, Union
from pathlib import Path

import torchaudio
from torch import Tensor
from torch.utils.data import Dataset
from torchaudio.datasets.utils import (
    download_url,
    extract_archive,
)
from typing import List
from utils import *

import pickle5


def collapse_target_phone(target_phone):
    phone_replacements = {}
    for phone_name in ['Th', 'Ph', 'Kh']:
        phone_replacement[phone_name] = phone_name[:-1]
    phone_replacements['AX'] = 'AH'
    phone_replacements['DX'] = 'T'
    target_phone = phone_replacements[target_phone]
    return target_phone



class EpaDB(Dataset):
    """
    Create a Dataset for EpaDB.

    Args:
        sample_list_path (str or Path): Path to the dataset sample list.
        root_path (str or Path): Path to where the EpaDB directory is found.
    """

    def __init__(
        self,
        root_path: Union[str, Path],
        sample_list_path: Union[str, Path],
        phones_list_path: Union[str, Path],
        labels_path: Union[str, Path],
        audio_ext=".wav"
    ) -> None:
        self._ext_audio = audio_ext

        # Get string representation of 'path' in case Path object is passed
        root_path = os.fspath(root_path)
        sample_list_path = os.fspath(sample_list_path)
        phones_list_path = os.fspath(phones_list_path)
        labels_path = os.fspath(labels_path)

        self._root_path = root_path
        self._labels_path = labels_path

        # Read from sample list and create dictionary mapping fileid to .wav path and file list mapping int to logid
        file_id_list = []
        logids_by_speaker = {}
        sample_list_fh = open(sample_list_path, "r")
        for line in sample_list_fh.readlines():
            line = line.split()
            logid = line[0]
            speaker_id = logid.split('_')[0]
            sample_path = line[1]
            file_id_list.append(logid)
            if speaker_id in logids_by_speaker:
                logids_by_speaker[speaker_id].append(logid)
            else:
                logids_by_speaker[speaker_id] = []
        self._filelist = file_id_list
        self._logids_by_speaker = logids_by_speaker

        #Define pure phone dictionary to map pure phone symbols to a label vector index  
        self._pure_phone_dict = {}
        #Open file that contains list of pure phones
        phones_list_fh = open(phones_list_path, "r")

        #Get phone number for each phone
        for i, phone_pure_name in enumerate(phones_list_fh.readlines()):
            self._pure_phone_dict[phone_pure_name.strip()] = i
            

    def _load_epa_item(self, file_id: str, path: str, labels_path: str) -> Tuple[Tensor, str, str, str, List[Tuple[str, str, str, int, int]]]:
        """Loads an EpaDB dataset sample given a file name and corresponding sentence name.

        Args:
            file_id (str): File id to identify both text and audio files corresponding to the sample
            path (dict): Dataset root path

        Returns:
            tuple: ``(features, transcript, speaker_id, utterance_id, annotation)``
                    annotation is List[(correct_phone, pronounced_phone, label, start_time, end_time]]
        """

        speaker_id = file_id.split("_")[0]
        utterance_id = file_id.split("_")[1]

        features = get_features_for_logid(file_id)

        transcript_path = os.path.join(path, speaker_id, "transcriptions", file_id)
        with open(transcript_path + ".lab") as f:
            transcript = f.readlines()[0]

        annotation_path = os.path.join(labels_path, speaker_id, file_id)
        annotation = []
        phone_count = self.phone_count()
        labels = np.zeros([features.shape[0], phone_count])
        phone_times = []

        with open(annotation_path + ".txt") as f:
            for line in f.readlines():
                line = line.split()
                target_phone = line[1]
                pronounced_phone = line[2]
                label = line[3]
                start_time = int(line[4])  
                end_time = int(line[5])
                try:
                    #These two if statements fix the mismatch between #frames in annotations and feature matrix
                    if end_time > features.shape[0]:
                        #Printear warning aca
                        if  end_time > features.shape[0] + 2:
                            raise Exception('End time in annotations longer than feature length by ' + str(features.shape[0] - end_time))
                        end_time = features.shape[0]
                    if start_time > end_time:
                        if  start_time > end_time + 2:
                            raise Exception('Start time in annotations longer than end time by ' + str(end_time - start_time))    
                        start_time = end_time

                    phone_times.append((target_phone, start_time, end_time))

                    #If the phone was mispronounced, put a -1 in the labels
                    if target_phone != pronounced_phone:
                        labels[start_time:end_time, self._pure_phone_dict[target_phone]] = np.full([end_time-start_time], -1)
                        #If the target phone is not defined, collapse it into similar Kaldi phone (i.e Th -> T)
                        if target_phone not in self._pure_phone_dict.keys():
                            target_phone = collapse_target_phone(target_phone)                    
                    #If the phone was pronounced correcly, put a 1 in the labels
                    if label == '+' and pronounced_phone != '0' :
                        labels[start_time:end_time, self._pure_phone_dict[target_phone]] = np.full([end_time-start_time], 1)
                        #If the target phone is not defined, collapse it into similar Kaldi phone (i.e Th -> T)
                        if target_phone not in self._pure_phone_dict.keys():
                            target_phone = collapse_target_phone(target_phone)

                except ValueError as e:
                    print("Bad item:")
                    print("Speaker: " + speaker_id)
                    print("Utterance: " + utterance_id)
                    print("#Frames in features: ")
                    print(features.shape[0])
                    print(line)
                    print(e)
                except KeyError as e:
                    print("Bad item:")
                    print("Speaker: " + speaker_id)
                    print("Utterance: " + utterance_id)
                    print("#Frames in features: ")
                    print(features.shape[0])
                    print(line)
                    print(e)

        return (features, transcript, speaker_id, utterance_id, torch.from_numpy(labels), phone_times)

   
    def __getitem__(self, n: int) -> Tuple[Tensor, str, str, str, List[Tuple[str, str, str, int, int]]]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            tuple: ``(features, transcript, speaker_id, utterance_id, annotation)``
                    annotation is List[(correct_phone, pronounced_phone, label, start_time, end_time]]        
        """
        fileid = self._filelist[n]
        return self._load_epa_item(fileid, self._root_path, self._labels_path)


    def __len__(self) -> int:
        """EpaDB dataset custom function overwritting len default behaviour.

        Returns:
            int: EpaDB dataset length
        """
        return len(self._filelist)

    def get_sample_indexes_from_spkr_indexes(self, spkr_indexes):
        """Takes list of valid indexes from the speaker list and returns
        list of logid indexes for the samples of said speakers. The logid
        indexes are positions in self._filelist.

        Returns:
            list: indexes of self._filelist
        """
        spkr_ids = self.get_speaker_list()
        spkr_ids = [spkr_ids[spkr_ix] for spkr_ix in spkr_indexes]
        logids_for_spkrs = []
        [logids_for_spkrs.extend(self.get_sample_logids_for_spkr(spkr_id)) for spkr_id in spkr_ids] 
        sample_indexes = [self._filelist.index(logid) for logid in logids_for_spkrs]
        return sample_indexes

    def get_sample_logids_for_spkr(self, spkr_id):
        """
        Returns:
            list: sample logid list for given spkr 
        """
        return self._logids_by_speaker[spkr_id]

    def get_speaker_list(self):
        """
        Returns:
            list: speaker list 
        """
        return list(self._logids_by_speaker.keys())


    def phone_count(self) ->int:
        """
        Returns:
            int: amount of phones in phone dictionary 
        """
        return len(self._pure_phone_dict.keys())

