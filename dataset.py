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

_RELEASE_CONFIGS = {
    "release1": {
        "folder_in_archive": "TEDLIUM_release1",
        "url": "http://www.openslr.org/resources/7/TEDLIUM_release1.tar.gz",
        "checksum": "30301975fd8c5cac4040c261c0852f57cfa8adbbad2ce78e77e4986957445f27",
        "data_path": "",
        "subset": "train",
        "supported_subsets": ["train", "test", "dev"],
        "dict": "TEDLIUM.150K.dic",
    },
    "release2": {
        "folder_in_archive": "TEDLIUM_release2",
        "url": "http://www.openslr.org/resources/19/TEDLIUM_release2.tar.gz",
        "checksum": "93281b5fcaaae5c88671c9d000b443cb3c7ea3499ad12010b3934ca41a7b9c58",
        "data_path": "",
        "subset": "train",
        "supported_subsets": ["train", "test", "dev"],
        "dict": "TEDLIUM.152k.dic",
    },
    "release3": {
        "folder_in_archive": "TEDLIUM_release-3",
        "url": "http://www.openslr.org/resources/51/TEDLIUM_release-3.tgz",
        "checksum": "ad1e454d14d1ad550bc2564c462d87c7a7ec83d4dc2b9210f22ab4973b9eccdb",
        "data_path": "data/",
        "subset": None,
        "supported_subsets": [None],
        "dict": "TEDLIUM.152k.dic",
    },
}

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
        root (str or Path): Path to the directory where the dataset is found or downloaded.
        subset (str, optional): The subset of dataset to use. Valid options are ``"train"``, ``"dev"``,
            and ``"test"``.
        download (bool, optional):
            Whether to download the dataset if it is not found at root path. (default: ``False``).
    """
    def __init__(
        self,
        root: Union[str, Path],
        phones_list: Union[str, Path],
        subset: str = None,
        #download: bool = False,
        audio_ext=".wav"
    ) -> None:
        self._ext_audio = audio_ext

        # Get string representation of 'root' in case Path object is passed
        root = os.fspath(root)
        phones_list = os.fspath(phones_list)

        #basename = os.path.basename(url)
        basename = "EpaDB"
        archive = os.path.join(root, basename)

        #basename = basename.split(".")[0]

        self._path = archive
        #if subset in ["train", "dev", "test"]:
            #self._path = os.path.join(self._path, subset)

        #if download:
        #    if not os.path.isdir(self._path):
        #        if not os.path.isfile(archive):
        #            checksum = _RELEASE_CONFIGS[release]["checksum"]
        #            download_url(url, root, hash_value=checksum)
        #        extract_archive(archive)

        # Create list for all samples
        filelist = []
        for file in sorted(glob.glob('EpaDB/*/waveforms/*')):
            if file.endswith(".wav"):
                fileid = os.path.basename(file).split('.')[0]
                filelist.append(fileid)
        self._filelist = filelist
        
        #Define pure phone dictionary to map pure phone symbols to a label vector index  
        self._pure_phone_dict = {}
        #Open file that contains list of pure phones
        phones_list_fh = open(phones_list, "r")

        #Get phone number for each phone
        for i, phone_pure_name in enumerate(phones_list_fh.readlines()):
            self._pure_phone_dict[phone_pure_name.strip()] = i
            

    def _load_epa_item(self, file_id: str, path: str) -> Tuple[Tensor, str, str, str, List[Tuple[str, str, str, int, int]]]:
        """Loads an EpaDB dataset sample given a file name and corresponding sentence name.

        Args:
            file_id (str): File id to identify both text and audio files corresponding to the sample
            path (str): Dataset root path

        Returns:
            tuple: ``(features, transcript, speaker_id, utterance_id, annotation)``
                    annotation is List[(correct_phone, pronounced_phone, label, start_time, end_time]]
        """

        speaker_id = file_id.split("_")[0]
        utterance_id = file_id.split("_")[1]

        wave_path = os.path.join(path, speaker_id, "waveforms", file_id)
        features = get_features_for_logid(file_id)

        transcript_path = os.path.join(path, speaker_id, "transcriptions", file_id)
        with open(transcript_path + ".lab") as f:
            transcript = f.readlines()[0]

        annotation_path = os.path.join(path, speaker_id, "labels", file_id)
        annotation = []
        phone_count = self.phone_count()
        labels = np.zeros([features.shape[0], phone_count])

        with open(annotation_path + ".txt") as f:
            for line in f.readlines():
                line = line.split()
                target_phone = line[1]
                pronounced_phone = line[2]
                label = line[3]
                start_time = int(line[4])  
                end_time = int(line[5])
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
                
        return (features, transcript, speaker_id, utterance_id, torch.from_numpy(labels))

   
    def __getitem__(self, n: int) -> Tuple[Tensor, str, str, str, List[Tuple[str, str, str, int, int]]]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            tuple: ``(features, transcript, speaker_id, utterance_id, annotation)``
                    annotation is List[(correct_phone, pronounced_phone, label, start_time, end_time]]        
        """
        fileid = self._filelist[n]
        return self._load_epa_item(fileid, self._path)


    def __len__(self) -> int:
        """EpaDB dataset custom function overwritting len default behaviour.

        Returns:
            int: EpaDB dataset length
        """
        return len(self._filelist)

    def phone_count(self) ->int:
        """
        Returns:
            int: amount of phones in phone dictionary 
        """
        return len(self._pure_phone_dict.keys())

