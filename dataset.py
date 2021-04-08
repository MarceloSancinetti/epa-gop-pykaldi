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
        subset: str = None,
        #download: bool = False,
        audio_ext=".wav"
    ) -> None:
        self._ext_audio = audio_ext
        # if release in _RELEASE_CONFIGS.keys():
        #     folder_in_archive = _RELEASE_CONFIGS[release]["folder_in_archive"]
        #     url = _RELEASE_CONFIGS[release]["url"]
        #     subset = subset if subset else _RELEASE_CONFIGS[release]["subset"]
        # else:
        #     # Raise warning
        #     raise RuntimeError(
        #         "The release {} does not match any of the supported tedlium releases{} ".format(
        #             release, _RELEASE_CONFIGS.keys(),
        #         )
        #     )
        # if subset not in _RELEASE_CONFIGS[release]["supported_subsets"]:
        #     # Raise warning
        #     raise RuntimeError(
        #         "The subset {} does not match any of the supported tedlium subsets{} ".format(
        #             subset, _RELEASE_CONFIGS[release]["supported_subsets"],
        #         )
        #     )

        # Get string representation of 'root' in case Path object is passed
        root = os.fspath(root)

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
        # Create dict path for later read
        #self._dict_path = os.path.join(root, folder_in_archive, _RELEASE_CONFIGS[release]["dict"])
        #self._phoneme_dict = None

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

        transcript_path = os.path.join(path, speaker_id, "transcriptions", file_id)
        with open(transcript_path + ".lab") as f:
            transcript = f.readlines()[0]

        annotation_path = os.path.join(path, speaker_id, "labels", file_id)
        annotation = []
        with open(annotation_path + ".txt") as f:
            for line in f.readlines():
                line = line.split()
                target_phone = line[1]
                pronounced_phone = line[2]
                label = line[3]
                #start_time = line[4]   #Comentado hasta que Jaz cambie los labels para incluir start time
                start_time = 5 #dummy             
                annotation.append((target_phone, pronounced_phone, label, start_time))

        wave_path = os.path.join(path, speaker_id, "waveforms", file_id)
        features = get_features_for_logid(file_id)

        return (features, transcript, speaker_id, utterance_id, annotation)

   
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

