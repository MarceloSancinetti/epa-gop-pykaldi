import argparse
from sklearn.model_selection import KFold
from dataset import EpaDB
from finetuning_utils import generate_fileid_list_and_spkr2logid_dict

def write_sample_list(epadb_root_path, sample_list_path, speaker_indexes, spkr_list, logids_by_speaker):
    sample_list_fh = open(sample_list_path, "w+")
    for spkr_idx in speaker_indexes:
        spkr_id = spkr_list[spkr_idx]
        for logid in logids_by_speaker[spkr_id]:
            sample_path = epadb_root_path + '/' + spkr_id + '/waveforms/' + logid + '.wav'
            sample_list_fh.write(logid + ' ' + sample_path + '\n')


def main(config_dict):
    
    utterance_list_path   = config_dict["utterance-list-path"]
    folds                 = config_dict["folds"]
    epadb_root_path       = config_dict["epadb-root-path"]
    train_sample_list_dir = config_dict["train-sample-list-dir"]
    test_sample_list_dir  = config_dict["test-sample-list-dir"]
    seed = 42

    kfold = KFold(n_splits=folds, shuffle=True, random_state = seed)
    _, logids_by_speaker = generate_fileid_list_and_spkr2logid_dict(utterance_list_path)
    spkr_list = list(logids_by_speaker.keys())
    for fold, (train_spkr_indexes, test_spkr_indexes) in enumerate(kfold.split(spkr_list)):
        train_list_path = train_sample_list_dir + 'train_sample_list_fold_' + str(fold)
        test_list_path  = test_sample_list_dir  + 'test_sample_list_fold_'  + str(fold)

        write_sample_list(epadb_root_path, train_list_path, train_spkr_indexes, spkr_list, logids_by_speaker)
        write_sample_list(epadb_root_path, test_list_path,  test_spkr_indexes,  spkr_list, logids_by_speaker)


