import argparse
from sklearn.model_selection import KFold
from dataset import EpaDB
from finetuning_utils import generate_fileid_list_and_spkr2logid_dict

def write_sample_list(sample_list_path, speaker_indexes, spkr_list, logids_by_speaker):
    sample_list_fh = open(sample_list_path, "w+")
    for spkr_idx in speaker_indexes:
        spkr_id = spkr_list[spkr_idx]
        for logid in logids_by_speaker[spkr_id]:
            sample_path = args.epa_root_path + '/' + spkr_id + '/waveforms/' + logid + '.wav'
            sample_list_fh.write(logid + ' ' + sample_path + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--utterance-list-path', dest='utterance_list_path',  help='Path to EpaDB sample list', default=None)
    parser.add_argument('--folds', dest='folds', help='How many folds to use', type=int, default=None)
    parser.add_argument('--epadb-root-path', dest='epa_root_path', help='EpaDB root path', default=None)
    parser.add_argument('--train-sample-list-dir', dest='train_sample_list_dir', help='Path where trainset sample lists will be generated', default=None)
    parser.add_argument('--test-sample-list-dir', dest='test_sample_list_dir', help='Path where testset sample lists will be generated', default=None)

    args = parser.parse_args()


    folds = args.folds
    seed = 42
    kfold = KFold(n_splits=folds, shuffle=True, random_state = seed)
    _, logids_by_speaker = generate_fileid_list_and_spkr2logid_dict(args.utterance_list_path)
    spkr_list = list(logids_by_speaker.keys())
    for fold, (train_spkr_indexes, test_spkr_indexes) in enumerate(kfold.split(spkr_list)):
        train_list_path = args.train_sample_list_dir + 'train_sample_list_fold_' + str(fold)
        test_list_path  = args.test_sample_list_dir  + 'test_sample_list_fold_'  + str(fold)

        write_sample_list(train_list_path, train_spkr_indexes, spkr_list, logids_by_speaker)
        write_sample_list(test_list_path,  test_spkr_indexes,  spkr_list, logids_by_speaker)


