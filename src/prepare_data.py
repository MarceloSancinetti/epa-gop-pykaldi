import glob
import os
from FeatureManager import FeatureManager


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epa-root-path', dest='epa_root_path', help='EpaDB root path', default=None)
    parser.add_argument('--features-path', dest='features_path', help='Path to features directory', default=None)
    parser.add_argument('--conf-path', dest='conf_path', help='Path to config directory used in feature extraction', default=None)
    parser.add_argument('--labels-path', dest='labels_path', help='Path to create symlinks to EpaDB ref labels', default=None)

    args = parser.parse_args()


    features_path = args.features_path
    conf_path = args.conf_path
    epadb_root_path = args.epa_root_path

    feature_manager = FeatureManager(epadb_root_path, features_path, conf_path)

    feature_manager.extract_features_using_kaldi()


    #Create symbolic links to labels used in evaluation stage
    for file in sorted(glob.glob(epa_root_path + '*/labels/*')):
        fullpath = os.path.abspath(file)
        basename = os.path.basename(file)
        #Get spkr id
        spkr = fullpath.split('/')[-3]
        labels_dir_for_spkr = args.labels_path + spkr+ '/labels/' 
        #Create directory for speaker's labels
        if not os.path.exists(labels_dir_for_spkr):
            os.system('mkdir -p ' + labels_dir_for_spkr)
        #Make symbolic link to speaker labels from EpaDB directory
        if not os.path.exists(labels_dir_for_spkr + '/' + basename):
            os.system('ln -s ' + fullpath + ' ' + labels_dir_for_spkr + '/')

    #Handle symbolic links for reference transcriptions used in evaluation stage
    if not os.path.exists(args.labels_path + '/reference_transcriptions.txt'):
        current_path = os.getcwd()
        cmd = 'ln -s ' + current_path + epadb_root_path + '/reference_transcriptions.txt ' + current_path + args.labels_path + '/reference_transcriptions.txt'
        print(cmd)
        os.system(cmd)


