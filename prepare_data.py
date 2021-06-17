import glob
import os
from FeatureManager import FeatureManager


features_path = 'epadb/test/data'
conf_path = 'conf'
epadb_root_path = 'EpaDB'
text_path = 'epadb/test/text'

feature_manager = FeatureManager(epadb_root_path, features_path, conf_path)

feature_manager.extract_features_using_kaldi(text_path)


#Create symbolic links to labels used in evaluation stage
for file in sorted(glob.glob('EpaDB/*/labels/*')):
    fullpath = os.path.abspath(file)
    basename = os.path.basename(file)
    #Get spkr id
    spkr = fullpath.split('/')[-3]
    labels_dir_for_spkr = 'evaluate/epadb_30/' + spkr+ '/labels/' 
    #Create directory for speaker's labels
    if not os.path.exists(labels_dir_for_spkr):
        os.system('mkdir -p ' + labels_dir_for_spkr)
    #Make symbolic link to speaker labels from EpaDB directory
    if not os.path.exists(labels_dir_for_spkr + '/' + basename):
        os.system('ln -s ' + fullpath + ' ' + labels_dir_for_spkr + '/')

#Handle symbolic links for reference transcriptions used in evaluation stage
if not os.path.exists('evaluate/epadb_30/reference_transcriptions.txt'):
    current_path = os.getcwd()
    print('ln -s ' + current_path + '/EpaDB/reference_transcriptions.txt ' + current_path + '/evaluate/epadb_30/reference_transcriptions.txt')
    os.system('ln -s ' + current_path + '/EpaDB/reference_transcriptions.txt ' + current_path + '/evaluate/epadb_30/reference_transcriptions.txt')


