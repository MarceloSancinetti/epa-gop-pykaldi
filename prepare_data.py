import glob
import os
from FeatureManager import FeatureManager


features_path = 'epadb/test/data'
conf_path = 'conf'
epadb_root_path = 'EpaDB'

feature_manager = FeatureManager(features_path, conf_path)

feature_manager.extract_features_using_kaldi(epadb_root_path)


#Create text file with transcriptions of all phrases
text_path = 'epadb/test/text'
text_file = open(text_path, 'w+')
for file in sorted(glob.glob(epadb_root_path + '/*/waveforms/*')):
    fullpath = os.path.abspath(file)
    basename = os.path.splitext(os.path.basename(file))[0]
    transcription_path = epadb_root_path + '/' + spkr + '/transcriptions/' + basename +'.lab'
    transcription_fh = open(transcription_path, 'r')
    transcription = transcription_fh.readline().upper()
    text_file.write(basename + ' ' + transcription + '\n')


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


