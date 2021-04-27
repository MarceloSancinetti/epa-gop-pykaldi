import glob
import os

data_path = 'epadb/test/data'
mfcc_path =  data_path + '/mfccs.ark'
ivectors_path =  data_path + '/ivectors.ark'


wav_scp_file = open("wav.scp","w+")
spk2utt_file = open("spk2utt","w+")

for file in sorted(glob.glob('EpaDB/*/waveforms/*')):
    fullpath = os.path.abspath(file)
    basename = os.path.splitext(os.path.basename(file))[0]
    wav_scp_file.write(basename + ' ' + fullpath + '\n')
    spkr = basename.split('_')[0]
    spk2utt_file.write(spkr + ' ' + basename + '\n')


if not os.path.isdir(data_path) or not os.path.exists('feats.scp'):
        os.mkdir(data_path)


if not os.path.exists(mfcc_path):
    os.system('compute-mfcc-feats --config=conf/mfcc_hires.conf \
              scp,p:wav.scp ark:- | copy-feats \
              --compress=true ark:- ark,scp:' + mfcc_path + ',feats.scp')



if not os.path.exists(ivectors_path):
    os.system('ivector-extract-online2 --config=conf/ivector_extractor.conf ark:spk2utt \
    	      scp:feats.scp ark:' + ivectors_path)

wav_scp_file.close()
spk2utt_file.close()



#Handle symbolic links for labels used in evaluation stage
if not os.path.exists('evaluate/epadb_30/reference_transcriptions.txt'):
    os.system('ln -s EpaDB/reference_transcriptions.txt evaluate/epadb_30/reference_transcriptions.txt')

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