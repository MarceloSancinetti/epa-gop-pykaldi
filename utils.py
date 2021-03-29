import glob
import os
from kaldi.util.table import RandomAccessMatrixReader
import numpy as np
import torch

def extract_features_using_kaldi(data_path, mfcc_path, ivectors_path, wav_scp_path, spk2utt_path, conf_path):


	wav_scp_file = open(wav_scp_path,"w+")
	spk2utt_file = open(spk2utt_path,"w+")

	#Iterate over waveforms and write their names in wav.scp and spk2utt
	for file in glob.glob('EpaDB/*/waveforms/*'):
	    fullpath = os.path.abspath(file)
	    basename = os.path.splitext(os.path.basename(file))[0]
	    wav_scp_file.write(basename + ' ' + fullpath + '\n')
	    spkr = basename.split('_')[0]
	    spk2utt_file.write(spkr + ' ' + basename + '\n')

	wav_scp_file.close()
	spk2utt_file.close()


	if not os.path.isdir(data_path):
	        os.mkdir(data_path)


	if not os.path.exists(mfcc_path):
	    os.system('compute-mfcc-feats --config='+conf_path+'/mfcc_hires.conf \
	              scp,p:'+wav_scp_path+' ark:- | copy-feats \
	              --compress=true ark:- ark,scp:' + mfcc_path + ',feats.scp')



	if not os.path.exists(ivectors_path):
	    os.system('ivector-extract-online2 --config='+conf_path+'/ivector_extractor.conf ark:spk2utt \
	    	      scp:feats.scp ark:' + ivectors_path)


#Returns features (MFCCs+iVectors) for given logid in the format the acoustic model expects
def get_features_for_logid(logid, data_path='epadb/test/data', ivector_period=10):

    mfccs_rspec = ("ark:" + data_path + "/mfccs.ark")
    ivectors_rspec = ("ark:" + data_path + "/ivectors.ark")

    with RandomAccessMatrixReader(mfccs_rspec) as mfccs_reader, \
    RandomAccessMatrixReader(ivectors_rspec) as ivectors_reader:
        if not ivectors_reader.has_key(logid):
            raise Exception("iVectors for utterance "+logid+" not present. Did you extract the features?")
        if not mfccs_reader.has_key(logid):
            raise Exception("MFCCs for utterance "+logid+" not present. Did you extract the features?")
        mfccs = mfccs_reader[logid]
        ivectors = ivectors_reader[logid]

        ivectors = np.repeat(ivectors, ivector_period, axis=0) 
        ivectors = ivectors[:mfccs.shape[0],:]
        x = np.concatenate((mfccs,ivectors), axis=1)
        x = np.expand_dims(x, axis=0)
        feats = torch.from_numpy(x)

        return feats