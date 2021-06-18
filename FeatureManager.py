import glob
import os
from kaldi.util.table import RandomAccessMatrixReader
import numpy as np
import torch



class FeatureManager:
	def __init__(self, epadb_root_path, features_path, conf_path):
		self.epadb_root_path = epadb_root_path
		self.features_path = features_path
		self.mfcc_path = features_path + '/mfccs.ark'
		self.ivectors_path = features_path + '/ivectors.ark'
		self.wav_scp_path = features_path + '/wav.scp'
		self.spk2utt_path = features_path + '/spk2utt'
		self.feats_scp_path = features_path + '/feats.scp'
		self.conf_path = conf_path

		self.ivector_period = self._read_ivector_period_from_conf()


	def extract_features_using_kaldi(self):

		if not os.path.isdir(self.features_path):
			os.mkdir(self.features_path)

		wav_scp_file = open(self.wav_scp_path,"w+")
		spk2utt_file = open(self.spk2utt_path,"w+")

		#Iterate over waveforms and write their names in wav.scp and spk2utt
		#Also, create text file with transcriptions of all phrases
		for file in glob.glob(self.epadb_root_path + '/*/waveforms/*'):
			fullpath = os.path.abspath(file)
			logid = os.path.splitext(os.path.basename(file))[0]
			wav_scp_file.write(logid + ' ' + fullpath + '\n')
			spkr = logid.split('_')[0]
			spk2utt_file.write(spkr + ' ' + logid + '\n')

		wav_scp_file.close()
		spk2utt_file.close()


		if not os.path.exists(self.mfcc_path):
			os.system('compute-mfcc-feats --config='+self.conf_path+'/mfcc_hires.conf \
					   scp,p:' + self.wav_scp_path+' ark:- | copy-feats \
					   --compress=true ark:- ark,scp:' + self.mfcc_path + ',' + self.feats_scp_path)


		if not os.path.exists(self.ivectors_path):
			os.system('ivector-extract-online2 --config='+self.conf_path+'/ivector_extractor.conf ark:' + self.spk2utt_path + '\
		    	      scp:' + self.feats_scp_path + ' ark:' + self.ivectors_path)



	#Returns features (MFCCs+iVectors) for given logid in the format the acoustic model expects
	def get_features_for_logid(self, logid):

		if not os.path.isfile(self.mfcc_path):
			raise Exception("MFCCs for utterance "+logid+" not found. Did you extract the features?")

		if not os.path.isfile(self.ivectors_path):
			raise Exception("iVectors for utterance "+logid+" not found. Did you extract the features?")


		transcription = self._get_transcription_for_logid(logid)

		mfccs_rspec = ("ark:" + self.mfcc_path)
		ivectors_rspec = ("ark:" + self.ivectors_path)

		with RandomAccessMatrixReader(mfccs_rspec) as mfccs_reader, \
		RandomAccessMatrixReader(ivectors_rspec) as ivectors_reader:
			if not ivectors_reader.has_key(logid):
				raise Exception("iVectors for utterance "+logid+" not found. Did you extract the features?")
			if not mfccs_reader.has_key(logid):
				raise Exception("MFCCs for utterance "+logid+" not found. Did you extract the features?")
			mfccs = mfccs_reader[logid]
			ivectors = ivectors_reader[logid]
			ivectors = np.repeat(ivectors, self.ivector_period, axis=0) 
			ivectors = ivectors[:mfccs.shape[0],:]
			x = np.concatenate((mfccs,ivectors), axis=1)
			#x = np.expand_dims(x, axis=0)
			feats = torch.from_numpy(x)

			return feats, transcription.strip()

	def _read_ivector_period_from_conf(self):
		conf_fh = open(self.conf_path + '/ivector_extractor.conf')
		ivector_period_line = conf_fh.readlines()[1]
		ivector_period = int(ivector_period_line.split('=')[1])
		return ivector_period

	def _get_transcription_for_logid(self, logid):
		spkr = logid.split('_')[0]
		transcription_path = self.epadb_root_path + '/' + spkr + '/transcriptions/' + logid +'.lab'

		if not os.path.isfile(transcription_path):
			raise Exception("Transcription file for logid " + logid + " not found in path " + transcription_path + ".")

		with open(transcription_path, 'r') as transcription_fh:
			transcription = transcription_fh.readlines()[0]
		return transcription