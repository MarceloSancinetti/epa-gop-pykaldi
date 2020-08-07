from kaldi.asr import NnetLatticeFasterRecognizer
from kaldi.decoder import LatticeFasterDecoderOptions
from kaldi.util.table import  *
from kaldi.alignment import NnetAligner
from kaldi.fstext import SymbolTable
from kaldi.lat.align import WordBoundaryInfoNewOpts, WordBoundaryInfo
from kaldi.nnet3 import NnetSimpleComputationOptions

# Set the paths and read/write specifiers
model = '0013_librispeech_v1/exp/chain_cleaned/tdnn_1d_sp/final.mdl'
tree = '0013_librispeech_v1/exp/chain_cleaned/tdnn_1d_sp/tree'
lang_graph ='0013_librispeech_v1/data/lang_chain/L.fst' 
words = '0013_librispeech_v1/data/lang_chain/words.txt'
disam = '0013_librispeech_v1/data/lang_chain/phones/disambig.int'
phones = '0013_librispeech_v1/exp/chain_cleaned/tdnn_1d_sp/phones.txt'
word_boundary ='0013_librispeech_v1/data/lang_chain/phones/word_boundary.int' 
text = 'epadb/test/text' 

decodable_opts = NnetSimpleComputationOptions()
decodable_opts.acoustic_scale = 1.0
decodable_opts.frame_subsampling_factor = 3
decodable_opts.frames_per_chunk = 150

feats_rspec = ("ark:epadb/test/data/raw_mfcc_test.1.ark")
ivectors_rspec = ("ark:epadb/test/data/ivector_online.1.ark")

aligner = NnetAligner.from_files(model, tree, lang_graph, words,
                                 disam, decodable_opts=decodable_opts)
phones = SymbolTable.read_text(phones)
wb_info = WordBoundaryInfo.from_file(WordBoundaryInfoNewOpts(),
                                     "0013_librispeech_v1/data/lang_test_tgmed/phones/word_boundary.int")



with SequentialMatrixReader(feats_rspec) as f, \
 	 SequentialMatrixReader(ivectors_rspec) as i, open(text) as t:
	for (fkey, feats), (ikey, ivectors), line in zip(f, i, t):
		tkey, text = line.strip().split(None, 1)
		#assert(fkey == ikey == tkey)
		out = aligner.align((feats, ivectors), text)
		print(fkey, out["alignment"], flush=True)
		phone_alignment = aligner.to_phone_alignment(out["alignment"], phones)
		print(fkey, phone_alignment, flush=True)
		word_alignment = aligner.to_word_alignment(out["best_path"], wb_info)
		print(fkey, word_alignment, flush=True)



