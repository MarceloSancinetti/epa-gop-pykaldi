from kaldi.asr import MappedLatticeFasterRecognizer
from kaldi.decoder import LatticeFasterDecoderOptions
from kaldi.matrix import Matrix
from kaldi.util.table import SequentialMatrixReader, DoubleMatrixWriter
from kaldi.nnet3 import NnetSimpleComputationOptions
from kaldi.alignment import MappedAligner
from kaldi.fstext import SymbolTable
from kaldi.lat.align import WordBoundaryInfoNewOpts, WordBoundaryInfo
from pytorch_models import *
import torch
import numpy as np
import pickle
import os

# Set the paths and read/write specifiers
acoustic_model_path = "model.pt"
transition_model_path = "exp/chain_cleaned/tdnn_1d_sp/final.mdl"
tree = 'exp/chain_cleaned/tdnn_1d_sp/tree'
disam = 'data/lang_test_tgsmall/phones/disambig.int'
lang_graph ='data/lang_test_tgsmall/L.fst' 
symbols_path = 'data/lang_test_tgsmall/words.txt'
phones = 'exp/chain_cleaned/tdnn_1d_sp/phones.txt'
text_path = 'epadb/test/text'
data_path = 'epadb/test/data'

mfccs_rspec = ("ark:" + data_path + "/mfccs.ark")

ivectors_rspec = ("ark:" + data_path + "/ivectors.ark")

loglikes_wspec = "ark:gop/loglikes.ark"

aligner = MappedAligner.from_files(transition_model_path, tree, lang_graph, symbols_path,
                                 disam, acoustic_scale = 1.0)
phones = SymbolTable.read_text(phones)
wb_info = WordBoundaryInfo.from_file(WordBoundaryInfoNewOpts(),
                                     "data/lang_test_tgsmall/phones/word_boundary.int")




# Instantiate the PyTorch acoustic model (subclass of torch.nn.Module)
model = FTDNN()
model.load_state_dict(torch.load(acoustic_model_path))
model.eval()


align_out_file = open("gop/align_output","w+")
# Extract the features, decode and write output lattices
with SequentialMatrixReader(mfccs_rspec) as mfccs_reader, \
 	 SequentialMatrixReader(ivectors_rspec) as ivectors_reader, open(text_path) as t, \
     DoubleMatrixWriter(loglikes_wspec) as loglikes_writer:
    for (mkey, mfccs), (ikey, ivectors), line in zip(mfccs_reader, ivectors_reader, t):
        if mkey != ikey:
            print("Algo anda mal")
        tkey, text = line.strip().split(None, 1)
        ivectors = np.repeat(ivectors, 10, axis=0)
        ivectors = ivectors[:mfccs.shape[0],:]
        x = np.concatenate((mfccs,ivectors), axis=1)
        x = np.expand_dims(x, axis=0)
        feats = torch.from_numpy(x)  # Convert to PyTorch tensor
        loglikes = model(feats)                  # Compute log-likelihoods
        loglikes = Matrix(loglikes.detach().numpy()[0])      # Convert to PyKaldi matrix
        loglikes_writer[mkey] = loglikes
        out = aligner.align(loglikes, text)
        phone_alignment = aligner.to_phone_alignment(out["alignment"], phones)
        #print(mkey + ' phones' + str(phone_alignment))
        #print(mkey + ' transitions' +str(out['alignment']))
        align_out_file.write(mkey + ' phones' + str(phone_alignment)  + '\n')
        align_out_file.write(mkey + ' transitions' + str(out['alignment']) + '\n') 
        #word_alignment = aligner.to_word_alignment(out["best_path"], wb_info)
