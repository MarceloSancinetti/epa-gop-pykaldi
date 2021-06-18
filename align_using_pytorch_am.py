from kaldi.matrix import Matrix
from kaldi.util.table import DoubleMatrixWriter
from kaldi.alignment import MappedAligner
from kaldi.fstext import SymbolTable
from kaldi.lat.align import WordBoundaryInfoNewOpts, WordBoundaryInfo
from pytorch_models_old import *
import torch
import numpy as np
import pickle
import os
from FeatureManager import FeatureManager

# Set the paths and read/write specifiers
acoustic_model_path   = "model.pt"
transition_model_path = "exp/chain_cleaned/tdnn_1d_sp/final.mdl"
tree_path             = 'exp/chain_cleaned/tdnn_1d_sp/tree'
disam_path            = 'data/lang_test_tgsmall/phones/disambig.int'
lang_graph_path       = 'data/lang_test_tgsmall/L.fst' 
symbols_path          = 'data/lang_test_tgsmall/words.txt'
phones_path           = 'exp/chain_cleaned/tdnn_1d_sp/phones.txt'
data_path             = 'epadb/test/data'
conf_path             = 'conf'
sample_list_path      = 'epadb_full_path_list'
epadb_root_path       = 'EpaDB'

mfccs_rspec    = ("ark:" + data_path + "/mfccs.ark")
ivectors_rspec = ("ark:" + data_path + "/ivectors.ark")

loglikes_wspec = "ark:gop/loglikes.ark"

aligner = MappedAligner.from_files(transition_model_path, tree_path, lang_graph_path, symbols_path,
                                 disam_path, acoustic_scale = 1.0)
phones = SymbolTable.read_text(phones_path)
wb_info = WordBoundaryInfo.from_file(WordBoundaryInfoNewOpts(),
                                     "data/lang_test_tgsmall/phones/word_boundary.int")


# Instantiate the PyTorch acoustic model (subclass of torch.nn.Module)
model = FTDNN()
model.load_state_dict(torch.load(acoustic_model_path))
model.eval()

#Create feature manager
feature_manager = FeatureManager(epadb_root_path, data_path, conf_path)


align_out_file = open("gop/align_output","w+")
# Decode and write output lattices
with DoubleMatrixWriter(loglikes_wspec) as loglikes_writer:
    for line in open(sample_list_path,'r').readlines():
        logid = line.split()[0]
        feats, text = feature_manager.get_features_for_logid(logid)
        text = text.upper()
        feats = torch.unsqueeze(feats, 0)
        loglikes = model(feats)                         # Compute log-likelihoods
        loglikes = Matrix(loglikes.detach().numpy()[0]) # Convert to PyKaldi matrix
        loglikes_writer[logid] = loglikes
        out = aligner.align(loglikes, text)
        phone_alignment = aligner.to_phone_alignment(out["alignment"], phones)
        align_out_file.write(logid + ' phones ' + str(phone_alignment)  + '\n')
        align_out_file.write(logid + ' transitions ' + str(out['alignment']) + '\n') 
        #word_alignment = aligner.to_word_alignment(out["best_path"], wb_info)
