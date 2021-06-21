from kaldi.matrix import Matrix
from kaldi.util.table import DoubleMatrixWriter
from kaldi.alignment import MappedAligner
from kaldi.fstext import SymbolTable
from kaldi.lat.align import WordBoundaryInfoNewOpts, WordBoundaryInfo
from pytorch_models_old import *
import torch
import numpy as np
import pickle
import argparse
import os
from FeatureManager import FeatureManager



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--utterance-list', dest='sample_list_path', help='Path to file with utt list', default=None)
    parser.add_argument('--acoustic-model-path', dest='acoustic_model_path', help='Path to acoustic model .pt', default=None)
    parser.add_argument('--transition-model-path', dest='transition_model_path', help='Path to transition model .mdl', default=None)
    parser.add_argument('--tree-path', dest='tree_path', help='Path to tree', default=None)
    parser.add_argument('--disam-path', dest='disam_path', help='Path to disambig.int', default=None)
    parser.add_argument('--word-boundary-path', dest='word_boundary_path', help='Path to word_boundary.int', default=None)
    parser.add_argument('--lang-graph-path', dest='lang_graph_path', help='Path to language FST', default=None)
    parser.add_argument('--words-path', dest='symbols_path', help='Path to word list', default=None)
    parser.add_argument('--phones-path', dest='phones_path', help='Path to kaldi lang phones.txt', default=None)    
    parser.add_argument('--features-path', dest='features_path', help='Path to features directory', default=None)
    parser.add_argument('--conf-path', dest='conf_path', help='Path to directory containing config files for feature extraction', default=None)
    parser.add_argument('--loglikes-path', dest='loglikes_path', help='Output path to save loglikes.ark', default=None)
    parser.add_argument('--align-path', dest='align_path', help='Path to save alignment output', default=None)
    parser.add_argument('--epadb-root-path', dest='epadb_root_path', help='EpaDB root path', default=None)

    args = parser.parse_args()


    # Set the paths and read/write specifiers
    acoustic_model_path   = args.acoustic_model_path
    transition_model_path = args.transition_model_path
    tree_path             = args.tree_path
    disam_path            = args.disam_path
    word_boundary_path    = args.word_boundary_path
    lang_graph_path       = args.lang_graph_path
    symbols_path          = args.symbols_path
    phones_path           = args.phones_path
    features_path         = args.features_path
    conf_path             = args.conf_path
    sample_list_path      = args.sample_list_path
    epadb_root_path       = args.epadb_root_path

    mfccs_rspec    = "ark:" + features_path + "/mfccs.ark"
    ivectors_rspec = "ark:" + features_path + "/ivectors.ark"

    loglikes_wspec = "ark:" + args.epadb_root_path

    aligner = MappedAligner.from_files(transition_model_path, tree_path, lang_graph_path, symbols_path,
                                     disam_path, acoustic_scale = 1.0)
    phones  = SymbolTable.read_text(phones_path)
    wb_info = WordBoundaryInfo.from_file(WordBoundaryInfoNewOpts(),
                                         )


    # Instantiate the PyTorch acoustic model (subclass of torch.nn.Module)
    model = FTDNN()
    model.load_state_dict(torch.load(acoustic_model_path))
    model.eval()

    #Create feature manager
    feature_manager = FeatureManager(epadb_root_path, features_path, conf_path)


    align_out_file = open(args.align_path,"w+")
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
