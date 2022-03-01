from kaldi.matrix import Matrix
from kaldi.util.table import DoubleMatrixWriter
from kaldi.alignment import MappedAligner
from kaldi.fstext import SymbolTable
from kaldi.lat.align import WordBoundaryInfoNewOpts, WordBoundaryInfo
import torch
import numpy as np
import pickle
import tqdm
import argparse
import os
from src.utils.FeatureManager import FeatureManager
from src.utils.utils import makedirs_for_file
from src.pytorch_models.FTDNNAcoustic import *
from IPython import embed

def log_alignments(aligner, phones, alignment, logid, align_output_fh):
    phone_alignment = aligner.to_phone_alignment(alignment, phones)
    transition_lists = []
    for phone, start_time, duration in phone_alignment:
        transitions_for_phone = alignment[start_time : (start_time + duration)]
        transition_lists.append(transitions_for_phone)
    align_output_fh.write(logid + ' phones '      + str(phone_alignment)  + '\n')
    align_output_fh.write(logid + ' transitions ')
    for transition_list in transition_lists:
        align_output_fh.write(str(transition_list) + ' ')
    align_output_fh.write('\n')

def main(config_dict):
    sample_list_path      = config_dict['utterance-list-path']
    acoustic_model_path   = config_dict['acoustic-model-path']
    transition_model_path = config_dict['transition-model-path']
    tree_path             = config_dict['tree-path']
    disam_path            = config_dict['disam-path']
    word_boundary_path    = config_dict['word-boundary-path']
    lang_graph_path       = config_dict['lang-graph-path']
    words_path            = config_dict['words-path']
    phones_path           = config_dict['libri-phones-path']
    features_path         = config_dict['features-path']
    conf_path             = config_dict['features-conf-path']
    loglikes_path         = config_dict['loglikes-path']
    align_path            = config_dict['alignments-path']
    epadb_root_path       = config_dict['data-root-path']

    mfccs_rspec    = "ark:" + features_path + "/mfccs.ark"
    ivectors_rspec = "ark:" + features_path + "/ivectors.ark"

    loglikes_wspec = "ark:" + loglikes_path

    aligner = MappedAligner.from_files(transition_model_path, tree_path, lang_graph_path, words_path,
                                     disam_path, acoustic_scale = 1.0)
    phones  = SymbolTable.read_text(phones_path)
    wb_info = WordBoundaryInfo.from_file(WordBoundaryInfoNewOpts(),
                                         word_boundary_path)


    # Instantiate the PyTorch acoustic model (subclass of torch.nn.Module)
    model = FTDNNAcoustic()
    model.load_state_dict(torch.load(acoustic_model_path))
    model.eval()

    #Create feature manager
    feature_manager = FeatureManager(epadb_root_path, features_path, conf_path)

    makedirs_for_file(align_path)
    align_out_file = open(align_path,"w+")
    # Decode and write output lattices
    with DoubleMatrixWriter(loglikes_wspec) as loglikes_writer:
        for line in tqdm.tqdm(open(sample_list_path,'r').readlines()):
            logid = line.split()[0]
            feats = feature_manager.get_features_for_logid(logid)
            text =  feature_manager.get_transcription_for_logid(logid)
            text = text.upper()
            feats = torch.unsqueeze(feats, 0)
            loglikes = model(feats)                         # Compute log-likelihoods
            loglikes = Matrix(loglikes.detach().numpy()[0]) # Convert to PyKaldi matrix
            loglikes_writer[logid] = loglikes
            out = aligner.align(loglikes, text)
            log_alignments(aligner, phones, out["alignment"], logid, align_out_file)
            #phone_alignment = aligner.to_phone_alignment(out["alignment"], phones)
            #align_out_file.write(logid + ' phones ' + str(phone_alignment)  + '\n')
            #align_out_file.write(logid + ' transitions ' + str(out['alignment']) + '\n') 




