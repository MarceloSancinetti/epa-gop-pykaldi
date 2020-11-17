from kaldi.asr import MappedLatticeFasterRecognizer
from kaldi.decoder import LatticeFasterDecoderOptions
from kaldi.matrix import Matrix
from kaldi.util.table import SequentialMatrixReader, CompactLatticeWriter
from pytorch_models import *
import torch
import numpy as np


# Set the paths and read/write specifiers
acoustic_model_path = "model.pt"
transition_model_path = "0013_librispeech_v1/exp/chain_cleaned/tdnn_1d_sp/final.mdl"
graph_path = "HCLG.fst"
symbols_path = '0013_librispeech_v1/data/lang_chain/words.txt'

mfccs_rspec = ("ark:epadb/test/data/raw_mfcc_test.1.ark")
ivectors_rspec = ("ark:epadb/test/data/ivector_online.1.ark")

lat_wspec = "ark:| gzip -c > lat.gz"



# Instantiate the recognizer
decoder_opts = LatticeFasterDecoderOptions()
decoder_opts.beam = 13
decoder_opts.max_active = 7000
asr = MappedLatticeFasterRecognizer.from_files(
    transition_model_path, graph_path, symbols_path, decoder_opts=decoder_opts)

# Instantiate the PyTorch acoustic model (subclass of torch.nn.Module)
model = FTDNN()
model.load_state_dict(torch.load(acoustic_model_path))
model.eval()

# Extract the features, decode and write output lattices
with SequentialMatrixReader(mfccs_rspec) as mfccs_reader, \
 	 SequentialMatrixReader(ivectors_rspec) as ivectors_reader,\
     CompactLatticeWriter(lat_wspec) as lat_writer:
    for (mkey, mfccs), (ikey, ivectors) in zip(mfccs_reader, ivectors_reader):
        ivectors = np.repeat(ivectors, 10, axis=0)
        ivectors = ivectors[:mfccs.shape[0],:]
        x = np.concatenate((mfccs,ivectors), axis=1)
        x = np.expand_dims(x, axis=0)
        feats = torch.from_numpy(x)  # Convert to PyTorch tensor
        loglikes = model(feats)                  # Compute log-likelihoods
        loglikes = Matrix(loglikes.detach().numpy()[0])      # Convert to PyKaldi matrix
        out = asr.decode(loglikes)
        print(mkey, out["text"])
        lat_writer[mkey] = out["lattice"]