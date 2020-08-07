from kaldi.asr import NnetLatticeFasterRecognizer
from kaldi.decoder import LatticeFasterDecoderOptions
from kaldi.nnet3 import NnetSimpleComputationOptions
from kaldi.util.table import  *
# Set the paths and read/write specifiers
model_path = "final.mdl"
graph_path = "HCLG.fst"
symbols_path = "words.txt"
#feats_rspec = ("ark:compute-mfcc-feats --config=mfcc_hires.conf "
#               "scp:wav.scp ark:- |")
#ivectors_rspec = (feats_rspec + "ivector-extract-online2 "
#                 "--config=ivector_extractor.conf "
#
feats_rspec = ("ark:/home/marcelo/Desktop/pasantia/pykaldi/raw_mfcc_test.1.ark")
ivectors_rspec = ("ark:/home/marcelo/Desktop/pasantia/pykaldi/ivector_online.1.ark")
lat_wspec = "ark:| gzip -c > lat.gz"

# Instantiate the recognizer
decoder_opts = LatticeFasterDecoderOptions()
decoder_opts.beam = 13
decoder_opts.max_active = 7000
decodable_opts = NnetSimpleComputationOptions()
decodable_opts.acoustic_scale = 1.0
decodable_opts.frame_subsampling_factor = 3
asr = NnetLatticeFasterRecognizer.from_files(
    model_path, graph_path, symbols_path,
    decoder_opts=decoder_opts, decodable_opts=decodable_opts)

# Extract the features, decode and write output lattices
with SequentialMatrixReader(feats_rspec) as feats_reader, \
     SequentialMatrixReader(ivectors_rspec) as ivectors_reader, \
     CompactLatticeWriter(lat_wspec) as lat_writer:
    for (fkey, feats), (ikey, ivectors) in zip(feats_reader, ivectors_reader):
        assert(fkey == ikey)
        out = asr.decode((feats, ivectors))
        print(fkey, out["text"])
