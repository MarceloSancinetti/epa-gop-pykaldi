import glob
import os

wav_scp_file= open("wav.scp","w+")

for file in glob.glob('EpaDB/*/waveforms/*'):
    fullpath = os.path.abspath(file)
    basename = os.path.splitext(os.path.basename(file))[0]
    wav_scp_file.write(basename + ' ' + fullpath + '\n')

wav_scp_file.close()