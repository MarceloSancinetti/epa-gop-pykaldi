pu_list = []
pu_fh = open("experiments/sample_lists/problematic_utterances", "r")
for line in pu_fh.readlines():
    pu_list.append(line.strip())

utts = []
utt_list_fh = open("experiments/sample_lists/epadb_full_path_list.txt")
for line in utt_list_fh.readlines():
    utt, path = line.strip().split(' ')
    utts.append((utt, path))

clean_utts = []
for utt, path in utts:
    if utt not in pu_list:
        clean_utts.append((utt, path))

clean_utts_fh = open("experiments/sample_lists/clean_utts.txt", "w+")
for utt, path in clean_utts:
    clean_utts_fh.write(utt + ' ' + path + "\n")