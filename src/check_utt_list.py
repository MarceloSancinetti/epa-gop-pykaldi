def generate_logid_list(utt_list_path):
    logid_list = []
    utt_list_fh = open(utt_list_path, "r")
    for line in utt_list_fh.readlines():
        logid_list.append(line[0])
    return sorted(logid_list)