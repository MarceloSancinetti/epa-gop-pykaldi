import os
from pathlib import Path
import textgrids
import pandas as pd
from pathlib import Path

def makedirs_for_file(acoustic_model_path):
    path = Path(acoustic_model_path)
    if not os.path.exists(path.parent):
        os.makedirs(path.parent)

def parse_textgrid(file):
    """
    this function uses textgrids library to parse textgrid files
    input: a textgrid file 
    output: a list with logid, phone, start and end time for the textgrid file
    """

    try:
        tg = textgrids.TextGrid(file)
    except:
        raise Exception(f'Failed on file {file}')

    p =  Path(file)
    logid = p.parts[1] + "_" + p.stem
    utt = []
    pos = 1
    for annot in tg['phones']:
        unit = []
        if annot.text not in ('sil', 'sp', ''):
            phone = str(annot.text)
            start = str(annot.xmin)
            end = str(annot.xmax)
            unit = (logid, pos, phone, start, end)
            utt.append(unit)
            pos = pos+1
        
    return utt 


def get_gop_alignments(path_filename):
    """
    function to parse Kaldis gop.1.txt file with phones and associated gop scores
    input: path to gop.1.txt file
    output: a dataframe with logid, phone and gop score
    """
    output_scores = []
    output_phones = []

    for line in open(path_filename).readlines():
        l=line.split()

        if len(l) < 2:
            print("Invalid line")
        else:
            logid = l[0].replace("'", "")
            data = l[1:]
            i = 0
            phones = []
            gop_scores = []

            while i < len(data):
                if data[i] == "[":
                    phone = int(data[i+1])
                    gop_score = float(data[i+2])
                    phones.append(phone)
                    gop_scores.append(gop_score)

                    i = i + 4

            output_scores.append({'logid': str(logid),
                            'gop_score': gop_scores})
            output_phones.append({'logid': str(logid),
                            'phones': phones})
            
            
            
    df_gop = pd.DataFrame(output_scores).set_index("logid")
    df_phones = pd.DataFrame(output_phones).set_index("logid")
    df_final = df_gop.join(df_phones)
    df_final = df_final.apply(pd.Series.explode).reset_index()

            
    return df_final



def phones2dic(path):
    """
    function that maps Kaldis logical phones to Kaldis pure phones
    input: path to phones_pure.txt
    output: a dictionary with logical phones as keys and pure phones as values
    """
    phones_dic = {}
    with open(path, "r") as fileHandler:
        line = fileHandler.readline()
        while line:
            l=line.split()
            phones_dic[int(l[1])] = l[0]
            line = fileHandler.readline()

    return phones_dic



def pos(df):
    """
    function that adds positions to dataframes
    input: any dataframe
    output: the same dataframe but changed
    """
    dfs=[]
    df['pos'] = 0
    df_g = df.groupby('logid')
    for group, df_ in df_g:
        df_ = df_.copy()
        df_.loc[:,'pos'] = np.arange(1, len(df_)+1, dtype=int)
        dfs.append(df_)
        df_ali_gop = pd.concat(dfs)
        df_ali_gop['id'] = df_ali_gop['logid'] + '_' + df_ali_gop['pos'].apply(str)
        df_ali_gop
    return df


def generate_arguments(args_dict):
    """
    Takes a dictionary of (argument_name, argument_value)
    and generates a string like --arg1name arg1value --arg2name arg2value ....
    to run a python script
    input: an argument dictinary
    output: a string of arguments as passed to a python script 
    """
    res = ""
    for arg_name, value in args_dict.items():
        res = res + "--" + arg_name + " " + str(value) + " "
    return res

def run_script(script, args_dict):
    """
    Runs a python script with arguments given by args_dict
    input: path to a python script and an argument dictinary
    output: None 
    """
    arguments = generate_arguments(args_dict)
    return os.system("python " + script + " " + arguments)
