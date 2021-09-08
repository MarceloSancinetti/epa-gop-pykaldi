import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq
from scipy import interpolate
import joblib
from IPython import embed
from sklearn.metrics import roc_curve, auc
import pandas as pd
import os, sys
from IPython import embed


def load_data_for_eval(path):
        df = joblib.load(path)
        df = df.rename(columns={"phone_automatic": "phones_names", "gop_scores": "gop_score"})
        return df            

def compare_models(dataframes, labels, output, colors):

    systems_count = len(dataframes)

    phones = dataframes[0].phones_names.unique()
    
    p_fa_count = 0
    fontsize = 30
    cota = 30

    phones_to_plot = []
    for phone in phones:
        targets     = [list(df.loc[(df['phones_names'] == phone) & (df['label'] == 1)].gop_score) for df in dataframes]
        non_targets = [list(df.loc[(df['phones_names'] == phone) & (df['label'] == 0)].gop_score) for df in dataframes]
        if all([len(tar) >= cota for tar in targets]) and all([len(non) >= cota for non in non_targets]):
            phones_to_plot.append(phone)

    nplots = len(phones_to_plot)
    fig, axs = plt.subplots(nplots, 2, figsize = (16, nplots*8))

    for phidx, phone in enumerate(phones_to_plot):
                
        targets     = [list(df.loc[(df['phones_names'] == phone) & (df['label'] == 1)].gop_score) for df in dataframes]
        non_targets = [list(df.loc[(df['phones_names'] == phone) & (df['label'] == 0)].gop_score) for df in dataframes]
            
        labels_target     = [list(df.loc[(df['phones_names'] == phone) & (df['label'] == 1)].label) for df in dataframes]
        labels_non_target = [list(df.loc[(df['phones_names'] == phone) & (df['label'] == 0)].label) for df in dataframes]
            
        all_samples = [targets[i] + non_targets[i] for i in range(systems_count)]

        rocs = [roc_curve(labels_target[i]+labels_non_target[i], targets[i]+non_targets[i]) for i in range(systems_count)]
        fprs = [x[0] for x in rocs]
        tprs = [x[1] for x in rocs]
        thrs = [x[2] for x in rocs]
        
        roc_aucs = [auc(fprs[i], tprs[i]) for i, _ in enumerate(fprs)]
        
        eers = [brentq(lambda x: 1. - x - interpolate.interp1d(fprs[i], tprs[i])(x), 0., 1.) for i,_ in enumerate(fprs)]
        
        histograms = [np.histogram(all_samples[i], bins=20) for i in range(systems_count)]
        h = [x[0] for x in histograms]
        e = [x[1] for x in histograms]
        c = [(e[i][:-1]+e[i][1:])/2 for i in range(systems_count)] # centros de los bins
        ht = [np.histogram(targets[i],     bins=e[i])[0] for i in range(systems_count)]
        hn = [np.histogram(non_targets[i], bins=e[i])[0] for i in range(systems_count)]

        ax1 = axs[phidx,0] 
        ax2 = axs[phidx,1] 
            
        for i in range(systems_count):
            legend = labels[i]+' (AUC = %0.2f ' % roc_aucs[i] + ' EER =  %0.2f '%eers[i]+ ' cant+ =  %0.2i '%len(targets[i])+ ' cant- =  %0.2i '%len(non_targets[i])+')'
            ax1.plot(fprs[i], tprs[i], color=colors[i],label=legend)                
            ax2.plot(c[i], ht[i]*1.0/np.sum(ht[i]), color=colors[i], label=labels[i]) 
            ax2.plot(c[i], hn[i]*1.0/np.sum(hn[i]), '--', color=colors[i])
            
        ax1.legend()
        ax2.legend()
        ax1.plot([0, 1], [0, 1], color='black', linestyle='--')
        ax1.set_title(phone)
        ax2.set_title(phone)
    fig.tight_layout()                                               
    fig.savefig(output)



table = sys.argv[1]
output = table+".pdf"
colors = ["darkorange", "green", "cyan", "violet", "red", "darkblue"]
#if not os.path.isdir(output_dir):
#   os.makedirs(output_dir)
                    
table_data = np.array([l.strip().split() for l in open(table).readlines()])
data_frames = []
for name, path in table_data:
   data_frames.append(load_data_for_eval(path))

compare_models(data_frames, table_data[:,0], output, colors)

