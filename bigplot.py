import os
import pdb
import argparse

import numpy as np
import pandas as pd

import matplotlib
if "DISPLAY" in os.environ:
    matplotlib.use('TkAgg')
else:
    # Ignore Tk while running in server
    matplotlib.use('Agg')
import matplotlib.pyplot as plt



DATASETS    = ['citeseer', 'cora', 'hateful', 'pubmed']
CLASSIFIERS = ['wvrn', 'cc', 'sgc', 'gsage']

ymap = {
    'rs'        :   'RS',
    'kms'       :   'KMS',
    'ns_dc_h'   :   'NS-DC-H',
    'ns_ct_h'   :   'NS-CT-H',
    'es_rs'     :   'ES-RS',
    'fp'        :   'FeatProp',
    'wls_2'     :   'WLS-2',
    'wls_3'     :   'WLS-3',
    'ss'        :   'SS',
    'ffs'       :   'FFS',
    'ms'        :   'MS'
}

cmap = {
    'wvrn'  :   'wvRN',
    'cc'    :   'ICA',
    'sgc'   :   'SGC',
    'gsage' :   'GraphSage'
}


def plot_init(xlabel, ylabel, fsize=48):
    plt.rcParams["font.family"] = "Times New Roman"

    fig, axs = plt.subplots(len(DATASETS), len(CLASSIFIERS))
    fig.set_size_inches((16, 16))
    
    return fig, axs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', default='results/', help='Results directory', required=True)
    parser.add_argument('-c', default='5_224_32_8', help='Exp config', required=False)
    parser.add_argument('-f', default='png', help='Image format.', required=False)
    args = parser.parse_args()

    fig, axs = plot_init(xlabel='Budget(#nodes)', ylabel='Micro-F1 Score')

    font_size = 16
    linestyles = ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-']
    markers = ["o", "^", "s", "P", "D", ">", "1", "3", "*", "+", 'x']
    colors = ['blue', 'green', 'gold', 'red', 'purple', 'aqua', 'peru', 'magenta', 'grey', 'black', 'springgreen']

    xticks = None
    yticks = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    for i in range(len(DATASETS)):
        for j in range(len(CLASSIFIERS)):
            ds, clf = DATASETS[i], CLASSIFIERS[j]

            result_file = os.path.join(args.r, '%s_%s_%s.csv' % (clf, ds, args.c))
            df = pd.read_csv(result_file)

            error_file = os.path.join(args.r, '%s_%s_%s_error.csv' % (clf, ds, args.c))
            error = pd.read_csv(error_file) if os.path.isfile(error_file) else None

            columns = list(df.columns)
            xcol = columns[0]
            ycols = columns[1:]

            for p in range(len(ycols)):
                if error is not None:
                    axs[i, j].errorbar(x=xcol, y=ycols[p], data=df, yerr=error[ycols[p]], linewidth=1, linestyle=linestyles[p], color=colors[p], marker=markers[p], markersize=2, capsize=2)
                else:
                    axs[i, j].plot(xcol, ycols[p], data=df, linewidth=1, linestyle=linestyles[p], color=colors[p], marker=markers[p], markersize=4)

            axs[i, j].set_title('%s-%s' % (ds.capitalize(), cmap[clf]), fontsize=font_size)

            if xticks is None:
                xticks = df[xcol].values

    plt.setp(axs, xticks=xticks, yticks=yticks)
    for ax in axs.flat:
        ax.set_xlabel('Budget (#Nodes)', fontsize=font_size)
        ax.set_ylabel('Micro-F1 score', fontsize=font_size)
        ax.tick_params(labelsize=font_size)

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    handles, labels = axs[0][0].get_legend_handles_labels()
    legend_labels = [ymap['_'.join(l.split('_')[1:])] for l in labels]
    fig.legend(handles, legend_labels, bbox_to_anchor=(0.5, 1.05), loc='upper center', ncol=6, fontsize='x-large') #
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.05)

    save_params = {'fname': 'plots/bigplot.' + args.f, 'format': args.f, 'bbox_inches': 'tight'} #, 'pad_inches': 0.2, 'bbox_extra_artists': (lgd,), 
    # https://stackoverflow.com/questions/10101700/moving-matplotlib-legend-outside-of-the-axis-makes-it-cutoff-by-the-figure-box

    if args.f == 'eps':
        save_params['dpi'] = 2000

    plt.savefig(**save_params)


if __name__ == "__main__":
    args = main()