import os
import pdb
import argparse
import matplotlib
if "DISPLAY" in os.environ:
    matplotlib.use('TkAgg')
else:
    # Ignore Tk while running in server
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from sklearn.metrics import auc



ymap = {
    'rs'        :   'RS',
    'kms'       :   'KMS',
    'ms'        :   'MS',
    'ns_dc_h'   :   'NS-DC-H',
    'ns_ct_h'   :   'NS-CT-H',
    'es_rs'     :   'ES-RS',
    'fp'        :   'FeatProp',
    'wls_2'     :   'WLS-2',
    'wls_3'     :   'WLS-3',
    'ss'        :   'SS',
    'ffs'       :   'FFS',
    'alfnet'    :   'ALFNET'
}


def plot_init(fsize, xlabel, ylabel):
    plt.figure(figsize=(20,16))
    plt.rc('legend', fontsize=fsize)
    plt.rc('xtick',labelsize=fsize)
    plt.rc('ytick',labelsize=fsize)
    plt.rcParams["font.family"] = "Times New Roman"

    plt.xlabel(xlabel, fontsize=fsize)
    plt.ylabel(ylabel, fontsize=fsize)


def draw_plot(df, xlabel, ylabel, filename, fmt='eps'):

    df['No gain'] = np.zeros(df.shape[0])

    columns = list(df.columns)
    xcol = columns[0]
    ycols = columns[1:]

    plot_init(fsize=48, xlabel=xlabel, ylabel=ylabel)

    legend_handles = []
    legends = []
    linestyles = [':', '--', '-.', '-', ':']
    ls = 0
    for ycol in ycols:
        params = {'linewidth':8, 'linestyle':linestyles[ls]}
        if ycol == 'No gain':
            params = {'linewidth':8, 'linestyle':linestyles[ls], 'marker':'*', 'color':'gray', 'markersize':16}
        line, = plt.plot(xcol, ycol, data=df, **params)
        legend_handles.append(line)
        legends.append(ycol)
        ls += 1

    plt.legend(handles=legend_handles, labels=legends, loc='upper right', prop={'size': 36})
    plt.margins(x=0, y=0.03)
    
    if fmt == 'eps':
        plt.savefig(filename, format='eps', dpi=2000, bbox_inches='tight')
    else:
        plt.savefig(filename, format=fmt, bbox_inches='tight')


def draw_multi_column(df, num_plots, labels, xlabel, ylabel, filename, fmt='eps'):
    columns = list(df.columns)
    
    xcols = columns[:num_plots]
    ycols = columns[num_plots:]

    plot_init(fsize=48, xlabel=xlabel, ylabel=ylabel)
    
    legend_handles = []
    linestyles = ['-', ':', '--', '-.']
    ls = 0
    for i in range(num_plots):
        # df[xcols[i]] = df[xcols[i]] * 60
        line, = plt.plot(xcols[i], ycols[i], data=df, linewidth=10, linestyle=linestyles[ls])
        legend_handles.append(line)
        ls += 1

    labels = ['_'.join(col.split('_')[1:]) for col in ycols]
    plt.legend(handles=legend_handles, labels=labels, loc='lower right')

    if fmt == 'eps':
        plt.savefig(filename, format='eps', dpi=2000, bbox_inches='tight')
    else:
        plt.savefig(filename, format=fmt, bbox_inches='tight')



def draw_multi_y_column(df, num_plots, xlabel, ylabel, filename, fmt='eps', error_df=None):
    columns = list(df.columns)
    
    xcol = columns[0]
    ycols = columns[1:]

    ax = plot_init(fsize=48, xlabel=xlabel, ylabel=ylabel)
    
    legend_handles = []
    linestyles = ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-']
    markers = ["o", "^", "s", "P", "D", ">", "1", "3", "*", "+", 'x']
    colors = ['blue', 'green', 'gold', 'red', 'purple', 'aqua', 'peru', 'magenta', 'grey', 'black', 'springgreen']
    # colors = ['peru', 'magenta', 'springgreen', 'royalblue']  # one-shot vs multi-shot
    ls = 0
    for i in range(num_plots):
        # df[xcols[i]] = df[xcols[i]] * 60
        line, = plt.plot(xcol, ycols[i], data=df, linewidth=4, linestyle=linestyles[ls], color=colors[ls], marker=markers[ls], markersize=12)
        line = plt.errorbar(data=df, x=xcol, y=ycols[i], yerr=error_df[ycols[i]], capsize=10, linewidth=2, linestyle=linestyles[ls], color=colors[ls], marker=markers[ls], markersize=12)
        legend_handles.append(line)
        ls += 1

    axes = plt.gca()
    legend_loc = ''
    if ylabel == 'Micro-F1':
        axes.set_ylim([-0.06, 0.95])
        # axes.set_ylim([0.47, 0.74]) # one-shot vs multi-shot
        legend_loc = 'lower right'
    elif ylabel == 'Execution Time (m)':
        axes.set_ylim([-0.05, 8])
        # axes.set_ylim([-1, 79]) # one-shot vs multi-shot
        legend_loc = 'upper left'
    else:
        axes.set_ylim([-5, 200])
    plt.xticks(df[xcol])

    labels = [ymap['_'.join(yl.split('_')[1:])] for yl in ycols]
    plt.legend(handles=legend_handles, labels=labels, fontsize=34, ncol=4, loc=legend_loc)

    if fmt == 'eps':
        plt.savefig(filename, format='eps', dpi=2000, bbox_inches='tight')
    else:
        plt.savefig(filename, format=fmt, bbox_inches='tight')



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', default='results/attr.cat01.graph00_5_200_8_ft1.00_nh3.csv', help='Results csv.')
    parser.add_argument('-fmt', default='eps', help='Image format.')
    parser.add_argument('-t', type=int, default=0, help='Is this a timing plot?.')
    parser.add_argument('-xl', default='Budget(%)', help='Label for X axis.')
    parser.add_argument('-yl', default='Accuracy', help='Label for Y axis.')
    parser.add_argument('--nargs', nargs='+')
    args = parser.parse_args()

    filename = args.f.split('/')[-1].replace(".csv", "." + args.fmt)

    if args.nargs:
        files = args.nargs
        data = []
        for f in files:
            df = pd.read_csv(f)
            data.append(df)
        draw_multi_df(data, args.xl, args.yl, "plots/" + filename)

    else:
        result = pd.read_csv(args.f)

        error_file = args.f.split('.')[0] + '_error.csv'
        error = pd.read_csv(error_file) if os.path.isfile(error_file) else None

        if args.t == 0:
            draw_plot(result, args.xl, args.yl, "plots/" + filename)
        elif args.t == 1:
            draw_multi_column(result, int(len(result.columns)/2), args.xl, args.yl, "plots/" + filename, fmt=args.fmt)
            # draw_multi_column(result, 2, ['WL-AL', 'ALFNET'], args.xl, args.yl, "plots/" + filename)
        else:
            draw_multi_y_column(result, len(result.columns)-1, args.xl, args.yl, "plots/" + filename, fmt=args.fmt, error_df=error)
        


if __name__ == "__main__":
    args = main()