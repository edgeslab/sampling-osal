import os
import pdb
import argparse

import numpy as np
import pandas as pd

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



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', default='results/', help='Results directory', required=True)
    parser.add_argument('-b', type=int, default=-1, help='Target budget', required=False)
    parser.add_argument('-c', default='5_224_32_8', help='Exp config', required=False)
    args = parser.parse_args()

    df = None

    i = 0
    for ds in DATASETS:
        for clf in CLASSIFIERS:
            result_file = os.path.join(args.r, '%s_%s_%s.csv' % (clf, ds, args.c))
            rdf = pd.read_csv(result_file)

            if df is None:
                algos = ['_'.join(col.split('_')[1:]) for col in list(rdf.columns)[1:]]
                algos = [ymap[algo] for algo in algos]
                # algos = [algo.upper().replace('_', '-') for algo in algos]
                df = pd.DataFrame(columns = ['Dataset', 'Classifier'] + algos)

            if clf == 'cc':
                clf = 'ICA'

            df.loc[i] = [ds, clf] + list(rdf[rdf['Budget'] == args.b].values[0][1:])

            i += 1

    df['Dataset'] = df['Dataset'].apply(lambda x: x.capitalize())
    df['Classifier'] = df['Classifier'].apply(lambda x: x.upper())
    
    print(df.to_latex(index=False, column_format=''.join(['l'] * len(df.columns)), label='table:f1', caption="Micro-F1 scores of all sampling methods across all datasets and classifiers for budget 224."))




    ranking = df[df.columns.drop(['Dataset', 'Classifier'])]
    ranking = ranking.transpose()
    ranking = ranking.rank(ascending=False)
    ranking['Rank Mean'] = [np.around(ranking.loc[ind].mean(), 2) for ind in list(ranking.index)]
    ranking['Rank Std'] = [np.around(ranking.loc[ind].std(), 2) for ind in list(ranking.index)]
    ranking['Final Rank'] = ranking['Rank Mean'].rank().astype(int)
    ranking = ranking[['Rank Mean', 'Final Rank', 'Rank Std']]
    ranking = ranking.reset_index()
    ranking = ranking.rename(columns={'index' : 'Sampling'})
    ranking = ranking.sort_values(by=['Final Rank'])
    ranking = ranking[['Sampling', 'Rank Mean', 'Rank Std']]
    ranking['Avg. Rank'] = ranking[['Rank Mean', 'Rank Std']].apply(lambda x: "$%0.2f \pm %0.2f$" % (x['Rank Mean'], x['Rank Std']), axis=1)
    ranking = ranking[['Sampling', 'Avg. Rank']]
    print(ranking.to_latex(index=False, escape=False, label='table:rank',caption="Rankings of sampling methods based on Micro-F1 scores."))

    # pdb.set_trace()



            




if __name__ == "__main__":
    args = main()