#!/bin/bash

# create directories
mkdir -p data       # -> stores all the datasets
mkdir -p logs       # -> stores experiment logs
mkdir -p cache      # -> stores cached results for each run
mkdir -p results    # -> stores all the generated results
mkdir -p plots      # -> stores all the generated plots


# Download and store .cites, .content files in data/citeseer, data/cora directories
# URL: https://linqs.soe.ucsc.edu/data
wget https://linqs-data.soe.ucsc.edu/public/lbc/citeseer.tgz && tar zxvf citeseer.tgz -C data/ && rm -rf citeseer.tgz
wget https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz && tar zxvf cora.tgz -C data/ && rm -rf cora.tgz
wget https://linqs-data.soe.ucsc.edu/public/Pubmed-Diabetes.tgz && tar zxvf Pubmed-Diabetes.tgz -C data/ && rm -rf Pubmed-Diabetes.tgz && mv data/Pubmed-Diabetes data/pubmed

# Process datasets
sh gen_csv.sh citeseer
sh gen_csv.sh cora
# follow notebooks/pubmed.ipynb directory to process pubmed dataset


## Hateful users
mkdir -p data/hateful

# Download the following file from kaggle: https://www.kaggle.com/manoelribeiro/hateful-users-on-twitter
# - users_neighborhood_anon.csv
# - users.edges

# then follow notebooks/hateful.ipynb directory to process pubmed dataset