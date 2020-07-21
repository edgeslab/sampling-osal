#!/bin/bash

# arg 1: number of trials, default: 5
# arg 2: budget, default 224

trials=5
budget=224


if [[ $# -gt 0 ]] ; then
    trials=$1
fi

if [[ $# -gt 1 ]] ; then
    budget=$2
fi

# generate datasets
sh gen_csv.sh citeseer
sh gen_csv.sh cora

# create directories
mkdir -p logs       # -> stores experiment logs
mkdir -p results    # -> stores all the generated results
mkdir -p plots      # -> stores all the generated plots
mkdir -p cache      # -> stores cached results for each run


echo "MLG-20 experiments starting for [trials: "$trials", budget: "$budget"]"

STARTTIME=$(date +%s)
MINUTES=60

# -------------------------------------- Results ---------------------------------------------

sh run_batch.sh "citeseer" "cc" $trials $budget
sh run_batch.sh "citeseer" "sgc" $trials $budget
sh run_batch.sh "citeseer" "wvrn" $trials $budget
sh run_batch.sh "citeseer" "gsage" $trials $budget


sh run_batch.sh "cora" "cc" $trials $budget
sh run_batch.sh "cora" "sgc" $trials $budget
sh run_batch.sh "cora" "wvrn" $trials $budget
sh run_batch.sh "cora" "gsage" $trials $budget


sh run_batch.sh "hateful" "cc" $trials $budget
sh run_batch.sh "hateful" "sgc" $trials $budget
sh run_batch.sh "hateful" "wvrn" $trials $budget
sh run_batch.sh "hateful" "gsage" $trials $budget


sh run_batch.sh "pubmed" "cc" $trials $budget
sh run_batch.sh "pubmed" "sgc" $trials $budget
sh run_batch.sh "pubmed" "wvrn" $trials $budget
sh run_batch.sh "pubmed" "gsage" $trials $budget


sleep 5

# generate Figure 1 from results
python bigplot.py -r results -f eps

# generate Table 2,3 from results
python table.py -r results -b 224 -c 5_224_32_8


# echo "zip all results"
# echo "==============="

zip -r results@mlg20.zip results plots cache > /dev/null


# -------------------------------------------------------------------------------

ENDTIME=$(date +%s)
elapsed=`echo "("$ENDTIME - $STARTTIME")/"$MINUTES | bc -l | xargs printf %.2f`
echo "MLG-20 experiments finished in ["$elapsed"] minutes!"