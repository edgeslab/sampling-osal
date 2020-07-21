#!/bin/bash

ds=$1
cls=$2
trials=$3
budget=$4

STARTTIME=$(date +%s)
MINUTES=60

# parallelly run individual sampling methods and cache results
python experiment.py -config configs/$2_$1.json -nt $trials -b $budget -algos rs kms &
python experiment.py -config configs/$2_$1.json -nt $trials -b $budget -algos ns_dc_h ns_ct_h &
python experiment.py -config configs/$2_$1.json -nt $trials -b $budget -algos es_rs fp &
python experiment.py -config configs/$2_$1.json -nt $trials -b $budget -algos wls_2 wls_3 &
python experiment.py -config configs/$2_$1.json -nt $trials -b $budget -algos ss ffs &
python experiment.py -config configs/$2_$1.json -nt $trials -b $budget -algos ms &
wait

# generate final results using the cached data
python experiment.py -config configs/$2_$1.json -nt $trials -b $budget --cached

# generate plots from the final results
python pplot.py -fmt eps -f "results/"$cls"_"$ds"_"$trials"_"$budget"_32_8.csv"         -t 2 -xl "Budget (#Nodes)" -yl "Micro-F1"
python pplot.py -fmt eps -f "results/"$cls"_"$ds"_"$trials"_"$budget"_32_8_timing.csv"  -t 2 -xl "Budget (#Nodes)" -yl "Execution Time (s)"

ENDTIME=$(date +%s)
elapsed=`echo "("$ENDTIME - $STARTTIME")/"$MINUTES | bc -l | xargs printf %.2f`

echo "finished '$ds'-'$cls' exp on WORDS in ['$elapsed'] minutes"