# awk '{print $1, $NF}' 'data/'$1'/'$1'.content' > 'data/'$1'/nodes.csv'
# awk '{$NF=""; print $0}' 'data/'$1'/'$1'.content' > 'data/'$1'/features.csv'
# cat 'data/'$1'/'$1'.cites' > 'data/'$1'/edges.csv'

#Comma Separated
awk -v OFS=',' '{print $1, $NF}' 'data/'$1'/'$1'.content' > 'data/'$1'/nodes.csv'
awk '{$NF="";print}' 'data/'$1'/'$1'.content' | awk -v OFS=',' '{$1=$1;print}' > 'data/'$1'/features.csv'
awk -v OFS=',' '{$1=$1;print $0}' 'data/'$1'/'$1'.cites' > 'data/'$1'/edges.csv'