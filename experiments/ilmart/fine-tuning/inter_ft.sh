# $1 represent the base folder
set -x #echo on

N_INTER_EFFECTS=50
for dataset in web30k yahoo istella; do
  for strategy_main in greedy prev contrib; do
    #for strategy_inter in greedy prev contrib; do
    for strategy_inter in prev; do
      python  train_inter.py $dataset $strategy_inter $1/$dataset/$strategy_main/no_inter.lgbm $N_INTER_EFFECTS  $1/$dataset/$strategy_main/inter_$strategy_inter.lgbm config_fine_tuning.json
    done
  done
done