# $1 represent the base folder
set -x #echo on
declare -A MAIN_EFFCTS_DICT
MAIN_EFFCTS_DICT[web30k]=50
MAIN_EFFCTS_DICT[yahoo]=50
MAIN_EFFCTS_DICT[istella]=50
for dataset in web30k yahoo istella; do
  for strategy in greedy contrib prev; do
    python train_no_inter.py $dataset $strategy ${MAIN_EFFCTS_DICT[$dataset]} $1/$dataset/$strategy/no_inter.lgbm config_fine_tuning.json
  done
done