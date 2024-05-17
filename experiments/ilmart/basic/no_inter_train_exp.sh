# $1 represent the base folder
set -x #echo on

for dataset in web30k yahoo istella; do
  for strategy in greedy prev contrib; do
    python basic/train_no_inter.py $dataset $strategy $1/no-inter/$dataset/$strategy/ $(realpath ../fixed_config.json
  done
done
