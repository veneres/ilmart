# $1 represent the base folder
set -x #echo on
for dataset in web30k yahoo istella; do
    python train.py $dataset $1/$dataset.lgbm $(realpath ./fixed_config.json)
done