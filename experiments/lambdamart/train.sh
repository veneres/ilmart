# $1 represent the base folder

for dataset in web30k yahoo istella; do
    COMMAND="python train.py $dataset $1/$dataset.lgbm $(realpath ./fixed_config.json)"
    echo $COMMAND
    $COMMAND
done