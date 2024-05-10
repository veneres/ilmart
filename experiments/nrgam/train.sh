# $1 represent the base folder

for dataset in "web30k" "istella" "yahoo"; do
    COMMAND="python train.py $dataset --base_dir=$1"
    echo $COMMAND
    $COMMAND
done