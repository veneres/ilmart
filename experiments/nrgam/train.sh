# $1 represent the base folder
set -x #echo on

for dataset in "web30k" "istella" "yahoo"; do
    python train.py $dataset --base_dir=$1
done