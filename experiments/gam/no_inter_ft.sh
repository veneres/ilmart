# $1 represent the base folder
set -x #echo on
for dataset in web30k yahoo istella; do
  python train.py $dataset ~/data/gam/no-inter $(realpath ft_config.json)
done