# $1 represent the base folder
set -x #echo on
for dataset in web30k yahoo istella; do
  python train.py $dataset /data/ebm/inter $(realpath ft_config.json) --interactions=50
done
