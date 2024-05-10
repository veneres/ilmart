# $1 represent the base folder

for dataset in web30k yahoo istella; do
  COMMAND="python train.py $dataset /data/ebm/inter $(realpath ft_config.json) --interactions=50"
  echo "$COMMAND"
  $COMMAND
done
