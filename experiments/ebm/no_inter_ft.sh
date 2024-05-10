# $1 represent the base folder

for dataset in web30k yahoo istella; do
  COMMAND="python  train.py $dataset /data/ebm/no-inter $(realpath ft_config.json)"
  echo "$COMMAND"
  $COMMAND
done
