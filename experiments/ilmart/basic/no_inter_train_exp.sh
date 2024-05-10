for dataset in web30k yahoo istella; do
  for strategy in greedy prev contrib; do
    COMMAND="python basic/train_no_inter.py $dataset $strategy $1/no-inter/$dataset/$strategy/ $(realpath ../fixed_config.json)"
    echo $COMMAND
    $COMMAND
  done
done
