FIXED_MAIN_EFFECTS=50

for dataset in web30k yahoo istella; do
  for main_strategy in greedy prev contrib; do
    for inter_strategy in greedy prev contrib; do
      COMMAND="python train_inter.py $dataset $inter_strategy $1/no-inter/$dataset/$main_strategy/$FIXED_MAIN_EFFECTS.lgbm $(realpath ../fixed_config.json) $1/inter/$dataset/$main_strategy/$inter_strategy"
      echo $COMMAND
      $COMMAND
      done
  done
done
