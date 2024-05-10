# $1 represent the base folder
for max_tree_interactions in 2 3 4 5 6; do
  for max_interactions in 10 100 500 1000; do
      id="$max_tree_interactions"_"$max_interactions"
      COMMAND="python ft_constrained.py web30k $1/$id.lgbm $(realpath ./fixed_config.json) $max_tree_interactions $max_interactions"
      echo "$COMMAND"
      $COMMAND
    done
done

