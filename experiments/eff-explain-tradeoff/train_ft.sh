# $1 represent the base folder
set -x #echo on

for max_tree_interactions in 2 3 4 5 6; do
  for max_interactions in 10 100 500 1000; do
      id="$max_tree_interactions"_"$max_interactions"
      python ft_constrained.py web30k $1/$id.lgbm $(realpath ./fixed_config.json) $max_tree_interactions $max_interactions
    done
done

