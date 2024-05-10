"""
Script to evaluate all the models created for the main effects.
#TODO make it more general and be able to use for more evaluation
"""
import argparse
from collections import namedtuple
from pathlib import Path
from rankeval.metrics import NDCG
from tqdm import tqdm
from ilmart.utils import load_datasets, has_inter_with_more_than_n, has_more_than_n_inter, eval_dataset

import lightgbm as lgbm
import pandas as pd

NDCGEntry = namedtuple("NDCGEntry",
                       ["dataset", "max_interactions", "max_tree_interactions", "cutoff", "ndcg", "query_id"])

DATASET = "web30k"


def main():
    parser = argparse.ArgumentParser(description="Batch evaluation script")
    parser.add_argument("base_folder", type=str, help="Base folder where to find all the models.")
    parser.add_argument("out", type=str, help="Output file for the final CSV.")

    rankeval_datasets = load_datasets([DATASET], verbose=True)[DATASET]

    test = rankeval_datasets["test"]

    args = parser.parse_args()

    output = Path(args.out)

    entries = []

    models_dir = Path(args.base_folder)

    model_file_names = [model_file_name for model_file_name in tqdm(models_dir.iterdir())
                        if model_file_name.is_file() and ".lgbm" in model_file_name.name]

    for model_file_name in tqdm(model_file_names):
        max_tree_interactions, max_interactions = model_file_name.name.replace(".lgbm", "").split("_")
        max_tree_interactions = int(max_tree_interactions)
        max_interactions = int(max_interactions)
        model = lgbm.Booster(model_file=model_file_name.resolve())
        if has_inter_with_more_than_n(model, max_tree_interactions):
            raise Exception(f"Problems with {model_file_name.resolve()}! - has_effects_with_more_than_n_inter")
        if has_more_than_n_inter(model, max_interactions, max_tree_interactions):
            raise Exception(f"Problem with {model_file_name.resolve()}! - has more than n_inter")

        ndcgs_eval = eval_dataset(model_file_name.resolve(), test)

        for cutoff, ndcg_per_query in ndcgs_eval.items():
            for query_id, value in ndcg_per_query.items():
                entries.append(NDCGEntry(dataset=DATASET,
                                         cutoff=cutoff,
                                         query_id=query_id,
                                         ndcg=value,
                                         max_interactions=max_interactions,
                                         max_tree_interactions=max_tree_interactions
                                         ))

    df = pd.DataFrame(entries)
    df.to_csv(output)


if __name__ == '__main__':
    main()
