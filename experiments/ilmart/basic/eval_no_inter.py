"""
Script to evaluate all the models created for the main effects.
#TODO make it more general and be able to use for more evaluation
"""
import argparse
from collections import namedtuple
from pathlib import Path
from rankeval.metrics import NDCG
from tqdm import tqdm
from ilmart.utils import load_datasets, get_effects, has_inter_with_more_than_n

import lightgbm as lgbm
import pandas as pd

NDCGEntry = namedtuple("NDCGEntry", ["dataset", "subset", "strategy", "budget", "n_feature_used", "cutoff", "ndcg"])

DATASETS = ["web30k", "yahoo", "istella"]
STRATEGIES = ["greedy", "prev", "contrib"]
DATASETS_SUBSETS = ["vali", "test"]


def main():
    parser = argparse.ArgumentParser(description="Batch evaluation script")
    parser.add_argument("base_folder", type=str, help="Base folder where to find all the models.")
    parser.add_argument("out", type=str, help="Output file for the final CSV.")

    args = parser.parse_args()

    base_folder = Path(args.base_folder)

    rankeval_datasets = load_datasets(verbose=True)

    entries = []
    for dataset in DATASETS:
        dataset_dir = base_folder / Path(dataset)
        for strategy in STRATEGIES:
            current_dir = dataset_dir / Path(strategy)
            for dataset_subset in DATASETS_SUBSETS:
                test = rankeval_datasets[dataset][dataset_subset]
                print(f"Currently in: {current_dir}, predicting {dataset_subset} dataset subset.")
                model_file_names = [file_name for file_name in current_dir.iterdir() if file_name.is_file() and ".lgbm" in file_name.name]
                for model_file_name in tqdm(model_file_names):
                    budget = int(model_file_name.name.replace(".lgbm", ""))
                    model = lgbm.Booster(model_file=model_file_name.resolve())
                    if has_inter_with_more_than_n(model, 1):
                        raise (f"Huston, we have a problem with {dataset}, {budget=}!")

                    n_me = len(get_effects(model, 1))
                    predictions = model.predict(test.X)
                    for cutoff in [1, 5, 10]:
                        ndcg = NDCG(cutoff=cutoff, no_relevant_results=1, implementation="exp")
                        res = ndcg.eval(test, predictions)
                        entries.append(
                            NDCGEntry(dataset=dataset, budget=budget, ndcg=res[0], cutoff=cutoff,
                                      strategy=strategy, subset=dataset_subset, n_feature_used=n_me))

    df = pd.DataFrame(entries)
    df.to_csv(args.out)


if __name__ == '__main__':
    main()
