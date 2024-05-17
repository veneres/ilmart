"""
Script to evaluate all the models created for the interaction effects.
#TODO make it more general and be able to use for more evaluation
"""
import argparse
from collections import namedtuple
from pathlib import Path
from rankeval.metrics import NDCG
from tqdm import tqdm
from ilmart.utils import load_datasets, is_interpretable, get_effects

import lightgbm as lgbm
import pandas as pd

NDCGEntry = namedtuple("NDCGEntry",
                       ["dataset", "budget", "n_feature_used", "ndcg", "cutoff", "main_strategy", "inter_strategy",
                        "subset"])

STRATEGIES = ["greedy", "prev", "contrib"]

DATASETS = ["web30k", "yahoo", "istella"]

DATASET_SUBSETS = ["vali"]


def compute_for_dir(dataset_name: str,
                    dataset_dir: dict,
                    model_dir: Path,
                    main_strategy: str,
                    inter_strategy: str) -> list[NDCGEntry]:
    entries = []
    file_names = [file_name for file_name in model_dir.iterdir() if file_name.is_file() and ".lgbm" in file_name.name]

    for model_file_name in tqdm(file_names):
        budget = int(model_file_name.name.replace(".lgbm", ""))
        model = lgbm.Booster(model_file=model_file_name.resolve())
        if not is_interpretable(model):
            raise Exception(f"Huston, we have a problem with {model_file_name.resolve()}, {budget=}!")
        n_inter_effects = len(get_effects(model, 2))
        for dataset_subset in DATASET_SUBSETS:
            test = dataset_dir[dataset_subset]
            predictions = model.predict(test.X)
            for cutoff in [1, 5, 10]:
                ndcg = NDCG(cutoff=cutoff, no_relevant_results=1, implementation="exp")
                mean_ndcg, detailed_scores = ndcg.eval(test, predictions)
                entries.append(NDCGEntry(dataset=dataset_name,
                                         budget=budget,
                                         n_feature_used=n_inter_effects,
                                         ndcg=mean_ndcg,
                                         cutoff=cutoff,
                                         main_strategy=main_strategy,
                                         inter_strategy=inter_strategy,
                                         subset=dataset_subset))
    return entries


def main():
    parser = argparse.ArgumentParser(description="Batch evaluation script for inter effect")
    parser.add_argument("base_folder", type=str, help="Base folder where to find all the models.")
    parser.add_argument("out", type=str, help="Output file for the final CSV.")

    args = parser.parse_args()

    base_folder = Path(args.base_folder)

    rankeval_datasets = load_datasets(verbose=True)

    entries = []
    for dataset_name in DATASETS:
        dataset_dir = base_folder / Path(dataset_name)
        current_dataset = rankeval_datasets[dataset_name]
        for main_strategy in STRATEGIES:
            main_strategy_dir = dataset_dir / Path(main_strategy)
            for inter_strategy in STRATEGIES:
                inter_dir = main_strategy_dir / Path(inter_strategy)
                entries += compute_for_dir(dataset_name,
                                           current_dataset,
                                           inter_dir,
                                           main_strategy,
                                           inter_strategy)
    df = pd.DataFrame(entries)
    df.to_csv(args.out)


if __name__ == '__main__':
    main()
