#!/usr/bin/env python
# coding: utf-8
import argparse
import json

from tqdm import tqdm

from ilmart.utils import load_datasets
from pathlib import Path
from ilmart import Ilmart


def main():
    parser = argparse.ArgumentParser(description="Ilmart training with hyper-parameters fixed")
    parser.add_argument("dataset", type=str, help="Dataset to use. Available options: web30k, yahoo, istella.")
    parser.add_argument("strategy", type=str, help="Strategy to use for learning the main effects."
                                                   "Available options: greedy, prev, contrib.")
    parser.add_argument("out", type=str, help="Output folder containing the final model.")
    parser.add_argument("fixed_config",
                        type=str,
                        help="""
                            Path to the JSON file containing the fine-tuning configuration. It contains the 
                            following keys:
                            - common_params: common params to use during the LGBM training.
                            - boosting_rounds: the number of boosting rounds to do.
                            - vali_rounds: number of round for early exit.
                            - contrib_rounds: number fo rounds to do to compute the contribution of each feature.
                            - feat_imp: dict of list containing the feature importance for each dataset.
                            - list_n_main_effects: list of value for n_main_effects parameters of ilmart.
                             """)

    args = parser.parse_args()

    strategy_main = args.strategy
    dataset = args.dataset

    with open(args.fixed_config) as f:
        json_args = json.load(f)
        training_params = json_args["common_params"]
        boosting_rounds = json_args["boosting_rounds"]
        contrib_boosting_rounds = json_args["contrib_boosting_rounds"]
        feat_imp = json_args["feat_imp"][dataset]
        vali_rounds = json_args["vali_rounds"]
        list_n_main_effects = json_args["list_n_main_effects"]

    output_folder = Path(args.out).resolve()

    output_folder.mkdir(parents=True, exist_ok=True)

    rankeval_datasets = load_datasets([dataset], verbose=True)

    for name, datasets in rankeval_datasets.items():
        for n_main_effects in tqdm(list_n_main_effects, desc="n_main_effects"):
            n_main_effects = int(n_main_effects)
            print(f"Fitting ilmart with n_main_effects={n_main_effects}")
            best_model = Ilmart(training_params, verbose=False, main_strategy=strategy_main)
            best_model.fit_main_effects(boosting_rounds,
                                        datasets["train"],
                                        datasets["vali"],
                                        n_main_effects=n_main_effects,
                                        feat_imp=feat_imp,
                                        contrib_boosting_rounds=contrib_boosting_rounds,
                                        early_stopping_rounds=vali_rounds
                                        )

            best_model.get_model().save_model(output_folder / Path(f"{n_main_effects}.lgbm"))


if __name__ == '__main__':
    main()
