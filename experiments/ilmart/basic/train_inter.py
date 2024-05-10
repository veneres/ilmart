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
    parser.add_argument("strategy", type=str, help="Strategy to use for learning the interaction effects."
                                                   "Available options: greedy, prev, contrib.")
    parser.add_argument("main_effect_model",
                        type=str,
                        help="Path to the main effect model to use to start the training")
    parser.add_argument("fixed_config",
                        type=str,
                        help="""
                            Path to the JSON file containing the fine-tuning configuration. It contains the 
                            following keys:
                            - common_params: common params to use during the LGBM training.
                            - boosting_rounds: the number of boosting rounds to do.
                            - vali_rounds: number of round for early exit.
                            - list_n_inter_effects: list of value for n_inter_effects parameters of ilmart.
                             """)
    parser.add_argument("out", type=str, help="Output folder containing the final model.")

    args = parser.parse_args()

    strategy_inter = args.strategy
    dataset = args.dataset

    main_effect_model_file = args.main_effect_model

    with open(args.fixed_config) as f:
        json_args = json.load(f)
        training_params = json_args["common_params"]
        boosting_rounds = json_args["boosting_rounds"]
        contrib_boosting_rounds = json_args["contrib_boosting_rounds"]
        vali_rounds = json_args["vali_rounds"]
        list_n_inter_effects = json_args["list_n_inter_effects"]

    output_folder = Path(args.out).resolve()

    output_folder.mkdir(parents=True, exist_ok=True)

    rankeval_datasets = load_datasets([dataset], verbose=True)

    for name, datasets in rankeval_datasets.items():
        for n_inter_effects in tqdm(list_n_inter_effects):
            n_inter_effects = int(n_inter_effects)
            best_model = Ilmart(training_params,
                                main_effect_model_file=main_effect_model_file,
                                verbose=False,
                                inter_strategy=strategy_inter)
            best_model.fit_inter_effects(
                boosting_rounds,
                datasets["train"],
                datasets["vali"],
                num_interactions=n_inter_effects,
                early_stopping_rounds=vali_rounds,
                contrib_boosting_rounds=contrib_boosting_rounds
            )

            best_model.get_model().save_model(output_folder / Path(f"{n_inter_effects}.lgbm"))


if __name__ == '__main__':
    main()
