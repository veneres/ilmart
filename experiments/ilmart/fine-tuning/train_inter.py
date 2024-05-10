#!/usr/bin/env python
# coding: utf-8
import copy
import typing
import argparse
import rankeval.dataset
import json
from ilmart.utils import load_datasets, is_interpretable
from pathlib import Path
from ilmart import Ilmart
from rankeval.metrics import NDCG
import optuna


def fun_to_optimize(trial: optuna.Trial,
                    fixed_params: dict,
                    hyper_opt_min_max: dict,
                    boosting_rounds: int,
                    contrib_boosting_rounds: int,
                    train_dataset: rankeval.dataset.Dataset,
                    vali_dataset: rankeval.dataset.Dataset,
                    inter_effect_strategy: str,
                    n_inter_effects: typing.Optional[int],
                    main_effect_model_file: str):
    params = copy.deepcopy(fixed_params)

    params["learning_rate"] = trial.suggest_float('learning_rate',
                                                  hyper_opt_min_max["learning_rate"]["min"],
                                                  hyper_opt_min_max["learning_rate"]["max"],
                                                  log=True)
    params["num_leaves"] = trial.suggest_int('num_leaves',
                                             hyper_opt_min_max["num_leaves"]["min"],
                                             hyper_opt_min_max["num_leaves"]["max"])

    model = Ilmart(params,
                   verbose=False,
                   main_effect_model_file=main_effect_model_file,
                   inter_strategy=inter_effect_strategy)

    model.fit_inter_effects(boosting_rounds,
                            train_dataset,
                            vali_dataset,
                            num_interactions=n_inter_effects,
                            contrib_boosting_rounds=contrib_boosting_rounds)

    predictions = model.get_model().predict(vali_dataset.X)

    new_ndcg = NDCG(cutoff=10, no_relevant_results=1, implementation="exp").eval(vali_dataset, predictions)[0]

    return new_ndcg


def main():
    parser = argparse.ArgumentParser(description="Ilmart fine tuning with interactions")
    parser.add_argument("dataset", type=str, help="Dataset to use. Available options: web30k, yahoo, istella.")
    parser.add_argument("strategy", type=str, help="Strategy to use for learning the interaction effects."
                                                   "Available options: greedy, prev, contrib.")
    parser.add_argument("base_model_path", type=str,
                        help="Path to the base model used to learn the interaction effects")

    parser.add_argument("n", type=int, help="Maximum number of main effects to use.")

    parser.add_argument("out", type=str, help="Output file containing the final model.")
    parser.add_argument("ft_config",
                        type=str,
                        help="""
                            Path to the JSON file containing the fine-tuning configuration. It contains the 
                            following keys:
                            - common_params: common params to use during the LGBM training.
                            - hyper_opt_min_max: the parameter grid to pass to the .
                            - boosting_rounds: the number of boosting rounds to do.
                            - n_trials: the number of trials to fine-tune the model
                             """)

    args = parser.parse_args()

    strategy_inter = args.strategy
    dataset = args.dataset

    base_model_path = args.base_model_path

    with open(args.ft_config) as f:
        json_args = json.load(f)
        n_trials = json_args["n_trials"]
        common_params = json_args["common_params"]
        boosting_rounds = json_args["boosting_rounds"]
        contrib_boosting_rounds = json_args["contrib_boosting_rounds"]
        hyper_opt_min_max = json_args["hyper_opt_min_max"]

    output_path = Path(args.out).resolve()
    n_inter_effects = args.n

    datasets = load_datasets([dataset], verbose=True)[dataset]

    config = {
        "hyper_opt_min_max": hyper_opt_min_max,
        "fixed_params": common_params,
        "boosting_rounds": boosting_rounds,
        "train_dataset": datasets["train"],
        "vali_dataset": datasets["vali"],
        "contrib_boosting_rounds": contrib_boosting_rounds,
        "inter_effect_strategy": strategy_inter,
        "n_inter_effects": n_inter_effects,
        "main_effect_model_file": base_model_path
    }
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: fun_to_optimize(trial, **config), n_trials=n_trials)

    best_params = dict(**study.best_trial.params, **common_params)

    best_model = Ilmart(best_params,
                        verbose=False,
                        main_effect_model_file=base_model_path,
                        inter_strategy=strategy_inter)

    best_model.fit_inter_effects(boosting_rounds,
                                 datasets["train"],
                                 datasets["vali"],
                                 num_interactions=n_inter_effects,
                                 contrib_boosting_rounds=contrib_boosting_rounds)

    best_model.get_model().save_model(output_path)


if __name__ == '__main__':
    main()
