#!/usr/bin/env python
# coding: utf-8
import typing
import argparse
import rankeval.dataset
import json
from ilmart.utils import load_datasets
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import ParameterGrid
from ilmart import Ilmart
from rankeval.metrics import NDCG


def hyperparams_grid_search(param_grid: dict,
                            common_params: dict,
                            train_dataset: rankeval.dataset.Dataset,
                            vali_dataset: rankeval.dataset.Dataset,
                            boosting_rounds: int,
                            initial_ilmart: typing.Optional[Ilmart] = None,
                            n_interactions: typing.Optional[int] = None):
    max_ndcg = 0
    best_model = None
    for params in tqdm(list(ParameterGrid(param_grid))):
        current_params = {**common_params, **params}
        if initial_ilmart is not None:
            model = initial_ilmart
            model.feat_inter_boosting_rounds = boosting_rounds
            model.fit_inter_effects(current_params,
                                    boosting_rounds,
                                    train_dataset,
                                    vali_dataset,
                                    n_interactions)
        else:
            model = Ilmart(verbose=False, inter_rank_strategy="greedy")
            model.fit_main_effects(current_params, boosting_rounds, train_dataset, vali_dataset)

        predictions = model.get_model().predict(vali_dataset.X)
        new_ndcg = NDCG(cutoff=10, no_relevant_results=1, implementation="exp").eval(vali_dataset, predictions)[0]

        if new_ndcg > max_ndcg:
            best_model = model
            max_ndcg = new_ndcg
            print(f"New max ndcg found: {max_ndcg}")
            print(f"With params: {params}")
    return best_model


def save_models(models_dir: str, models_dict: dict):
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    for name, ilmart_model in models_dict.items():
        file_name = f"{models_dir}/{name}.lgbm"
        print(f"Saving to {file_name}")
        ilmart_model.get_model().save_model(file_name)


def main():
    parser = argparse.ArgumentParser(description="Ilmart benchmark")
    parser.add_argument("--config_path",
                        type=str,
                        default="config.json",
                        help="""
                            Path to the JSON file containing the configuration for the benchmark. It contains the following keys:
                            - path_out: path where to save the models.
                            - common_params: common params to use during the LGBM training.
                            - param_grid: the parameter grid to pass to ParameterGrid (sklearn).
                            - boosting_rounds: the number of boosting rounds to do.
                            - n_interactions: the number of interactions to add to the model.
                             """)

    args = parser.parse_args()

    with open(args.config_path) as f:
        try:
            json_args = json.load(f)
            path_out = json_args["path_out"]
            common_params = json_args["common_params"]
            param_grid = json_args["param_grid"]
            boosting_rounds = json_args["boosting_rounds"]
            n_interactions = json_args["n_interactions"]
        except Exception as e:
            print(f"Problems reading the configuration file {args.config_path} ")
            print(e)

    rankeval_datasets = load_datasets()

    print("Start computing best models with interactions")
    best_no_inter_ilmart = {}
    for name, datasets in rankeval_datasets.items():
        best_no_inter_ilmart[name] = hyperparams_grid_search(param_grid,
                                                             common_params,
                                                             datasets["train"],
                                                             datasets["vali"],
                                                             boosting_rounds)
    save_models(f"{path_out}/without_inter", best_no_inter_ilmart)

    print("Start computing best models with interactions")
    best_inter_ilmart = {}
    for name, ilmart_model in best_no_inter_ilmart.items():
        print(f"Find best model for {name}")
        train_ds = rankeval_datasets[name]["train"]
        vali_ds = rankeval_datasets[name]["vali"]
        best_inter_ilmart[name] = hyperparams_grid_search(param_grid, common_params, train_ds, vali_ds, boosting_rounds,
                                                          ilmart_model, n_interactions)
    save_models(f"{path_out}/with_inter", best_inter_ilmart)


if __name__ == '__main__':
    main()
