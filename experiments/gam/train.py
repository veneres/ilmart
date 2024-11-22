#!/usr/bin/env python
# coding: utf-8
import itertools
import logging
import pickle
import argparse

import optuna
import rankeval
from pygam import LinearGAM, s, te

from ilmart.utils import load_datasets
import json
from rankeval.metrics import NDCG
from pathlib import Path
from optuna.samplers import GridSampler

_logger = logging.getLogger(__name__)

optuna.logging.set_verbosity(optuna.logging.DEBUG)


def fun_to_optimize(trial: optuna.Trial,
                    optuna_min_max: dict,
                    train_dataset: rankeval.dataset.Dataset,
                    vali_dataset: rankeval.dataset.Dataset,
                    model_checkpoint_folder: Path,
                    n_main: int,
                    feat_imp: list[int],
                    n_inter: int,
                    ) -> float:
    params = {}

    params["lambda"] = trial.suggest_float('lambda',
                                           min(optuna_min_max["lambda"]),
                                           max(optuna_min_max["lambda"]))

    print(f"Params used: {params=}")

    X = train_dataset.X
    # Hack to solve problem with very large values
    X[(X > 10 ** 10)] = 10 ** 8

    # Create the main effects
    main_effects = [s(feat_idx, lam=params["lambda"]) for feat_idx in feat_imp[:n_main]]
    main_effects_sum = main_effects[0]
    for i in range(1, len(main_effects)):
        main_effects_sum += main_effects[i]

    effects_to_use = main_effects_sum

    # Create the interaction effects
    if n_inter > 0:

        # Same loop used in ilmart to find the most promising feature interaction given a precomputed feature importance.
        # Looping to find the right amount of main features to use. In this way we avoid problems of dealing with the
        # decimal part of computing n from (n*(n-1) )/ 2 = n_inter_effects
        comb_found = False
        comb_n_mif = None
        n = 2
        while not comb_found and n <= len(feat_imp):
            comb_n_mif = list(itertools.combinations(feat_imp[:n], 2))
            comb_found = len(comb_n_mif) > n_inter
            n += 1

        for pair in comb_n_mif:
            effects_to_use += te(pair[0], pair[1], lam=params["lambda"])

    print("Fitting model with effects:")
    print(effects_to_use)
    gam = LinearGAM(effects_to_use, verbose=True).fit(X, train_dataset.y)

    model_checkpoint_folder.mkdir(parents=True, exist_ok=True)

    with open(model_checkpoint_folder / f"{trial.number}.pickle", "wb") as fout:
        pickle.dump(gam, fout)

    predictions = gam.predict(vali_dataset.X)

    new_ndcg = NDCG(cutoff=10, no_relevant_results=1, implementation="exp").eval(vali_dataset, predictions)[0]

    return new_ndcg


def main():
    parser = argparse.ArgumentParser(description="GAM fine tuning",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("dataset", type=str, help="Dataset to use. Available options: web30k, yahoo, istella.")
    parser.add_argument("out_folder", type=str,
                        help="Output folder containing all the models for the same setup (inter, no_inter)")
    parser.add_argument("ft_config",
                        type=str,
                        help="""
                            Path to the JSON file containing the fine-tuning configuration. It contains the 
                            following key:
                            - optuna_min_max: the list of parameter for the grid search
                             """)

    parser.add_argument("--n_jobs", type=int, help="Number of core to use", default=1)
    parser.add_argument("--inter", action="store_true", help="Use feature interactions")

    args = parser.parse_args()

    dataset = args.dataset

    n_jobs = args.n_jobs

    inter_on = args.inter  # activate feature interactions

    with open(args.ft_config) as f:
        json_args = json.load(f)
        optuna_min_max = json_args["optuna_min_max"]
        n_main = json_args["n_main"]
        n_inter = json_args["n_inter"] if inter_on else 0
        feat_imp = json_args["feat_imp"]

    output_path = Path(args.out_folder).resolve()
    output_path = output_path.expanduser()
    output_path = output_path / dataset

    Path(output_path).mkdir(parents=True, exist_ok=True)

    rankeval_datasets = load_datasets([dataset], verbose=True)

    model_checkpoint_folder = output_path / "study_checkpoints"

    for name, datasets in rankeval_datasets.items():
        config = {
            "optuna_min_max": optuna_min_max,
            "train_dataset": datasets["train"],
            "vali_dataset": datasets["vali"],
            "n_main": n_main,
            "n_inter": n_inter,
            "feat_imp": feat_imp[name],
            "model_checkpoint_folder": model_checkpoint_folder
        }
        sampler = GridSampler(search_space=optuna_min_max)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(lambda trial: fun_to_optimize(trial, **config), n_trials=len(optuna_min_max["lambda"]),
                       n_jobs=n_jobs)

        print(f"{study.best_trial.params=}")
        print(f"{study.best_trial.number=}")

        # copy best model
        with open(model_checkpoint_folder / f"{study.best_trial.number}.pickle", "rb") as fin:
            best_model = pickle.load(fin)

        with open(output_path / f"{dataset}.pickle", "wb") as fout:
            pickle.dump(best_model, fout)


if __name__ == '__main__':
    main()
