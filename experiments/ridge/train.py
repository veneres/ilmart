#!/usr/bin/env python
# coding: utf-8
import logging
import pickle
import argparse
import optuna
import rankeval
from sklearn.linear_model import Ridge

from ilmart.utils import load_datasets
import json
from rankeval.metrics import NDCG
from pathlib import Path
from optuna.samplers import GridSampler

_logger = logging.getLogger(__name__)

optuna.logging.set_verbosity(optuna.logging.DEBUG)

MAX_JOBS_EBM = 60


def fun_to_optimize(trial: optuna.Trial,
                    optuna_min_max: dict,
                    train_dataset: rankeval.dataset.Dataset,
                    vali_dataset: rankeval.dataset.Dataset,
                    model_checkpoint_folder: Path,
                    ) -> float:
    params = {}

    params["alpha"] = trial.suggest_float('alpha',
                                          min(optuna_min_max["alpha"]),
                                          max(optuna_min_max["alpha"]))

    print(f"Params used: {params=}")

    regressor = Ridge(random_state=42, **params)

    X = train_dataset.X
    X[(X > 10 ** 10)] = 10 ** 10
    regressor.fit(X, train_dataset.y)

    model_checkpoint_folder.mkdir(parents=True, exist_ok=True)

    with open(model_checkpoint_folder / f"{trial.number}.pickle", "wb") as fout:
        pickle.dump(regressor, fout)

    predictions = regressor.predict(vali_dataset.X)

    new_ndcg = NDCG(cutoff=10, no_relevant_results=1, implementation="exp").eval(vali_dataset, predictions)[0]

    return new_ndcg


def main():
    parser = argparse.ArgumentParser(description="Ridge fine tuning",
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

    parser.add_argument("--n_jobs", type=int, help="Number of core to use", default=3)

    args = parser.parse_args()

    dataset = args.dataset

    n_jobs = args.n_jobs

    with open(args.ft_config) as f:
        json_args = json.load(f)
        optuna_min_max = json_args["optuna_min_max"]

    output_path = Path(args.out_folder).resolve()

    output_path = output_path / dataset

    Path(output_path).mkdir(parents=True, exist_ok=True)

    rankeval_datasets = load_datasets([dataset], verbose=True)

    model_checkpoint_folder = output_path / "study_checkpoints"

    for name, datasets in rankeval_datasets.items():
        config = {
            "optuna_min_max": optuna_min_max,
            "train_dataset": datasets["train"],
            "vali_dataset": datasets["vali"],
            "model_checkpoint_folder": model_checkpoint_folder
        }
        sampler = GridSampler(search_space=optuna_min_max)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(lambda trial: fun_to_optimize(trial, **config), n_trials=len(optuna_min_max["alpha"]),
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
