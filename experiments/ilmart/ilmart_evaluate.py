#!/usr/bin/env python
# coding: utf-8
from pathlib import Path

import lightgbm as lgbm
import argparse
import json
import pickle
from collections import defaultdict
from ilmart.utils import load_datasets, is_interpretable
from rankeval.metrics import NDCG


def evaluate(models_dir, rankeval_datasets, path_out):
    boosters_dict = {}
    for name in rankeval_datasets.keys():
        file_path = f"{models_dir}/{name}.lgbm"
        boosters_dict[name] = lgbm.Booster(model_file=file_path)

    ndcgs_ilmart = defaultdict(dict)
    for name, model in boosters_dict.items():
        print(f"Computing NCDG for {name}")
        test_dataset = rankeval_datasets[name]["test"]
        predictions = model.predict(test_dataset.X)
        for cutoff in [1, 5, 10]:
            ndcg = NDCG(cutoff=cutoff, no_relevant_results=1, implementation="exp")
            res = ndcg.eval(test_dataset, predictions)
            ndcgs_ilmart[name][cutoff] = res[1]
            print(f"\tCutoff {cutoff} (mean): {res[0]}")
        print(f"Is interpretable? {is_interpretable(model)}")

    with open(path_out, "wb") as f:
        pickle.dump(ndcgs_ilmart, f)


def main():
    parser = argparse.ArgumentParser(description="Ilmart benchmark")
    parser.add_argument("--config_path",
                        type=str,
                        default="config.json",
                        help="""
                Path to the JSON file containing the configuration for the benchmark. It contains the following keys:
                - path_out: path where to save the models.
                - path_eval: path where to save the evaluation results of the model
                """)

    args = parser.parse_args()

    with open(args.config_path) as f:
        try:
            json_args = json.load(f)
            path_out = json_args["path_out"]
            path_eval = json_args["path_eval"]
        except Exception as e:
            print(f"Problems reading the configuration file {args.config_path} ")
            print(e)
    rankeval_datasets = load_datasets()

    Path(path_eval).mkdir(parents=True, exist_ok=True)

    evaluate(f"{path_out}/without_inter", rankeval_datasets, f"{path_eval}/ilmart.pickle")

    evaluate(f"{path_out}/with_inter", rankeval_datasets, f"{path_eval}/ilmart_i.pickle")


if __name__ == '__main__':
    main()
