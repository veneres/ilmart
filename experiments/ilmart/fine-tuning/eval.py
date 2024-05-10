#!/usr/bin/env python
# coding: utf-8
from __future__ import annotations

import argparse
from collections import namedtuple
from pathlib import Path

import rankeval.dataset

from ilmart.utils import load_datasets, eval_dataset, is_interpretable, get_effects
import pandas as pd
import lightgbm as lgbm

STRATEGIES = ["greedy",
              "prev",
              "contrib"]

NO_INTER_MODEL_NAME = "no_inter.lgbm"

INTER_MODEL_NAMES = {
    "greedy": "inter_greedy.lgbm",
    "prev": "inter_prev.lgbm",
    "contrib": "inter_contrib.lgbm",

}

CSVLine = namedtuple("CSVLine", ["dataset",
                                 "main_strategy",
                                 "inter_strategy",
                                 "cutoff",
                                 "ndcg",
                                 "n_main_effects",
                                 "n_inter_effects",
                                 "query_id"])


def create_csv_entries(model_path: Path,
                       test_dataset: rankeval.dataset.Dataset,
                       dataset_name: str,
                       main_strategy_name: str,
                       inter_strategy_name: str | None) -> [namedtuple]:
    model_path = model_path.resolve()
    model = lgbm.Booster(model_file=model_path)
    if not is_interpretable(model):
        raise Exception(f"Problem with {model_path}")
    n_main_effects = len(get_effects(model, 1))
    n_inter_effects = len(get_effects(model, 2))

    comp_ndcg = eval_dataset(model_path, test_dataset)

    res = []
    for cutoff, (ndcg_per_query) in comp_ndcg.items():
        for q_idx, value in ndcg_per_query.items():
            res.append(CSVLine(dataset=dataset_name,
                               main_strategy=main_strategy_name,
                               inter_strategy=inter_strategy_name,
                               cutoff=cutoff,
                               query_id=q_idx,
                               ndcg=value,
                               n_main_effects=n_main_effects,
                               n_inter_effects=n_inter_effects))
    return res


def main():
    parser = argparse.ArgumentParser(description="Evaluation of all the fine tuned models")
    parser.add_argument("--base_folder", type=str,
                        help="base folder containing all fine-tuned ilmart models",
                        default="/data/ft")

    parser.add_argument("--out",
                        type=str,
                        help="Output file containing a csv with the results",
                        default="/data/ft/eval.csv")

    args = parser.parse_args()

    base_folder = args.base_folder
    out = args.out

    rankeval_datasets = load_datasets(verbose=True)

    res = []
    for dataset_name, datasets in rankeval_datasets.items():
        for strategy in STRATEGIES:

            main_strategy_folder = Path(base_folder) / Path(dataset_name) / Path(strategy)
            no_inter_path = main_strategy_folder / Path(NO_INTER_MODEL_NAME)
            inter_file_names = {
                inter_strategy: main_strategy_folder / filename
                for inter_strategy, filename in INTER_MODEL_NAMES.items()
            }
            res += create_csv_entries(model_path=no_inter_path,
                                      test_dataset=datasets["test"],
                                      dataset_name=dataset_name,
                                      main_strategy_name=strategy,
                                      inter_strategy_name=None)

            for inter_strategy, inter_file_path in inter_file_names.items():
                res += create_csv_entries(model_path=inter_file_path.resolve(),
                                          test_dataset=datasets["test"],
                                          dataset_name=dataset_name,
                                          main_strategy_name=strategy,
                                          inter_strategy_name=inter_strategy)
    pd.DataFrame(res).to_csv(out, index=False)


if __name__ == '__main__':
    main()
