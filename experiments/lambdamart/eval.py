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
                                 "cutoff",
                                 "ndcg",
                                 "query_id"])


def create_csv_entries(model_path: Path,
                       test_dataset: rankeval.dataset.Dataset,
                       dataset_name: str) -> [namedtuple]:
    model_path = model_path.resolve()
    comp_ndcg = eval_dataset(model_path, test_dataset)

    res = []
    for cutoff, ndcg_per_query in comp_ndcg.items():
        for query_id, value in ndcg_per_query.items():
            res.append(CSVLine(dataset=dataset_name,
                               cutoff=cutoff,
                               query_id=query_id,
                               ndcg=value))
    return res


def main():
    parser = argparse.ArgumentParser(description="Evaluation of the full lambdaMART")
    parser.add_argument("--base_folder",
                        type=str,
                        help="Base folder containing the models to evaluate",
                        default="/data/lambdamart")

    parser.add_argument("--out",
                        type=str,
                        help="Output file containing a csv with the results",
                        default="/data/lambdamart/eval.csv")

    args = parser.parse_args()

    base_folder = args.base_folder
    out = args.out

    Path(out).parent.mkdir(parents=True, exist_ok=True)

    rankeval_datasets = load_datasets(verbose=True)

    res = []
    for dataset_name, datasets in rankeval_datasets.items():

        dataset_model_path = Path(base_folder) / f"{dataset_name}.lgbm"
        res += create_csv_entries(model_path=dataset_model_path,
                                  test_dataset=datasets["test"],
                                  dataset_name=dataset_name)
    pd.DataFrame(res).to_csv(out, index=False)


if __name__ == '__main__':
    main()
