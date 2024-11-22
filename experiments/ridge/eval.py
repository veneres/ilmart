#!/usr/bin/env python
# coding: utf-8
from __future__ import annotations

import argparse
import pickle
from collections import namedtuple
from pathlib import Path

import rankeval.dataset
from rankeval.metrics import NDCG

from ilmart.utils import load_datasets
import pandas as pd

STRATEGIES = ["greedy",
              "prev",
              "contrib"]

CSVLine = namedtuple("CSVLine", ["dataset",
                                 "cutoff",
                                 "ndcg",
                                 "query_id"])
CUTOFFS = [1, 5, 10]


def create_csv_entries(model_path: Path,
                       test_dataset: rankeval.dataset.Dataset,
                       dataset_name: str) -> [namedtuple]:
    model_path = model_path.resolve()
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    res = []
    X = test_dataset.X
    X[(X > 10 ** 10)] = 10 ** 10

    predicted = model.predict(X)

    for cutoff in [1, 5, 10]:
        ndcg = NDCG(cutoff=cutoff, no_relevant_results=1, implementation="exp")
        for qid, q_y, q_y_pred in ndcg.query_iterator(test_dataset, predicted):
            ndcg_value = ndcg.eval_per_query(q_y, q_y_pred)
            res.append(CSVLine(dataset=dataset_name,
                               cutoff=cutoff,
                               query_id=qid,
                               ndcg=ndcg_value))
    return res


def main():
    parser = argparse.ArgumentParser(description="Evaluation of EBM")
    parser.add_argument("--base_folder",
                        type=str,
                        help="Base folder containing the models to evaluate",
                        default="/data/ridge/")

    parser.add_argument("--out",
                        type=str,
                        help="Output file containing a csv with the results",
                        default="/data/ridge/eval.csv")

    args = parser.parse_args()

    base_folder = args.base_folder
    out = args.out

    Path(out).parent.mkdir(parents=True, exist_ok=True)

    rankeval_datasets = load_datasets(verbose=True)

    res = []
    for dataset_name, datasets in rankeval_datasets.items():
        dataset_model_path = Path(f"{base_folder}/{dataset_name}/{dataset_name}.pickle")
        if not dataset_model_path.exists():
            print(f"Model for {dataset_name} not found in {dataset_model_path}")
            continue
        res += create_csv_entries(model_path=dataset_model_path,
                                  test_dataset=datasets["test"],
                                  dataset_name=dataset_name)
    pd.DataFrame(res).to_csv(out, index=False)


if __name__ == '__main__':
    main()
