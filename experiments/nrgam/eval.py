#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_ranking as tfr
import argparse
from tqdm import tqdm
from rankeval.metrics.ndcg import NDCG
from collections import defaultdict, namedtuple
import yahoo_dataset
import web30k_dataset
import istella_dataset
import numpy as np
from utils import ds_transform, LOG_NORMALIZATION

DATASETS = ["web30k", "istella", "yahoo"]

CSVLine = namedtuple("CSVLine", ["dataset", "cutoff", "ndcg", "query_id"])


def compute_ndcg_results(batch_results, ds_test_y, cutoffs):
    ndcg_results = defaultdict(list)
    for batch_id, batch_y_true in tqdm(enumerate(ds_test_y)):
        for query_in_batch, y_true_padded in enumerate(batch_y_true):
            start_padding_index = np.argmax(y_true_padded == -1)
            y_true = y_true_padded[:start_padding_index].numpy()
            y_pred = batch_results[batch_id][query_in_batch][:start_padding_index]
            for cutoff in cutoffs:
                ndcg = NDCG(cutoff=cutoff, no_relevant_results=1, implementation="exp")
                ndcg_results[cutoff].append(ndcg.eval_per_query(y_true, y_pred))
    return ndcg_results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the accuracy of Neural Rank GAM for the three dataset: istella, web30k, yahoo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--base_dir", default="/data/nrgam", type=str,
                        help="Base path where the models are saved")
    parser.add_argument("--output_file", default="/data/nrgam/eval.csv", type=str,
                        help="Output file where the results will be saved")

    args = parser.parse_args()
    base_path = args.base_dir
    model_paths = {dataset_name: f"{base_path}/{dataset_name}_model" for dataset_name in DATASETS}


    test_datasets = {}
    for dataset_name in tqdm(DATASETS, desc="Loading datasets"):
        test_datasets[dataset_name] = ds_transform(tfds.load(f"{dataset_name}_rankeval", split="test"),
                                                   log=LOG_NORMALIZATION[dataset_name],
                                                   train=False)


    best_tf_models = {}
    for dataset_name, path in tqdm(model_paths.items(), desc="Loading models"):
        best_tf_models[dataset_name] = tf.keras.models.load_model(path)

    cutoffs = [1, 5, 10]
    res = []
    for dataset_name, model in tqdm(best_tf_models.items(), desc="Evaluating models"):
        ds_test_y = test_datasets[dataset_name].map(lambda feature_map, query_id, doc_id, label: label)
        ds_test_X = test_datasets[dataset_name].map(lambda feature_map, query_id, doc_id, label: feature_map)
        query_ids = test_datasets[dataset_name].map(lambda feature_map, query_id, doc_id, label: query_id)
        query_ids = [elem.numpy().decode("utf-8") for elem in query_ids.unbatch()]
        batch_results = [model.predict(batch_sample) for batch_sample in tqdm(ds_test_X)]
        comp_ndcgs = compute_ndcg_results(batch_results, ds_test_y, cutoffs)

        for cutoff, ndcg_values in comp_ndcgs.items():
            for i, value in enumerate(ndcg_values):
                res.append(CSVLine(dataset=dataset_name,
                                   cutoff=cutoff,
                                   query_id=query_ids[i],
                                   ndcg=value))

    pd.DataFrame(res).to_csv(args.output_file, index=False)


if __name__ == '__main__':
    main()
