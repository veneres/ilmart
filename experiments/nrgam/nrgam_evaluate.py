#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_ranking as tfr
import pickle
import argparse
from tqdm import tqdm
from rankeval.metrics.ndcg import NDCG
from collections import defaultdict
import yahoo_dataset
import numpy as np


DATASET_DICT = {
    "web30k": "mslr_web/30k_fold1",
    "istella": "istella/s",
    "yahoo": "yahoo"
}

LOG_NORMALIZATION = {
    "web30k": False,
    "istella": True,
    "yahoo": False
}
BATCH_SIZE = 128
NORMALIZATION_CONSTANT = 10


def ds_transform(ds, log=False):
    ds = ds.map(
        lambda feature_map: {key: tf.where(value < 10 ** 6, value, 10 ** 6) for key, value in feature_map.items()})
    ds = ds.map(lambda feature_map: {
        "_mask": tf.ones_like(feature_map["label"], dtype=tf.bool),
        **feature_map
    })
    ds = ds.padded_batch(batch_size=BATCH_SIZE)
    ds = ds.map(lambda feature_map: (feature_map, tf.where(feature_map["_mask"], feature_map.pop("label"), -1.)))
    if log:
        ds = ds.map(
            lambda feature_map, label: (
                {key: value + NORMALIZATION_CONSTANT for key, value in feature_map.items() if key != "_mask"}, label))
        ds = ds.map(
            lambda feature_map, label: (
                {key: tf.math.log1p(value) for key, value in feature_map.items() if key != "_mask"}, label))
    else:
        ds = ds.map(
            lambda feature_map, label: ({key: value for key, value in feature_map.items() if key != "_mask"}, label))
    return ds


def compute_ndcg_results(batch_results, ds_test_y, cutoffs):
    ndcg_results = defaultdict(list)
    for batch_id, batch_y_true in tqdm(enumerate(ds_test_y)):
        for query_in_batch, y_true_padded in enumerate(batch_y_true):
            start_padding_index = np.argmax(y_true_padded == -1)
            y_true = y_true_padded[:start_padding_index].numpy()
            y_pred = np.array(batch_results[batch_id][query_in_batch][:start_padding_index])
            for cutoff in cutoffs:
                ndcg = NDCG(cutoff=cutoff, no_relevant_results=1, implementation="exp")
                ndcg_results[cutoff].append(ndcg.eval_per_query(y_true, y_pred))
    return ndcg_results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the accuracy of Neural Rank GAM for the three dataset: istella, web30k, yahoo")

    parser.add_argument("-base_dir", default="../best_models/nrgam", type=str,
                        help="Base path where the models are saved")
    parser.add_argument("-output_file", default="../results/ndcg/nrgam.pickle", type=str,
                        help="Path of the model to continue to train")

    args = parser.parse_args()
    base_path = args.base_dir
    model_paths = {
        "istella": f"{base_path}/istella_model",
        "web30k": f"{base_path}/web30k_model",
        "yahoo": f"{base_path}/yahoo_model",
    }

    best_tf_models = {}
    for name, path in model_paths.items():
        best_tf_models[name] = tf.keras.models.load_model(path)

    test_datasets = {}
    for name in model_paths.keys():
        test_datasets[name] = ds_transform(tfds.load(DATASET_DICT[name], split="test"), log=LOG_NORMALIZATION[name])

    ndcgs_nrgam = {}
    cutoffs = [1, 5, 10]
    for name, model in best_tf_models.items():
        ds_test_y = test_datasets[name].map(lambda feature_map, label: label)
        ds_test_X = test_datasets[name].map(lambda feature_map, label: feature_map)
        batch_results = [model.predict(batch_sample) for batch_sample in tqdm(ds_test_X)]
        ndcgs_nrgam[name] = compute_ndcg_results(batch_results, ds_test_y, cutoffs)

    with open(args.output_file, "wb") as f:
        pickle.dump(ndcgs_nrgam, f)


if __name__ == '__main__':
    main()
