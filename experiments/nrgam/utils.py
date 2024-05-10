from pathlib import Path

import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm

import yahoo_dataset
import web30k_dataset
import istella_dataset
from rankeval.dataset.dataset import Dataset as RankEvalDataset
import os

BATCH_SIZE = 128
NORMALIZATION_CONSTANT = 10

# The minimum value in istella for 4 features (50, 134, 148, 176) could be slightly less than 0,
# and to avoid numerical issue with the log1p transformation we added a constant value to each feature.
NORMALIZATION_CONSTANT = 10

LOG_NORMALIZATION = {
    "web30k": False,
    "istella": True,
    "yahoo": False
}


def ds_transform(ds, log=False, train=True):
    ds = ds.map(
        lambda feature_map: {
            key: tf.where(value < 10 ** 6, value, 10 ** 6) if key not in ["query_id", "doc_id", "label"] else value
            for key, value in feature_map.items()
        })
    ds = ds.map(lambda feature_map: {
        "_mask": tf.ones_like(feature_map["label"], dtype=tf.bool),
        **feature_map
    })
    ds = ds.padded_batch(batch_size=BATCH_SIZE)
    ds = ds.map(lambda feature_map: (feature_map,
                                     feature_map.pop("query_id"),
                                     tf.where(feature_map["_mask"], feature_map.pop("doc_id"), -1),
                                     tf.where(feature_map["_mask"], feature_map.pop("label"), -1),
                                     ))

    ds = ds.map(
        lambda feature_map, query_id, doc_id, label: (
            {key: value for key, value in feature_map.items() if key != "_mask"},
            query_id,
            doc_id,
            label))

    if log:
        ds = ds.map(
            lambda feature_map, query_id, doc_id, label: (
                {key: tf.math.log1p(value + NORMALIZATION_CONSTANT)
                 for key, value in feature_map.items() if key != "_mask"},
                query_id,
                doc_id,
                label))

    if train:
        ds = ds.map(lambda feature_map, query_id, doc_id, label: (feature_map, label))

    return ds


def test_tfds_wrappers():
    DATASETS = ["web30k", "istella", "yahoo"]

    DATA_HOME = os.environ.get('RANKEVAL_DATA', os.path.join('~', 'rankeval_data'))
    DATA_HOME = os.path.expanduser(DATA_HOME)
    DICT_NAME_FOLDER = {
        "web30k": f"{DATA_HOME}/msn30k/dataset/Fold1",
        "yahoo": f"{DATA_HOME}/yahoo/set1",
        "istella": f"{DATA_HOME}/istella-sample/dataset/sample"
    }

    for dataset_name in tqdm(DATASETS, desc="Datasets"):
        for split in tqdm(["train", "vali", "test"], desc="Splits", leave=False):
            dataset_rankeval = RankEvalDataset.load(str(Path(DICT_NAME_FOLDER[dataset_name]) / f"{split}.txt"))
            dataset_tfds = tfds.load(f"{dataset_name}_rankeval", split=split)

            query_ids_rankeval = []
            query_n_docs_rankeval = {}
            for elem in tqdm(dataset_rankeval.query_iterator(), desc="Queries iterator rankeval", leave=False):
                query_id = elem[0]
                query_ids_rankeval.append(query_id)
                # elem[1] is the start offset and elem[2] is the end offset
                query_n_docs_rankeval[query_id] = elem[2] - elem[1]

            query_ids_tfds = []
            query_n_docs_tfds = {}
            for elem in tqdm(dataset_tfds, desc="Queries iterator tfds", leave=False):
                query_id = int(elem["query_id"].numpy().decode("utf-8"))
                query_ids_tfds.append(query_id)
                query_n_docs_tfds[query_id] = len(elem["doc_id"])

            # First check that the ids are the same
            assert sorted(query_ids_rankeval) == sorted(query_ids_tfds)

            # Second check that each query have the same number of documents
            for query_id in query_ids_rankeval:
                assert query_n_docs_rankeval[query_id] == query_n_docs_tfds[query_id]

if __name__ == '__main__':
   test_tfds_wrappers()
