# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

from ilmart.utils import load_datasets
from collections import namedtuple
import numpy as np
import pandas as pd

rankeval_datasets = load_datasets(verbose=True)

EntryDF = namedtuple("EntryDF",
                     ["dataset_name", "subset", "n_feat", "n_queries", "n_instances", "avg_doc_query", "avg_label_0",
                      "avg_label_1", "avg_label_2", "avg_label_3", "avg_label_4", "without_relevant", "n_queries_less_than_10_docs"])
entries = []
for dataset_name, datasets_by_name in rankeval_datasets.items():
    for subpart, dataset in datasets_by_name.items():
        n_feat = dataset.n_features
        n_queries = dataset.n_queries
        n_instances = dataset.n_instances
        labels = [0, 0, 0, 0, 0]
        for label_id in range(len(labels)):
            labels[label_id] = np.sum(dataset.y == label_id)
        no_rel_q = 0
        n_queries_less_than_10_docs = 0
        for query_id, start, end in dataset.query_iterator():
            if np.sum(dataset.y[start:end] != 0) == 0:
                no_rel_q += 1
            if end - start < 10:
                n_queries_less_than_10_docs += 1

        avg_doc_query = n_instances / n_queries
        entries.append(EntryDF(dataset_name=dataset_name,
                               subset=subpart,
                               n_feat=n_feat,
                               n_queries=n_queries,
                               n_instances=n_instances,
                               avg_doc_query=avg_doc_query,
                               avg_label_0=labels[0] / n_queries,
                               avg_label_1=labels[1] / n_queries,
                               avg_label_2=labels[2] / n_queries,
                               avg_label_3=labels[3] / n_queries,
                               avg_label_4=labels[4] / n_queries,
                               without_relevant=no_rel_q,
                               n_queries_less_than_10_docs =n_queries_less_than_10_docs))

df = pd.DataFrame(entries)

df

for query_id, start, end in dataset.query_iterator():
    print(query_id, start, end)

dataset.get_query_sizes()


