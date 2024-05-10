"""istella_rankeval dataset."""
from pathlib import Path

import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow_datasets.ranking.libsvm_ranking_parser import LibSVMRankingParser
import os

""" 
This is a wrapper for the files download from the rankeval package
"""
DATA_HOME = os.environ.get('RANKEVAL_DATA', os.path.join('~', 'rankeval_data'))
DATA_HOME = os.path.expanduser(DATA_HOME)

PATH = Path(DATA_HOME) / "istella-sample" / "dataset" / "sample"

_DESCRIPTION = """
The Istella LETOR full dataset is composed of 33,018 queries and 220 features representing each query-document pair. 
It consists of 10,454,629 examples labeled with relevance judgments ranging from 0 (irrelevant) to 4 (perfectly relevant). 
The average number of per-query examples is 316. It has been splitted in train and test sets according to a 80%-20% scheme.
This is a wrapper for the files download from the rankeval package of Istella-s.
"""

_CITATION = """
@article{10.1145/2987380,
author = {Dato, Domenico and Lucchese, Claudio and Nardini, Franco Maria and Orlando, Salvatore and Perego, Raffaele and Tonellotto, Nicola and Venturini, Rossano},
title = {Fast Ranking with Additive Ensembles of Oblivious and Non-Oblivious Regression Trees},
year = {2016},
issue_date = {April 2017},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {35},
number = {2},
issn = {1046-8188},
url = {https://doi.org/10.1145/2987380},
doi = {10.1145/2987380},
journal = {ACM Trans. Inf. Syst.},
month = {dec},
articleno = {15},
numpages = {31},
keywords = {Learning to rank, additive ensembles of regression trees, cache-awareness, document scoring, efficiency}
}
"""
_FEATURE_NAMES = {n: f"feature_{n}" for n in range(1, 221)}
_LABEL_NAME = "label"


class IstellaRankeval(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for yahoo dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        encoding = tfds.features.Encoding.ZLIB
        features = {
            name: tfds.features.Tensor(
                shape=(None,), dtype=np.float64, encoding=encoding)
            for name in _FEATURE_NAMES.values()
        }
        features[_LABEL_NAME] = tfds.features.Tensor(
            shape=(None,), dtype=np.float64, encoding=encoding)
        features["query_id"] = tfds.features.Text()
        features["doc_id"] = tfds.features.Tensor(shape=(None,), dtype=np.int64, encoding=encoding)
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(features),
            homepage='https://istella.ai/data/letor-dataset/',
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        # We do not download the dataset from the web, we just assume that is already in PATH

        splits = {
            "train": self._generate_examples(str(PATH / "train.txt")),
            "vali": self._generate_examples(str(PATH / "vali.txt")),
            "test": self._generate_examples(str(PATH / "test.txt"))
        }

        return splits

    def _generate_examples(self, path: str):
        """"Yields examples."""

        with tf.io.gfile.GFile(path, "r") as f:
            yield from LibSVMRankingParser(f, _FEATURE_NAMES, _LABEL_NAME)
