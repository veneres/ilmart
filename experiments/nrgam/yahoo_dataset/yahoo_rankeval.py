"""yahoo_rankeval dataset."""
from pathlib import Path

import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow_datasets.ranking.libsvm_ranking_parser import LibSVMRankingParser
import os

""" 
The dataset cannot be shared online due to license constraint, so the download phase is skipped and the data will be
loaded from the folder available in the PATH variable below. 
The folder must contain three file:
- train.txt
- vali.txt
- test.txt
"""
DATA_HOME = os.environ.get('RANKEVAL_DATA', os.path.join('~', 'rankeval_data'))
DATA_HOME = os.path.expanduser(DATA_HOME)

PATH = Path(DATA_HOME) / "yahoo" / "set1"

_DESCRIPTION = """
C14 Yahoo! Learning to Rank Challenge, version 1.0

Machine learning has been successfully applied to web search ranking and the goal of this dataset to benchmark such 
machine learning algorithms. The dataset consists of features extracted from (query,url) pairs along with relevance 
judgments. The queries, ulrs and features descriptions are not given, only the feature values are.

There are two datasets in this distribution: a large one and a small one. Each dataset is divided in 3 sets: training, 
validation, and test. Statistics are as follows: Set 1 Set 2 Train Val Test Train Val Test 
# queries 19,944 2,994 6,983 1,266 1,266 3,798 # urls 473,134 71,083 165,660 34,815 34,881 103,174 # features 519 596

Number of features in the union of the two sets: 700; in the intersection: 415. Each feature has been normalized to be 
in the [0,1] range.

Each url is given a relevance judgment with respect to the query. There are 5 levels of relevance from 0 
(least relevant) to 4 (most relevant). 
"""

_CITATION = """
@inproceedings{chapelle_yahoo_2011,
	title = {Yahoo! {Learning} to {Rank} {Challenge} {Overview}},
	url = {https://proceedings.mlr.press/v14/chapelle11a.html},
	language = {en},
	urldate = {2022-02-10},
	booktitle = {Proceedings of the {Learning} to {Rank} {Challenge}},
	publisher = {PMLR},
	author = {Chapelle, Olivier and Chang, Yi},
	month = jan,
	year = {2011},
	note = {ISSN: 1938-7228},
	pages = {1--24},
	}
"""
_FEATURE_NAMES = {n: f"feature_{n}" for n in range(1, 700)}
_LABEL_NAME = "label"


class YahooRankeval(tfds.core.GeneratorBasedBuilder):
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
            homepage='https://webscope.sandbox.yahoo.com/catalog.php?datatype=c',
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
