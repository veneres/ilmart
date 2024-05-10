"""web30k_rankeval dataset."""
from pathlib import Path

import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow_datasets.ranking.libsvm_ranking_parser import LibSVMRankingParser
import os

DATA_HOME = os.environ.get('RANKEVAL_DATA', os.path.join('~', 'rankeval_data'))
DATA_HOME = os.path.expanduser(DATA_HOME)

PATH = Path(DATA_HOME) / "msn30k" / "dataset" / "Fold1"

_DESCRIPTION = """ 
The datasets are machine learning data, in which queries and urls are represented by IDs. The datasets consist of 
feature vectors extracted from query-url pairs along with relevance judgment labels:

(1) The relevance judgments are obtained from a retired labeling set of a commercial web search engine (Microsoft Bing),
 which take 5 values from 0 (irrelevant) to 4 (perfectly relevant).

(2) The features are basically extracted by us, and are those widely used in the research community.

In the data files, each row corresponds to a query-url pair. The first column is relevance label of the pair,
 the second column is query id, and the following columns are features. The larger value the relevance label has, 
 the more relevant the query-url pair is. A query-url pair is represented by a 136-dimensional feature vector.

This is a wrapper for the files download from the rankeval package of MSLR-WEB30K.
"""

_CITATION = """
@article{DBLP:journals/corr/QinL13,
  author    = {Tao Qin and
               Tie{-}Yan Liu},
  title     = {Introducing {LETOR} 4.0 Datasets},
  journal   = {CoRR},
  volume    = {abs/1306.2597},
  year      = {2013},
  url       = {http://arxiv.org/abs/1306.2597},
  timestamp = {Mon, 01 Jul 2013 20:31:25 +0200},
  biburl    = {http://dblp.uni-trier.de/rec/bib/journals/corr/QinL13},
  bibsource = {dblp computer science bibliography, http://dblp.org}
}
"""
_FEATURE_NAMES = {n: f"feature_{n}" for n in range(1, 137)}
_LABEL_NAME = "label"


class Web30kRankeval(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for web30k dataset."""

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
            homepage='https://www.microsoft.com/en-us/research/project/mslr/',
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
