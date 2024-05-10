from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy
import numpy as np
from collections import defaultdict
import lightgbm as lgbm


class IlmartDistill:
    FEAT_MIN = -np.inf
    FEAT_MAX = np.inf

    def __init__(self, model: lgbm.Booster, n_sample=None):
        self.model = model
        self.feat_name_to_index = {feat: i for i, feat in enumerate(self.model.dump_model()['feature_names'])}

        self.n_sample = n_sample

        self.feat_used = None  # can be single features or pairs of features

        # To be computed later
        self.hist = None
        self.splitting_values = None
        self._distill()

    def __compute_hist(self, tree_structure: dict, feat_used: tuple, bin_min_max=None):
        if bin_min_max is None:
            bin_min_max = np.array([[IlmartDistill.FEAT_MIN, IlmartDistill.FEAT_MAX] for _ in feat_used], dtype='f')
        if "leaf_index" in tree_structure:
            limits = []
            for i, feat in enumerate(feat_used):
                close_to_start = np.nonzero(np.isclose(bin_min_max[i][0], self.splitting_values[feat],
                                                       rtol=1e-06,
                                                       atol=1e-12), )[0]

                if len(close_to_start) > 2:
                    raise ValueError(f"Too many values close to the start of the bin {close_to_start}")

                if len(close_to_start) == 2 and bin_min_max[i][0] > 0:  # we probably have a very small number near 0
                    start = close_to_start[1]
                else:
                    start = close_to_start[0]
                close_to_end = np.nonzero(np.isclose(bin_min_max[i][1],
                                                     self.splitting_values[feat],
                                                     rtol=1e-06,
                                                     atol=1e-12))[0]
                if len(close_to_end) > 2:
                    raise ValueError(f"Too many values close to the end of the bin {close_to_end}")
                if len(close_to_end) == 2 and bin_min_max[i][1] > 0:  # we probably have two very small number
                    end = close_to_end[1]
                else:
                    end = close_to_end[0]
                limits.append((start, end))

            selection = self.hist[feat_used]
            slicing = tuple([slice(start, end) for (start, end) in limits])
            selection[slicing] += tree_structure["leaf_value"]
            return

        split_index = feat_used.index(tree_structure["split_feature"])

        if "left_child" in tree_structure:
            new_min_max = np.copy(bin_min_max)
            new_min_max[split_index][1] = min(new_min_max[split_index][1], tree_structure["threshold"])
            self.__compute_hist(tree_structure["left_child"], feat_used, bin_min_max=new_min_max)

        if "right_child" in tree_structure:
            new_min_max = np.copy(bin_min_max)
            new_min_max[split_index][0] = max(new_min_max[split_index][0], tree_structure["threshold"])
            self.__compute_hist(tree_structure["right_child"], feat_used, bin_min_max=new_min_max)

        return

    def _get_splitting_values(self, tree_structure: dict):
        split_feat = tree_structure.get("split_feature", None)
        if split_feat is None:
            return
        self.splitting_values[split_feat].add(tree_structure["threshold"])
        self._get_splitting_values(tree_structure["left_child"])
        self._get_splitting_values(tree_structure["right_child"])
        return

    def _get_feat_used(self, tree_structure: dict) -> set:
        feat_used = set()
        split_feat = tree_structure.get("split_feature", None)
        if split_feat is None:
            return feat_used
        feat_used.add(split_feat)
        feat_used = feat_used.union(self._get_feat_used(tree_structure["left_child"]))
        feat_used = feat_used.union(self._get_feat_used(tree_structure["right_child"]))
        return feat_used

    def _distill(self):
        self.hist = {}
        tree_infos = self.model.dump_model()["tree_info"]

        self.splitting_values = defaultdict(set)

        # Retrive all the splitting values and insert them inside
        for tree_info in tree_infos:
            tree_structure = tree_info["tree_structure"]
            self._get_splitting_values(tree_structure)

        # transform the sets in sorted lists and add extreme values
        self.splitting_values = {key: [IlmartDistill.FEAT_MIN] + sorted(list(values)) + [IlmartDistill.FEAT_MAX]
                                 for key, values in self.splitting_values.items()}

        # Retrive all the features used and store them in self.feat_used as a set and in feat_used_per_tree as list
        feat_used_per_tree = []
        for tree_info in tree_infos:
            tree_structure = tree_info["tree_structure"]
            feat_used_per_tree.append(tuple(sorted(list(self._get_feat_used(tree_structure)))))

        self.feat_used = set(feat_used_per_tree)

        # Create a numpy array with shape corresponding to the feature dimension to store the scores
        for feat_used in self.feat_used:
            shape = tuple([len(self.splitting_values[feat]) - 1 for feat in feat_used])
            self.hist[feat_used] = np.zeros(shape)

        # Compute hist for each tree
        for tree_info, feats_key in zip(tree_infos, feat_used_per_tree):
            tree_structure = tree_info["tree_structure"]
            self.__compute_hist(tree_structure, feats_key)

    def _predict(self, row):
        res = 0
        for feats_hist, hist in self.hist.items():
            indices = []
            for feat in feats_hist:
                feat_value = row[feat]
                index_to_add = None
                for i, value in enumerate(self.splitting_values[feat]):
                    if feat_value < value or math.isclose(feat_value, value):
                        index_to_add = i - 1
                        break
                indices.append(index_to_add)
            res += hist[tuple(indices)]

        return res

    # This is mainly for model debugging, all the functions have not been optimized
    def predict(self, X: np.ndarray):
        res = numpy.zeros(X.shape[0])
        for row_index in range(X.shape[0]):
            res[row_index] = self._predict(X[row_index, :])
        return res

    def expected_contribution(self):
        return {feats: np.abs(hist).mean() for feats, hist in self.hist.items()}

    def export(self, file_out: str | Path) -> None:
        with open(file_out, "w") as f:
            # Write number of features
            f.write(f"{self.model.num_feature()}\n")

            # Write splitting values
            f.write(f"{str(len(self.hist.keys()))}\n")

            for feats, scores in self.hist.items():
                f.write(f"{len(feats)}\n")
                for feat in feats:
                    f.write(f"{feat}\n")
                for feat in feats:
                    f.write(f"{' '.join(str(v) for v in self.splitting_values[feat])}\n")

                f.write(f"{' '.join(str(v) for v in scores.flatten())}\n")
