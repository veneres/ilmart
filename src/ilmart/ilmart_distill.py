import itertools

import numpy as np
from collections import defaultdict
import lightgbm as lgbm


class IlmartDistill:
    def __init__(self, model: lgbm.Booster, distill_mode="full", n_sample=None):
        self.model = model
        self.feat_name_to_index = {feat: i for i, feat in enumerate(self.model.dump_model()['feature_names'])}
        self.feat_min = {}
        self.feat_max = {}
        for feat_name, feat_info in self.model.dump_model()["feature_infos"].items():
            feat_index = self.feat_name_to_index[feat_name]
            feat_range = feat_info["max_value"] - feat_info["min_value"]
            self.feat_min[feat_index] = feat_info["min_value"] - feat_range * 0.5
            self.feat_max[feat_index] = feat_info["max_value"] + feat_range * 0.5
        self.n_sample = n_sample
        self.distill_mode = distill_mode

        # To be computed later
        self.hist = None
        self.splitting_values = None
        self.__create_hist_dict()

    def __compute_hist(self, tree_structure: dict, feat_used: tuple, feat_min_max=None):
        if feat_min_max is None:
            feat_min_max = np.array([[self.feat_min[feat], self.feat_max[feat]] for feat in feat_used], dtype='f')
        if "leaf_index" in tree_structure:
            limits = []
            for i, feat in enumerate(feat_used):
                start = np.nonzero(np.isclose(self.splitting_values[feat], feat_min_max[i][0]))[0][0]
                try:
                    end = np.nonzero(np.isclose(self.splitting_values[feat], feat_min_max[i][1]))[0][0]
                except Exception as e:
                    end = len(self.splitting_values[feat]) - 1
                limits.append((start, end))

            selection = self.hist[feat_used]
            slicing = tuple([slice(start, end) for (start, end) in limits])
            selection[slicing] += tree_structure["leaf_value"]
            return

        split_index = feat_used.index(tree_structure["split_feature"])
        if "left_child" in tree_structure:
            new_min_max = np.copy(feat_min_max)
            new_min_max[split_index][1] = min(new_min_max[split_index][1], tree_structure["threshold"])
            self.__compute_hist(tree_structure["left_child"], feat_used, feat_min_max=new_min_max)

        if "right_child" in tree_structure:
            new_min_max = np.copy(feat_min_max)
            new_min_max[split_index][0] = max(new_min_max[split_index][0], tree_structure["threshold"])
            self.__compute_hist(tree_structure["right_child"], feat_used, feat_min_max=new_min_max)

        return

    @staticmethod
    def __splitting_values(tree_structure, splitting_values_forest, feat_used=None):
        split_feat = tree_structure.get("split_feature", None)
        if split_feat is None:
            return feat_used
        if feat_used is None:
            feat_used = set()
        feat_used.add(split_feat)
        splitting_values_forest[split_feat].add(tree_structure["threshold"])
        IlmartDistill.__splitting_values(tree_structure["left_child"], splitting_values_forest, feat_used)
        IlmartDistill.__splitting_values(tree_structure["right_child"], splitting_values_forest, feat_used)
        return feat_used

    def __create_hist_dict(self):
        self.hist = {}
        feats_used = []
        tree_infos = self.model.dump_model()["tree_info"]

        splitting_values_set = defaultdict(set)
        self.splitting_values = {}

        # Retrive all the splitting values
        for tree_info in tree_infos:
            tree_structure = tree_info["tree_structure"]
            feats_used.append(IlmartDistill.__splitting_values(tree_structure, splitting_values_set))

        if self.distill_mode == "full":
            # Add maximum and minimum to have the complete range
            for feat in splitting_values_set.keys():
                splitting_values_set[feat].add(self.feat_max[feat])
                splitting_values_set[feat].add(self.feat_min[feat])

            # From the set created to a numpy array with all the values and saved on the current object
            for feat, values in splitting_values_set.items():
                self.splitting_values[feat] = np.array(sorted(list(splitting_values_set[feat])))
        else:
            feat_infos = self.model.dump_model()["feature_infos"]
            for feat, infos in feat_infos.items():
                feat_i = self.feat_name_to_index[feat]
                # self.n_sample + 1 because we want exactly self.n_sample bins
                step = (self.feat_max[feat_i] - self.feat_min[feat_i]) / (self.n_sample + 1)
                self.splitting_values[feat_i] = np.arange(self.feat_min[feat_i], self.feat_max[feat_i], step)

        # Create a numpy array with shape corresponding to the feature dimension
        for feat_used in feats_used:
            feats_key = tuple(sorted(feat_used))
            if feats_key not in self.hist:
                shape = tuple([len(self.splitting_values[feat]) - 1 for feat in feats_key])
                self.hist[feats_key] = np.zeros(shape)

        # Compute hist for each tree
        if self.distill_mode == "full":
            for tree_info, feats in zip(tree_infos, feats_used):
                tree_structure = tree_info["tree_structure"]
                feats_key = tuple(sorted(feats))
                self.__compute_hist(tree_structure, feats_key)
        else:
            for feats_used in self.hist.keys():
                mid_points = [(self.splitting_values[feat_used][1:] + self.splitting_values[feat_used][:-1]) / 2
                              for feat_used in feats_used]
                for coord, value in enumerate(itertools.product(*mid_points)):
                    sample = np.zeros(self.model.num_feature())
                    for i, feat_i in enumerate(feats_used):
                        sample[feat_i] = value[i]

                    sample = sample.reshape((1, -1))
                    if len(feats_used) == 1:
                        self.hist[feats_used][coord] = self.model.predict(sample)
                    else:
                        self.hist[feats_used][coord // self.n_sample, coord % self.n_sample] = self.model.predict(
                            sample)

    @staticmethod
    def __predict(row, model, interactions_limit=-1):
        res = 0
        interaction_to_exclude = []
        if interactions_limit != -1:
            inter_contrib = [(feats, value)for feats, value in model.expected_contribution().items() if len(feats) > 1]
            inter_contrib.sort(key=lambda x: x[1], reverse=True)
            interaction_to_exclude = [feats for feats, value in inter_contrib[interactions_limit:]]
        for feats_hist, hist in model.hist.items():
            if feats_hist in interaction_to_exclude:
                continue
            indices = []
            for feat in feats_hist:
                index_to_add = np.searchsorted(model.splitting_values[feat], row[feat])
                index_to_add -= 1
                index_to_add = max(0, index_to_add)
                index_to_add = min(len(model.splitting_values[feat]) - 2, index_to_add)
                indices.append(index_to_add)
            res += hist[tuple(indices)]
        return res

    def predict(self, X, interactions_limit=-1):
        res = np.apply_along_axis(IlmartDistill.__predict, 1, X, self, interactions_limit=interactions_limit)
        return res

    def expected_contribution(self):
        return {feats: np.abs(hist).mean() for feats, hist in self.hist.items()}
