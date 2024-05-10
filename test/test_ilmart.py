import unittest
import lightgbm as lgbm
from rankeval.dataset.dataset import Dataset as RankEvalDataset

from ilmart import Ilmart

LGBM_PARAMS = {
    "objective": "lambdarank",
    "min_data_in_leaf": 50,
    "min_sum_hessian_in_leaf": 0,
    "lambdarank_truncation_level": 10,
    "num_threads": 20,
    "eval_at": 10,
    "force_col_wise": True,
    "verbosity": -1
}


class IlmartTestCase(unittest.TestCase):
    def test_inter_greedy_ilmart(self, n_inter=5):

        train_dataset = RankEvalDataset.load("train.txt")
        vali_dataset = RankEvalDataset.load("train.txt")
        ilmart_model = Ilmart(lgbm_params=LGBM_PARAMS,
                              verbose=True,
                              main_effect_model_file="web30k_10_main_effects.lgbm",
                              inter_strategy="greedy")
        ilmart_model.fit_inter_effects(1000, train=train_dataset, vali=vali_dataset, num_interactions=n_inter)

        tree_df = ilmart_model.get_model().trees_to_dataframe()
        all_pairs = set()
        singletons = set()
        for tree_index in tree_df["tree_index"].unique():
            tree_df_per_index = tree_df[tree_df["tree_index"] == tree_index]
            feat_used = [feat for feat in tree_df_per_index["split_feature"].unique() if feat is not None]
            assert len(feat_used) <= 2
            if len(feat_used) > 1:
                all_pairs.add(tuple(sorted(feat_used)))
            elif len(feat_used) == 1:
                singletons.add(feat_used[0])
        for f1, f2 in all_pairs:
            assert f1 in singletons and f2 in singletons

        assert len(all_pairs) <= n_inter

        print(f"{all_pairs=}")
        print(f"{singletons=}")

    def test_inter_prev_ilmart(self, n_inter=5):

        train_dataset = RankEvalDataset.load("train.txt")
        vali_dataset = RankEvalDataset.load("train.txt")

        main_effect_model = lgbm.Booster(model_file="web30k_10_main_effects.lgbm")
        feat_importances = [(feat_ids, value) for feat_ids, value in
                            enumerate(main_effect_model.feature_importance("gain"))]
        feat_importances.sort(key=lambda x: x[1], reverse=True)
        most_important_feat = [feat_id for feat_id, value in feat_importances][:5]

        ilmart_model = Ilmart(lgbm_params=LGBM_PARAMS,
                              verbose=True,
                              main_effect_model_file="web30k_10_main_effects.lgbm",
                              inter_strategy="prev")
        ilmart_model.fit_inter_effects(1000, train=train_dataset, vali=vali_dataset, num_interactions=n_inter)

        tree_df = ilmart_model.get_model().trees_to_dataframe()
        all_pairs = set()
        singletons = set()
        for tree_index in tree_df["tree_index"].unique():
            tree_df_per_index = tree_df[tree_df["tree_index"] == tree_index]
            feat_used = [feat for feat in tree_df_per_index["split_feature"].unique() if feat is not None]
            assert len(feat_used) <= 2
            if len(feat_used) > 1:
                all_pairs.add(tuple(sorted(feat_used)))
            elif len(feat_used) == 1:
                singletons.add(feat_used[0])
        for f1, f2 in all_pairs:
            f1_index = main_effect_model.feature_name().index(f1)
            f2_index = main_effect_model.feature_name().index(f2)
            assert f1 in singletons and f2 in singletons
            assert f1_index in most_important_feat and f2_index in most_important_feat

        assert len(all_pairs) <= n_inter

        print(f"{all_pairs=}")
        print(f"{singletons=}")

    # TODO better check if the feat used are really the one that contribute the most
    def test_inter_contrib_ilmart(self, n_inter=5):

        train_dataset = RankEvalDataset.load("train.txt")
        vali_dataset = RankEvalDataset.load("train.txt")

        ilmart_model = Ilmart(lgbm_params=LGBM_PARAMS,
                              verbose=True,
                              main_effect_model_file="web30k_10_main_effects.lgbm",
                              inter_strategy="contrib")
        ilmart_model.fit_inter_effects(1000, train=train_dataset, vali=vali_dataset, num_interactions=n_inter)

        tree_df = ilmart_model.get_model().trees_to_dataframe()
        all_pairs = set()
        singletons = set()
        for tree_index in tree_df["tree_index"].unique():
            tree_df_per_index = tree_df[tree_df["tree_index"] == tree_index]
            feat_used = [feat for feat in tree_df_per_index["split_feature"].unique() if feat is not None]
            assert len(feat_used) <= 2
            if len(feat_used) > 1:
                all_pairs.add(tuple(sorted(feat_used)))
            elif len(feat_used) == 1:
                singletons.add(feat_used[0])
        for f1, f2 in all_pairs:
            assert f1 in singletons and f2 in singletons

        assert len(all_pairs) <= n_inter

        print(f"{all_pairs=}")
        print(f"{singletons=}")


if __name__ == '__main__':
    unittest.main()
