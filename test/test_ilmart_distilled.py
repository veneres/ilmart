import unittest
import lightgbm as lgbm
from rankeval.dataset.dataset import Dataset as RankEvalDataset
from src import IlmartDistill
from src import is_interpretable
import numpy as np


class IlmartDistlliedTestCase(unittest.TestCase):
    def test_only_main_effects(self):
        test_dataset = RankEvalDataset.load("test.txt")
        model = lgbm.Booster(model_file="web30k_10_main_effects.lgbm")
        print(model.num_trees())
        is_interpretable(model)
        distilled = IlmartDistill(model)
        res = model.predict(test_dataset.X)

        distill_res = distilled.predict(test_dataset.X)

        assert np.allclose(res, distill_res)

    def test_inter_effects(self):
        test_dataset = RankEvalDataset.load("test.txt")
        model = lgbm.Booster(model_file="20_20.lgbm")
        is_interpretable(model)
        distilled = IlmartDistill(model)
        res = model.predict(test_dataset.X)
        distill_res = distilled.predict(test_dataset.X)

        assert np.allclose(res, distill_res)


if __name__ == '__main__':
    unittest.main()
