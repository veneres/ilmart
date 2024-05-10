from __future__ import annotations

import argparse
from collections import defaultdict, namedtuple
from pathlib import Path

import pandas as pd
from rankeval.analysis.statistical import _randomization
import itertools
from joblib import parallel_backend, Parallel, delayed
from scipy.stats import ttest_rel

from tqdm import tqdm

StatCSVLine = namedtuple("StatCSVLine", ["p_value", "max_interaction", "max_tree_interaction"])


def compute_stat_sign(ilmart_res: pd.DataFrame,
                      lmart_res: pd.DataFrame,
                      max_interaction: int,
                      max_tree_interaction: int):
    ilmart_res = ilmart_res[(ilmart_res["max_interactions"] == max_interaction) &
                            (ilmart_res["max_tree_interactions"] == max_tree_interaction)]

    ilmart_res = ilmart_res.sort_values(by=["query_id"])
    lmart_res = lmart_res.sort_values(by=["query_id"])

    ilmart_ndcg_values = ilmart_res["ndcg"].values

    lmart_ndcg_values = lmart_res["ndcg"].values

    statistic, p_value = ttest_rel(ilmart_ndcg_values, lmart_ndcg_values)
    if p_value > 0.01:
        print(f"The mean of of the distribution of ilmart ndcgs is not significantly less than the mean of the full "
              f"lambdamart ndcgs for {max_interaction} and {max_tree_interaction} - {p_value}")
    return StatCSVLine(p_value, max_interaction, max_tree_interaction)


def main():
    parser = argparse.ArgumentParser(description="Constrained lmart vs lmart statistical significance check")
    parser.add_argument("--constrained_res",
                        type=str,
                        help="Path to the results of the constrained model",
                        default="/data/tradeoff/ft/eval.csv")
    parser.add_argument("--lmart_res",
                        type=str,
                        help="Path to the results of the full model",
                        default="/data/lambdamart/eval.csv")
    parser.add_argument("--out_file",
                        type=str,
                        help="Output folder the pvalues of the statistical significance test",
                        default="/data/tradeoff/ft/stat_sign.csv")
    parser.add_argument("--n_jobs", type=int, help="Number of core to use, default=8", default=50)

    args = parser.parse_args()

    out_path = Path(args.out_file)

    n_jobs = args.n_jobs

    ilmart_res = pd.read_csv(args.constrained_res)
    ilmart_res = ilmart_res[ilmart_res["cutoff"] == 10]
    max_interactions = ilmart_res["max_interactions"].unique()
    max_tree_interactions = ilmart_res["max_tree_interactions"].unique()

    lmart_res = pd.read_csv(args.lmart_res)

    lmart_res = lmart_res[(lmart_res["cutoff"] == 10) & (lmart_res["dataset"] == "web30k")]

    all_comb = list(itertools.product(max_interactions, max_tree_interactions))
    with parallel_backend('multiprocessing', n_jobs=n_jobs):
        stat_diff = Parallel()(delayed(compute_stat_sign)(ilmart_res,
                                                          lmart_res,
                                                          max_interaction,
                                                          max_tree_interaction)
                               for max_interaction, max_tree_interaction in tqdm(all_comb))


    stat_diff = pd.DataFrame(stat_diff)
    stat_diff.to_csv(out_path, index=False)


if __name__ == '__main__':
    main()
