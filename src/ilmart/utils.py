from __future__ import annotations

from collections import defaultdict
import os
from pathlib import Path

import rankeval.dataset
from rankeval.dataset.dataset import Dataset as RankEvalDataset
from rankeval.dataset.datasets_fetcher import load_dataset
from rankeval.metrics import NDCG
from tqdm import tqdm
import lightgbm as lgbm
import numpy as np

DATA_HOME = os.environ.get('RANKEVAL_DATA', os.path.join('~', 'rankeval_data'))
DATA_HOME = os.path.expanduser(DATA_HOME)

DICT_NAME_FOLDER = {
    "web30k": f"{DATA_HOME}/msn30k/dataset/Fold1",
    "yahoo": f"{DATA_HOME}/yahoo/set1",
    "istella": f"{DATA_HOME}/istella-sample/dataset/sample"
}

RANKEVAL_MAPPING = {
    "web30k": "msn30k",
    "istella": "istella-sample"
}
RANKEVAL_MAPPING_FOLD = {
    "web30k": 1,
    "istella": None
}


def load_datasets(datasets: list[str] = None, verbose: bool = False, remove_unlabeled_docs=False):
    rankeval_datasets = defaultdict(dict)
    datasets_to_load = DICT_NAME_FOLDER
    if datasets is not None:
        datasets_to_load = {name: DICT_NAME_FOLDER[name] for name in datasets}

    if verbose:
        dataset_items = tqdm(datasets_to_load.items())
    else:
        dataset_items = datasets_to_load.items()
    for name, dataset_folder in dataset_items:
        if not os.path.isdir(dataset_folder):
            if name == "yahoo":
                raise Exception(
                    """
                    For copyright reason you have to download the Yahoo dataset separately: 
                    https://webscope.sandbox.yahoo.com/catalog.php?datatype=c
                    """
                )
            # load_dataset used only to download the dataset
            load_dataset(dataset_name=RANKEVAL_MAPPING[name],
                         fold=RANKEVAL_MAPPING_FOLD[name],
                         download_if_missing=True,
                         force_download=False,
                         with_models=False)
        for split in ["train", "vali", "test"]:
            rankeval_datasets[name][split] = RankEvalDataset.load(f"{dataset_folder}/{split}.txt")
    if remove_unlabeled_docs:
        for name, datasets in rankeval_datasets.items():
            test_dataset = datasets["test"]
            query_to_keep = test_dataset.query_ids
            for query_id, query_start, query_end in test_dataset.query_iterator():
                if np.sum(test_dataset.y[query_start:query_end]) == 0:
                    query_to_keep = np.delete(query_to_keep, np.where(query_to_keep == query_id))
            rankeval_datasets[name]["test"] = datasets["test"].subset(query_to_keep)
    return rankeval_datasets


def is_interpretable(model):
    main_effects = get_effects(model, 1)
    inter_effects = get_effects(model, 2)
    # Check the maximum number of interactions per tree
    if has_inter_with_more_than_n(model, 2):
        return False
    # Check hierarchical principal
    for f1, f2 in inter_effects:
        if (f1,) not in main_effects or (f2,) not in main_effects:
            return False
    return True


def get_effects(model: lgbm.Booster, n: int):
    """

    Parameters
    ----------
    model: the model to inspect
    n: 1 for main effects, 2 for interaction effects of 2 variables, >2 for interaction effects of higher order

    Returns
    -------
    The se of features used to represent each effect

    """
    tree_df = model.trees_to_dataframe()
    effects = set()
    for tree_index in tree_df["tree_index"].unique():
        tree_df_per_index = tree_df[tree_df["tree_index"] == tree_index]
        feat_used = [feat for feat in tree_df_per_index["split_feature"].unique() if feat is not None]
        if len(feat_used) == n:
            effects.add(tuple(sorted(feat_used)))
    return effects


def has_inter_with_more_than_n(model: lgbm.Booster, n: int):
    tree_df = model.trees_to_dataframe()
    for tree_index in tree_df["tree_index"].unique():
        tree_df_per_index = tree_df[tree_df["tree_index"] == tree_index]
        feat_used = [feat for feat in tree_df_per_index["split_feature"].unique() if feat is not None]
        if len(feat_used) > n:
            return True
    return False


def has_more_than_n_inter(model: lgbm.Booster, n: int, max_tree_interactions: int):
    tree_df = model.trees_to_dataframe()
    inter_dict = defaultdict(set)
    for tree_index in tree_df["tree_index"].unique():
        tree_df_per_index = tree_df[tree_df["tree_index"] == tree_index]
        feat_used = [feat for feat in tree_df_per_index["split_feature"].unique() if feat is not None]
        if len(feat_used) != max_tree_interactions:
            continue
        inter_dict[f"{len(feat_used)}"].add(tuple(sorted(feat_used)))
        if len(inter_dict[f"{len(feat_used)}"]) > n:
            return True
    return False


def eval_dataset(model_file_path: str | Path, test_dataset: rankeval.dataset.Dataset) -> dict:
    model = lgbm.Booster(model_file=model_file_path)

    predictions = model.predict(test_dataset.X)

    # Do not save the date, usually it does not take too much time to score the entire dataset
    ndcg_dict = defaultdict(dict)
    for cutoff in [1, 5, 10]:
        ndcg = NDCG(cutoff=cutoff, no_relevant_results=1, implementation="exp")

        for qid, q_y, q_y_pred in ndcg.query_iterator(test_dataset, predictions):
            ndcg_dict[cutoff][qid] = ndcg.eval_per_query(q_y, q_y_pred)
    return ndcg_dict


if __name__ == '__main__':
    load_datasets(["web30k"])
