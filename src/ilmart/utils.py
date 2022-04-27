from collections import defaultdict
import os
from rankeval.dataset.dataset import Dataset as RankEvalDataset
from rankeval.dataset.datasets_fetcher import load_dataset
from tqdm import tqdm


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


def load_datasets(datasets=None):
    rankeval_datasets = defaultdict(dict)
    datasets_to_load = DICT_NAME_FOLDER
    if datasets is not None:
        datasets_to_load = {name: DICT_NAME_FOLDER[name] for name in datasets}
    for name, dataset_folder in tqdm(datasets_to_load.items()):
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
    return rankeval_datasets


def is_interpretable(model, verbose=True):
    tree_df = model.trees_to_dataframe()
    all_pairs = set()
    singletons = set()
    for tree_index in tree_df["tree_index"].unique():
        tree_df_per_index = tree_df[tree_df["tree_index"] == tree_index]
        feat_used = [feat for feat in tree_df_per_index["split_feature"].unique() if feat is not None]
        if len(feat_used) > 2:
            print(tree_index)
            print(tree_df_per_index["split_feature"])
            return False
        if len(feat_used) > 1:
            all_pairs.add(tuple(sorted(feat_used)))
        elif len(feat_used) == 1:
            singletons.add(feat_used[0])
    for f1, f2 in all_pairs:
        if f1 not in singletons or f2 not in singletons:
            return False
    if verbose:
        print(f"len(all_pairs) = {len(all_pairs)}")
        print(f"len(singletons) = {len(singletons)}")
    return True
