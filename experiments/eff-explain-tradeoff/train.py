import argparse
import json
from pathlib import Path

import lightgbm as lgbm

from ilmart.utils import load_datasets

from tqdm import tqdm

COMMON_PARAMS = {
    "objective": "lambdarank",
    "min_data_in_leaf": 50,
    "min_sum_hessian_in_leaf": 0,
    "lambdarank_truncation_level": 10,
    "num_threads": 60,
    "eval_at": 10,
    "force_col_wise": True,
    "verbosity": -1,
    "learning_rate": 0.05,
    "num_leaves": 127
}

def main():
    # parse outputdir from terminal
    parser = argparse.ArgumentParser(description="Analysis of the tradeoff between interpretability and explainability.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("outputdir", type=str, help="Output directory for the models.",
                        default="/data/tradeoff")
    parser.add_argument("--config_path", type=str, help="Output directory for the models.",
                        default="./config.json")
    args = parser.parse_args()
    output_dir = args.outputdir



    with open(args.config_path, "r") as f:
        config = json.load(f)
        max_tree_interactions = config["max_tree_interactions"]
        max_interactions_range = config["max_interactions_range"]
        dataset = config["dataset"]
    rankeval_datasets = load_datasets([dataset], verbose=True)[dataset]

    train_ds = rankeval_datasets["train"]
    vali_ds = rankeval_datasets["vali"]
    train_lgbm = lgbm.Dataset(train_ds.X, group=train_ds.get_query_sizes(), label=train_ds.y, free_raw_data=False)
    vali_lgbm = lgbm.Dataset(vali_ds.X, group=vali_ds.get_query_sizes(), label=vali_ds.y, free_raw_data=False)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for max_tree_interactions in tqdm(max_tree_interactions, desc="max_tree_interactions"):
        for max_interactions in tqdm(max_interactions_range, desc="max_interactions", leave=False):
            params = COMMON_PARAMS.copy()
            params["max_tree_interactions"] = max_tree_interactions
            params["max_interactions"] = max_interactions

            callbacks = [lgbm.early_stopping(100, verbose=False)]
            model = lgbm.train(params,
                               train_lgbm,
                               num_boost_round=10000,
                               valid_sets=[vali_lgbm],
                               callbacks=callbacks)

            model.save_model(output_dir / Path(f"{max_tree_interactions}_{max_interactions}.lgbm"))


if __name__ == '__main__':
    main()
