from pathlib import Path

from rankeval.metrics import NDCG
from ilmart.utils import load_datasets
import argparse
import pickle
import json


def evaluate_and_save(models_dict, rankeval_datasets, file_out):
    cutoffs = [1, 5, 10]

    ndcgs_ebm = {}
    for name, model in models_dict.items():
        print(f"Evaluating: {name}")
        ndcgs = {}
        for cutoff in cutoffs:
            ndcg = NDCG(cutoff=cutoff, no_relevant_results=1, implementation="exp")
            res = ndcg.eval(rankeval_datasets[name]["test"], model.predict(rankeval_datasets[name]["test"].X))
            ndcgs[cutoff] = res[1]
            print(f"\tCutoff {cutoff} (mean): {res[0]}")
        ndcgs_ebm[name] = ndcgs
    with open(file_out, "wb") as f:
        print(f"Writing results to {file_out}")
        pickle.dump(ndcgs_ebm, f)


def main():
    parser = argparse.ArgumentParser(description="EBM benchmark evaluation")
    parser.add_argument("--config_path",
                        type=str,
                        default="config.json",
                        help="""
                 Path to the JSON file containing the configuration for the benchmark. It contains the following keys:
                 - best_models_dir: where to find the best models previously created.
                 - path_eval: main path to save the NDCG results.
                 """)

    args = parser.parse_args()

    with open(args.config_path) as f:
        try:
            json_args = json.load(f)
            best_models_dir = json_args["best_models_dir"]
            path_eval = json_args["path_eval"]
        except Exception as e:
            print(f"Problems reading the configuration file {args.config_path} ")
            print(e)

    rankeval_datasets = load_datasets()

    best_ebm_no_inter = {}
    for name in rankeval_datasets.keys():
        with open(f"{best_models_dir}/without_inter/{name}.pickle", "rb") as f:
            model = pickle.load(f)
            best_ebm_no_inter[name] = model
    best_ebm_inter = {}
    for name in rankeval_datasets.keys():
        with open(f"{best_models_dir}/with_inter/{name}.pickle", "rb") as f:
            model = pickle.load(f)
            best_ebm_inter[name] = model

    Path(path_eval).mkdir(parents=True, exist_ok=True)
    evaluate_and_save(best_ebm_no_inter, rankeval_datasets, f"{path_eval}/ebm.pickle")
    evaluate_and_save(best_ebm_inter, rankeval_datasets, f"{path_eval}/ebm_i.pickle")


if __name__ == '__main__':
    main()
