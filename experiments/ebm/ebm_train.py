#!/usr/bin/env python
# coding: utf-8
import pickle
import argparse
from ilmart.utils import load_datasets
import json
from interpret.glassbox import ExplainableBoostingRegressor
from rankeval.metrics import NDCG
from pathlib import Path
from tqdm import tqdm


def train(rankeval_datasets, outerbags, models_dir, n_inter):
    for name, datasets in tqdm(rankeval_datasets.items()):
        for bag in outerbags:
            path = f"{models_dir}/{name}_{bag}.pickle"
            if Path(path).exists():
                print(f"Model already computed: {name} - {bag} - {path}")
                print("Skipping...")
                continue
            ebm = ExplainableBoostingRegressor(random_state=42, interactions=n_inter, outer_bags=bag, n_jobs=40)
            ebm.fit(datasets["train"].X, datasets["train"].y)
            print(f"Writing model to: {path}")
            with open(path, "wb") as f:
                pickle.dump(ebm, f)


def train_all(rankeval_datasets, outerbags, models_dir):
    print("Train models without interactions")
    out_dir = f"{models_dir}/without_inter"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    train(rankeval_datasets, outerbags, out_dir, 0)

    print("Train models with 50 interactions")
    out_dir = f"{models_dir}/with_inter"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    train(rankeval_datasets, outerbags, out_dir, 50)


def evaluate(vali_dataset, model):
    ndcg = NDCG(cutoff=10, no_relevant_results=1, implementation="exp")
    ndcg_stats = ndcg.eval(vali_dataset, model.predict(vali_dataset.X))
    return ndcg_stats[0]


def find_best_model(rankeval_datasets, outerbags, models_dir, out_dir):
    best_models = {}

    for name, datasets in rankeval_datasets.items():
        max_ndcg_10 = 0
        best_model = None
        print(f"Evaluating models found for {name}")
        for outerbag in outerbags:
            file_path = f"{models_dir}/{name}_{outerbag}.pickle"
            with open(file_path, "rb") as f:
                ebm = pickle.load(f)
                ndcg_10 = evaluate(datasets["vali"], ebm)
                print(f"NDCG for {file_path}: {ndcg_10}")
                if ndcg_10 > max_ndcg_10:
                    print(f"Best outerbag at the moment: {outerbag}")
                    max_ndcg_10 = ndcg_10
                    best_model = ebm
        best_models[name] = best_model

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for name, model in best_models.items():
        print(f"Saving the model to {out_dir}")
        with open(f"{out_dir}/{name}.pickle", "wb") as f:
            pickle.dump(model, f)


def find_best_model_all(rankeval_datasets, outerbags, models_dir, best_models_dir):
    print("Starting searching the best model without inter")
    find_best_model(rankeval_datasets, outerbags, f"{models_dir}/without_inter", f"{best_models_dir}/without_inter")
    print("Starting searching the best model with inter")
    find_best_model(rankeval_datasets, outerbags, f"{models_dir}/with_inter", f"{best_models_dir}/with_inter")


def main():
    parser = argparse.ArgumentParser(description="EBM train script.")
    parser.add_argument("--config_path",
                        type=str,
                        default="config.json",
                        help="""
                Path to the JSON file containing the configuration for the benchmark. It contains the following keys:
                - models_dir: where to save all the models created.
                - best_models_dir: where to save the best models created (fine-tuned).
                - outerbags: values of outerbags to try.
                                 """)

    args = parser.parse_args()

    with open(args.config_path) as f:
        try:
            json_args = json.load(f)
            models_dir = json_args["models_dir"]
            print(f"{models_dir=}")
            best_models_dir = json_args["best_models_dir"]
            print(f"{best_models_dir=}")
            outerbags = json_args["outerbags"]
            print(f"{outerbags=}")
        except Exception as e:
            print(f"Problems reading the configuration file {args.config_path} ")
            print(e)

    Path(models_dir).mkdir(parents=True, exist_ok=True)

    print("Load datasets")
    rankeval_datasets = load_datasets()

    train_all(rankeval_datasets, outerbags, models_dir)

    find_best_model_all(rankeval_datasets, outerbags, models_dir, best_models_dir)


if __name__ == '__main__':
    main()
