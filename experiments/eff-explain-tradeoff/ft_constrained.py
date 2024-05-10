import argparse
import copy
import json
from pathlib import Path
import numpy as np
import lightgbm as lgbm
import optuna
from ilmart.utils import load_datasets


EARLY_STOPPING_ROUNDS = 100

def fun_to_optimize(trial: optuna.Trial,
                    fixed_params: dict,
                    hyper_opt_min_max: dict,
                    train_lgbm: lgbm.Dataset,
                    vali_lgbm: lgbm.Dataset,
                    num_boost_round: int):
    params = copy.deepcopy(fixed_params)

    params["learning_rate"] = trial.suggest_float('learning_rate',
                                                  hyper_opt_min_max["learning_rate"]["min"],
                                                  hyper_opt_min_max["learning_rate"]["max"],
                                                  log=True)
    params["num_leaves"] = trial.suggest_int('num_leaves',
                                             hyper_opt_min_max["num_leaves"]["min"],
                                             hyper_opt_min_max["num_leaves"]["max"])
    dict_search_res = {}
    early_stopping = lgbm.early_stopping(EARLY_STOPPING_ROUNDS, verbose=True)
    eval_result_callback = lgbm.record_evaluation(dict_search_res)

    callbacks = [early_stopping, eval_result_callback]

    lgbm.train(params, train_lgbm, num_boost_round=num_boost_round, valid_sets=[vali_lgbm], callbacks=callbacks)
    last_ndcg = np.max(dict_search_res['valid_0']['ndcg@10'])

    return last_ndcg


def main():
    parser = argparse.ArgumentParser(description="Constrained LambdaMART fine tuning")
    parser.add_argument("dataset", type=str, help="Dataset to use. Available options: web30k, yahoo, istella.")

    parser.add_argument("out", type=str, help="Output file containing the final model.")
    parser.add_argument("ft_config",
                        type=str,
                        help="""
                                Path to the JSON file containing the fine-tuning configuration. It contains the 
                                following keys:
                                - common_params: common params to use during the LGBM training.
                                - hyper_opt_min_max: the parameter grid to pass to the optimizer.
                                - boosting_rounds: the number of boosting rounds to do.
                                - n_trials: the number of trials to fine-tune the model
                                 """)

    parser.add_argument("max_tree_interactions", type=str, help="Max number of interactions per tree")
    parser.add_argument("max_interactions", type=str, help="Max number of interactions for the ensemble")

    args = parser.parse_args()

    dataset = args.dataset

    with open(args.ft_config) as f:
        json_args = json.load(f)
        n_trials = json_args["n_trials"]
        common_params = json_args["common_params"]
        boosting_rounds = json_args["boosting_rounds"]
        hyper_opt_min_max = json_args["hyper_opt_min_max"]

    output_path = Path(args.out).resolve()

    common_params["max_tree_interactions"] = int(args.max_tree_interactions)
    common_params["max_interactions"] = int(args.max_interactions)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    rankeval_datasets = load_datasets([dataset], verbose=True)

    for name, datasets in rankeval_datasets.items():
        x_train = datasets["train"].X
        y_train = datasets["train"].y
        x_train_qs = datasets["train"].get_query_sizes()
        train_lgbm = lgbm.Dataset(x_train, group=x_train_qs, label=y_train)

        x_vali = datasets["vali"].X
        y_vali = datasets["vali"].y
        x_vali_qs = datasets["vali"].get_query_sizes()
        vali_lgbm = lgbm.Dataset(x_vali, group=x_vali_qs, label=y_vali)

        config = {
            "hyper_opt_min_max": hyper_opt_min_max,
            "fixed_params": common_params,
            "train_lgbm": train_lgbm,
            "vali_lgbm": vali_lgbm,
            "num_boost_round": boosting_rounds,
        }
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: fun_to_optimize(trial, **config), n_trials=n_trials)

        best_params = dict(**study.best_trial.params, **common_params)

        early_stopping = lgbm.early_stopping(EARLY_STOPPING_ROUNDS, verbose=True)

        callbacks = [early_stopping]

        best_model = lgbm.train(best_params, train_lgbm, num_boost_round=boosting_rounds, valid_sets=[vali_lgbm],
                                callbacks=callbacks)

        best_model.save_model(output_path)


if __name__ == '__main__':
    main()
