from pathlib import Path
import numpy as np
import lightgbm as lgbm
from tqdm import tqdm
from rankeval.metrics import NDCG
from sklearn.model_selection import ParameterGrid
from ilmart.utils import load_datasets


def fine_tuning(train_lgbm, vali_lgbm, common_params, param_grid, verbose=True):
    param_grid_list = list(ParameterGrid(param_grid))
    best_ndcg_10 = 0
    best_model = None
    for params in tqdm(param_grid_list):
        new_params = dict(**common_params, **params)
        dict_search_res = {}
        early_stopping = lgbm.early_stopping(50, verbose=True)
        eval_result_callback = lgbm.record_evaluation(dict_search_res)
        log_eval = lgbm.log_evaluation(period=1, show_stdv=True)

        callbacks = [early_stopping, eval_result_callback]

        if verbose:
            callbacks.append(log_eval)

        new_model = lgbm.train(new_params,
                               train_lgbm,
                               num_boost_round=2000,
                               valid_sets=[vali_lgbm],
                               callbacks=callbacks)
        last_ndcg = np.max(dict_search_res['valid_0']['ndcg@10'])
        if last_ndcg > best_ndcg_10:
            best_ndcg_10 = last_ndcg
            best_model = new_model
            if verbose:
                print(f"Best NDCG@10: {best_ndcg_10}")

    return best_model


def main():
    common_params = {
        "objective": "lambdarank",
        "min_data_in_leaf": 50,
        "min_sum_hessian_in_leaf": 0,
        "num_threads": 40,
        "force_col_wise": True,
        "verbosity": -1,
        "eval_at": 10,
        "lambdarank_truncation_level": 13,
    }
    leaves = list(map(int, np.geomspace(64, 512, num=4)))
    param_grid = {'learning_rate': np.geomspace(0.001, 0.1, num=4),
                  'num_leaves': leaves}

    best_models = {}

    rankeval_datasets = load_datasets()

    models_dir = "../best_models/full"
    Path(models_dir).mkdir(parents=True, exist_ok=True)

    for name, dataset_dict in rankeval_datasets.items():
        model_path = f"{models_dir}/{name}.lgbm"
        if Path(model_path).is_file():
            print(f"Found {model_path}, loading...")
            best_models[name] = lgbm.Booster(model_file=model_path)
        else:
            train_lgbm = lgbm.Dataset(dataset_dict["train"].X,
                                      group=dataset_dict["train"].get_query_sizes(),
                                      label=dataset_dict["train"].y)
            vali_lgbm = lgbm.Dataset(dataset_dict["vali"].X,
                                     group=dataset_dict["vali"].get_query_sizes(),
                                     label=dataset_dict["vali"].y)
            best_models[name] = fine_tuning(train_lgbm, vali_lgbm, common_params, param_grid, verbose=True)
            best_models[name].save_model(f"{models_dir}/{name}.lgbm")

    res_ndcg = {}
    for name, model in best_models.items():

        test_dataset = rankeval_datasets[name]["test"]

        ndcgs = {}
        for cutoff in [1, 5, 10]:
            ndcg = NDCG(cutoff=cutoff, no_relevant_results=1, implementation="exp")
            ndcg_stats = ndcg.eval(test_dataset, model.predict(test_dataset.X))
            ndcgs[cutoff] = ndcg_stats
        res_ndcg[name] = ndcgs
    print(res_ndcg)


if __name__ == '__main__':
    main()
