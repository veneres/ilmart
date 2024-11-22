import argparse
import json
import re
from collections import namedtuple
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
import subprocess
import lightgbm as lgbm

RunEntry = namedtuple("RunEntry", ["model", "run_id", "time", "exp_name"])

DELTA_LIMIT = 0.001

######################################################
# Auxiliar functions
######################################################

def check_results(results_oracle: Path, results_scorer: Path, delta_limit: float = DELTA_LIMIT):

    res_oracle = pd.read_csv(results_oracle, header=None).values

    res_scorer = pd.read_csv(results_scorer, header=None).values

    assert res_oracle.shape[0] == res_scorer.shape[0]

    assert len(np.where(res_oracle - res_scorer > delta_limit)[0]) == 0



def get_max_n_leaves(lgbm_model):
    model = lgbm.Booster(model_file=str(lgbm_model))
    df_trees = model.trees_to_dataframe()
    df_trees = df_trees[df_trees["split_feature"].isna()]
    df_trees = df_trees.groupby(["tree_index"]).count()
    n_leaves_stats = df_trees["count"].value_counts()
    max_n_leaves = max(n_leaves_stats.index)
    return max_n_leaves

######################################################
# Functions to run the experiments for each model
######################################################

def run_lightgbm(preproc_folder: Path,
                 shared_params: dict,
                 numa_args: list,
                 debug: bool) -> tuple[float, Path]:
    lgbm_symlink = preproc_folder / shared_params['lgbm_symlink_name']
    lgbm_config = preproc_folder / shared_params['prediction_config_lgbm']
    results_lgbm = preproc_folder / shared_params["result_lightgbm"]
    lgbm_args = [str(lgbm_symlink), f"config={lgbm_config}"]
    lgbm_args = numa_args + lgbm_args
    if debug:
        print(" ".join(lgbm_args))
    res = subprocess.run(lgbm_args, capture_output=True)
    # The end of output of the command is something like:
    # [LightGBM] [Info] Finished prediction in 272060.026542 milliseconds
    if debug:
        print(res.stdout.decode('utf-8'))
    ms_elapsed = float(res.stdout.decode('utf-8').split()[-2])
    return ms_elapsed, results_lgbm


def run_ilmart(preproc_folder: Path,
               shared_params: dict,
               numa_args: list,
               results_lgbm: Path,
               debug: bool):
    results_ilmart = preproc_folder / shared_params["result_ilmart"]
    ilmart_args = [str(preproc_folder / shared_params["ilmart_symlink_name"]),  # lgbm executable
                   str(preproc_folder / shared_params["ilmart_model_file_name"]),  # model file
                   str(preproc_folder / shared_params["ds_symlink_name_lgbm"]),  # dataset file
                   str(results_ilmart),  # output file
                   str(1)]  # number of reps
    ilmart_args = numa_args + ilmart_args
    if debug:
        print(" ".join(ilmart_args))

    res = subprocess.run(ilmart_args, capture_output=True)

    if debug:
        print(res.stdout.decode('utf-8'))

    ms_elapsed = float(res.stdout.decode('utf-8').split()[-1])
    # Always check the results between lightgbm and ilmart to double-check the implementation
    try:
        check_results(results_lgbm, results_ilmart)
    except AssertionError as e:
        print(f"Results differ between lightgbm and quickscorer: \n {results_lgbm} \n {results_ilmart}")
        raise e
    return ms_elapsed

def run_quickscorer(preproc_folder:Path,
                    shared_params: dict,
                    max_n_leaves: int,
                    numa_args: list,
                    results_lgbm: Path,
                    debug: bool):
    results_quickscorer = preproc_folder / shared_params["result_quickscorer"]
    exe_quickscorer = str(preproc_folder / shared_params["quickscorer_symlink_name"])
    ds_quickscorer = str(preproc_folder / shared_params["ds_symlink_name"])
    model_quickscorere = str(preproc_folder / shared_params["quickscorer_model_file_name"])
    quickscorer_args = [exe_quickscorer,
                        "-d", ds_quickscorer,
                        "-m", model_quickscorere,
                        "-t", "0",
                        "-l", str(max_n_leaves),
                        "-r", "1",
                        "-s", str(results_quickscorer)]
    quickscorer_args = numa_args + quickscorer_args
    res = subprocess.run(quickscorer_args, capture_output=True)

    if debug:

        print(" ".join(quickscorer_args))
        print(res.stdout.decode('utf-8'))
        try:
            # Check the results of quickscorer with lightgbm only in debug mode
            check_results(results_lgbm, results_quickscorer)
        except AssertionError as e:
            print(f"Results differ between lightgbm and quickscorer: \n {results_lgbm} \n {results_quickscorer}")

    regex_res = re.search("Total scoring time: (\d*.\d*) s.", res.stdout.decode('utf-8'))
    elapsed_time = float(regex_res.groups()[0]) * 1000  # elapsed time in ms, quickscorer reports in seconds
    return elapsed_time

######################################################
# Collect data
######################################################
def run_perf_experiment(preproc_folder: str,
                        shared_params: dict,
                        experiment_param: dict,
                        exp_name: str,
                        numa: int | None,
                        debug: bool = False) -> list[RunEntry]:
    preproc_folder = Path(preproc_folder) / experiment_param["name"]

    lgbm_model = preproc_folder / shared_params["lgbm_model_symlink_name"]

    max_n_leaves = get_max_n_leaves(lgbm_model)

    entries = []

    numa_args = []
    if numa is not None:
        numa_args = ["numactl", "-C", f"{numa}", f"--localalloc"]

    for run_id in tqdm(range(shared_params["reps"]),
                       desc=f"Running {exp_name} {shared_params['reps']} times",
                       leave=False):

        ms_elapsed, results_lgbm = run_lightgbm(preproc_folder, shared_params, numa_args, debug)
        entries.append(RunEntry("lightgbm", run_id, ms_elapsed, exp_name))

        ms_elapsed = run_ilmart(preproc_folder, shared_params, numa_args, results_lgbm, debug)
        entries.append(RunEntry("ilmart", run_id, ms_elapsed, exp_name))

        # Only run quickscorer if the number of leaves is less than 64
        if max_n_leaves <= 64:
            elapsed_time = run_quickscorer(preproc_folder, shared_params, max_n_leaves, numa_args, results_lgbm, debug)
            entries.append(RunEntry("quickscorer", run_id, elapsed_time, exp_name))

        if debug:
            print(entries)

    if debug:
        print(entries)

    return entries




def main():
    parser = argparse.ArgumentParser(description="Scoring time", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--preproc_folder", type=str,
                        help="Path to the folder with all the preprocessed data", default="~/data/perf/preproc")
    parser.add_argument("--config_file", type=str, help="Config file for the experiment", default="config.json")
    parser.add_argument("--out_file", type=str, help="Output_file", default="~/data/perf/eval.csv")
    parser.add_argument("--numa_core", type=int, help="NUMA core to run the experiments", default=None)
    parser.add_argument("--debug", help="Debug the script", action="store_true", default=False)
    args = parser.parse_args()

    results = []

    prepoc_folder = Path(args.preproc_folder)
    prepoc_folder = prepoc_folder.expanduser()
    prepoc_folder = str(prepoc_folder)

    out_file = Path(args.out_file)
    out_file = out_file.expanduser()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file = str(out_file)

    with open(args.config_file, "r") as f:
        config = json.load(f)
        shared_params = config["shared_params"]

        for exp in tqdm(config["experiments"]):
            results += run_perf_experiment(prepoc_folder,
                                           shared_params,
                                           exp,
                                           exp_name=exp["name"],
                                           debug=args.debug,
                                           numa=args.numa_core)

    pd.DataFrame(results).to_csv(out_file, index=False)


if __name__ == '__main__':
    main()
