import argparse
import json
import re
from collections import namedtuple
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from ilmart import IlmartDistill
from ilmart.utils import load_datasets, DICT_NAME_FOLDER
import lightgbm as lgbm
import subprocess

RunEntry = namedtuple("RunEntry", ["model", "run_id", "time", "exp_name"])


def check_results(results_lgbm: Path, results_ilmart: Path):
    res_lgbm = pd.read_csv(results_lgbm, header=None).values
    res_ilmart = pd.read_csv(results_ilmart, header=None).values

    assert res_lgbm.shape[0] == res_ilmart.shape[0]

    assert len(np.where(res_lgbm - res_ilmart > 0.0001)[0]) == 0


def run_perf_experiment(preproc_folder: str, shared_params: dict, experiment_param: dict, exp_name: str,
                        debug=False) -> list[
    RunEntry]:
    preproc_folder = Path(preproc_folder) / experiment_param["name"]
    lgbm_symlink = preproc_folder / shared_params['lgbm_symlink_name']
    lgbm_config = preproc_folder / shared_params['prediction_config_lgbm']

    entries = []

    for run_id in tqdm(range(shared_params["reps"]),
                       desc=f"Running {exp_name} {shared_params['reps']} times",
                       leave=False):

        results_lgbm = preproc_folder / shared_params["result_lgbm"]
        res = subprocess.run([str(lgbm_symlink), f"config={lgbm_config}"], capture_output=True)

        # The end of output of the command is something like:
        # [LightGBM] [Info] Finished prediction in 272060.026542 milliseconds
        if debug:
            print(res.stdout.decode('utf-8'))
        ms_elapsed = float(res.stdout.decode('utf-8').split()[-2])
        entries.append(RunEntry("lightgbm", run_id, ms_elapsed, exp_name))

        results_ilmart = preproc_folder / shared_params["result_ilmart"]
        res = subprocess.run([str(preproc_folder / shared_params["ilmart_symlink_name"]),
                              str(preproc_folder / shared_params["ilmart_model_file_name"]),
                              str(preproc_folder / shared_params["ds_symlink_name_lgbm"]),
                              str(results_ilmart),
                              str(1)],
                             capture_output=True)
        if debug:
            print(res.stdout.decode('utf-8'))
        ms_elapsed = float(res.stdout.decode('utf-8').split()[-1])
        entries.append(RunEntry("ilmart", run_id, ms_elapsed, exp_name))

        results_quickscorer = preproc_folder / shared_params["result_quickscorer"]
        res = subprocess.run(
            [str(preproc_folder / shared_params["quickscorer_symlink_name"]),
             "-d", str(preproc_folder / shared_params["ds_symlink_name"]),
             "-m", str(preproc_folder / shared_params["quickscorer_model_file_name"]),
             "-t", "0",
             "-l", "63",
             "-r", "1",
             "-s", str(results_quickscorer)], capture_output=True)
        if debug:
            print(res.stdout.decode('utf-8'))
        regex_res = re.search("Total scoring time: (\d*.\d*) s.", res.stdout.decode('utf-8'))
        elapsed_time = float(regex_res.groups()[0]) * 1000  # elapsed time in ms, quickscorer reports in seconds
        entries.append(RunEntry("quickscorer", run_id, elapsed_time, exp_name))

        if debug:
            print(entries)

        check_results(results_lgbm, results_ilmart)

    if debug:
        print(entries)

    return entries


def main():
    parser = argparse.ArgumentParser(description="Scoring time", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--preproc_folder", type=str,
                        help="Path to the folder with all the preprocessed data", default="/data/perf/preproc")
    parser.add_argument("--config_file", type=str, help="Config file for the experiment", default="config.json")
    parser.add_argument("--out_file", type=str, help="Output_file", default="/data/perf/eval.csv")
    parser.add_argument("--debug", type=bool, help="Debug the script", default=False)
    args = parser.parse_args()

    results = []
    with open(args.config_file, "r") as f:
        config = json.load(f)
        shared_params = config["shared_params"]

        for exp in tqdm(config["experiments"]):
            results += run_perf_experiment(args.preproc_folder, shared_params, exp, exp_name=exp["name"])

    pd.DataFrame(results).to_csv(args.out_file, index=False)


if __name__ == '__main__':
    main()
