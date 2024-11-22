# Preprocess the dataset to be used in the experiments with lightgbm C++ and ilmart C++
import argparse
import json
from pathlib import Path
import os

from rankeval.dataset.dataset import Dataset as RankEvalDataset
from rankeval.model import RTEnsemble
from tqdm import tqdm

from ilmart import IlmartDistill
from ilmart.utils import DICT_NAME_FOLDER
import lightgbm as lgbm


def create_exp_folder(shared_params: dict, experiment_param: dict, output_folder: str):
    output_folder = Path(output_folder) / experiment_param["name"]
    output_folder.mkdir(parents=True, exist_ok=True)

    # Set shared params
    lgbm_path = Path(shared_params["lightgbm_path"])  # Path to the lightgbm executable
    ilmart_distilled_path = Path(shared_params["ilmart_distilled_path"])  # Path to the ilmart executable
    quickscorer_path = Path(shared_params['quickscorer_path']) # Path to the quickscorer executable

    # Here we admit the usage of tildes in the paths
    lgbm_path = lgbm_path.expanduser().absolute()
    ilmart_distilled_path = ilmart_distilled_path.expanduser().absolute()
    quickscorer_path = quickscorer_path.expanduser().absolute()

    # Create symlinks to the executables
    lgbm_symlink_out = output_folder / shared_params['lgbm_symlink_name']
    lgbm_symlink_out = lgbm_symlink_out.absolute()
    print(f"Creating symlink to {lgbm_path} in {lgbm_symlink_out}")
    os.symlink(lgbm_path, lgbm_symlink_out)

    ilmart_symlink_out = output_folder / shared_params['ilmart_symlink_name']
    ilmart_symlink_out = ilmart_symlink_out.absolute()
    print(f"Creating symlink to {ilmart_distilled_path} in {ilmart_symlink_out}")
    os.symlink(ilmart_distilled_path, output_folder / shared_params["ilmart_symlink_name"])

    quickscorer_symlink = output_folder / shared_params['quickscorer_symlink_name']
    quickscorer_symlink = quickscorer_symlink.absolute()
    print(f"Creating symlink to {quickscorer_path} in {quickscorer_symlink}")
    os.symlink(quickscorer_path, quickscorer_symlink)

    # Set experiments specific parameters
    split = experiment_param["split"]
    dataset_name = experiment_param["dataset"]
    rankeval_dataset_path = Path(DICT_NAME_FOLDER[dataset_name]) / f"{split}.txt"
    rankeval_dataset_path = rankeval_dataset_path.absolute()
    lgbm_model_path = Path(experiment_param["lgbm_model_path"])

    print(f"Loading dataset from {rankeval_dataset_path}")
    rankeval_dataset = RankEvalDataset.load(str(rankeval_dataset_path))

    # Create symlinks to the dataset and the model
    lgbm_model_symlink = output_folder / shared_params['lgbm_model_symlink_name']
    lgbm_model_symlink = lgbm_model_symlink.absolute()
    print(f"Creating symlink to {lgbm_model_path} in {lgbm_model_symlink}")
    os.symlink(lgbm_model_path, lgbm_model_symlink)

    rankeval_symlink = output_folder / shared_params['ds_symlink_name']
    rankeval_symlink = rankeval_symlink.absolute()
    print(f"Creating symlink to {rankeval_dataset_path} in {rankeval_symlink}")
    os.symlink(rankeval_dataset_path, rankeval_symlink)

    # Create the files for lightgbm

    # First create a copy of the dataset but 0-indexed to avoid problem with lightgbm inference
    with open(rankeval_dataset_path, "r") as fr:
        ds_out_folder = output_folder / f"{shared_params['ds_symlink_name_lgbm']}"
        with open(ds_out_folder, "w") as fw:
            for line in tqdm(fr,
                             desc=f"Converting dataset and save it to {ds_out_folder}",
                             total=rankeval_dataset.y.shape[0]):
                line_splits = line.split()
                features = line_splits[2:]
                fw.write(f"{line_splits[0]} ")
                for feat_value in features:
                    feat, value = feat_value.split(":")
                    feat = str(int(feat) - 1)
                    fw.write(f"{feat}:{value} ")
                fw.write("\n")

    # Then create the query file
    query_file = ds_out_folder.parent / (ds_out_folder.name + '.query')
    print(f"Creating query file to {query_file}")
    query_sizes = rankeval_dataset.get_query_sizes()
    with open(query_file, "w") as f:
        for query_size in query_sizes:
            f.write(f"{query_size}\n")

    absolute_path_to_data = (output_folder / shared_params["ds_symlink_name_lgbm"]).absolute()
    absolute_path_to_model = (output_folder / shared_params["lgbm_model_symlink_name"]).absolute()

    # Create the config file for lightgbm predictions
    config_file = output_folder / shared_params['prediction_config_lgbm']
    print(f"Creating config file to {config_file}")
    with open(config_file, "w") as f:
        f.write(f"task = predict\n")
        f.write(f"data = {absolute_path_to_data}\n")
        f.write(f"input_model= {absolute_path_to_model}\n")

        absolute_path_lgbm_output = (output_folder / shared_params['result_lightgbm']).absolute()
        f.write(f"output_result = {absolute_path_lgbm_output}\n")
        f.write(f"num_threads = 1\n")
        f.write(f"verbosity = 2\n")

    # Create the config file for ilmart

    print(f"Distilling the model {absolute_path_to_model}")
    ilmart_distilled_model = IlmartDistill(lgbm.Booster(model_file=absolute_path_to_model))

    ilmart_model_export_path = output_folder / shared_params['ilmart_model_file_name']
    print(f"Exporting the distilled model to {ilmart_model_export_path}")
    ilmart_distilled_model.export(ilmart_model_export_path)

    # Create the files for quickscorer

    quickscorer_model_path = output_folder / shared_params["quickscorer_model_file_name"]
    print(f"Converting the model to QuickRank format and saving it in {quickscorer_model_path}")
    model_rankeval_format = RTEnsemble(file_path=str(absolute_path_to_model), format="LightGBM")
    model_rankeval_format.save(quickscorer_model_path, format="QuickRank")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess the dataset to be used in the experiments with lightgbm C++ and ilmart C++.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--config_file", type=str, help="Config file for the experiment", default="config.json")
    parser.add_argument("--out_folder", type=str, help="Output folder", default="~/data/perf/preproc")
    args = parser.parse_args()

    with open(args.config_file, "r") as f:
        config = json.load(f)
        shared_params = config["shared_params"]

        out_folder = Path(args.out_folder)
        out_folder = out_folder.expanduser()
        out_folder.mkdir(parents=True, exist_ok=True)
        out_folder = str(out_folder)

        for exp in tqdm(config["experiments"]):
            create_exp_folder(shared_params, exp, out_folder)


if __name__ == '__main__':
    main()
