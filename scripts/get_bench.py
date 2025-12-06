import os
import shutil
import argparse
import subprocess
import typing
from typing import List, Optional, Tuple
import time
from datetime import datetime
import json

def get_args() -> Tuple[dict, str]:
    """ get_args.

    Return:
        Tuple[dict,str]: Args and name of file used
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="Name of configuration file")
    args = parser.parse_args()
    print(f"Loading experiment from: {args.config_file}\n")
    args_new = json.load(open(args.config_file, "r"))
    return args_new, args.config_file

def dump_config_file(save_dir : str, config : str):
    """ dump_config_file.

    Try to dump the output config file continuously. If it doesn't work,
    increment it.

    Args:
        save_dir (str): Name of the save dir where to put this
        config (str): Location of the config file
    """

    # Dump experiment
    new_file = "experiment.config"
    config_path = os.path.join(save_dir, new_file)
    ctr = 1
    os.makedirs(save_dir, exist_ok=True)
    # Keep incrementing the counter
    while os.path.exists(config_path):
        new_file =  f"experiment_{ctr}.config"
        config_path = os.path.join(save_dir, new_file)
        ctr += 1

    shutil.copy2(config, config_path)
    TIME_STAMP_DATE = datetime.now().strftime("%m_%d_%y")

    # Open timestamps file
    with open(os.path.join(save_dir, "timestamps.txt"), "a") as fp:
          fp.write(f"Experiment {new_file} run on {TIME_STAMP_DATE}.\n")

def main(seeds : List[int] = range(10), n_ensembles : int = 5,
         datasets : List[str] = ["SI-2_KAW"], dataset_types = ["regression"] ,
         dataset_splits : List[str] = ["random"],
         methods : List[str] = ["evidence_new_reg"],
         reg_coefs : List[float] = [0.2],
         split_sizes: List[float] = [0.8,0.1,0.1],
         save_dir : str = "results/pfas",
         dropout : float = 0.1,
         epochs: float = 40,
         ensemble_threads: int = 2,
         debug : bool = False,
         use_gpu: bool = True,
         experiment_file_name: Optional[str]= None,
         no_smiles_export : bool = False):

    # Verify the length constraints
    assert (len(datasets) == len(dataset_types))
    assert (len(datasets) == len(dataset_splits))
    assert (len(dataset_types) == len(dataset_splits))
    assert (len(methods) == len(reg_coefs))
    assert (len(split_sizes) == 3)

    # Dump out file
    if experiment_file_name:
        dump_config_file(save_dir, experiment_file_name)

    BASE_ARGS = "--save_confidence conf.txt --use_entropy --confidence_evaluation_methods cutoff"

    if no_smiles_export:
        BASE_ARGS = f"{BASE_ARGS} --no_smiles_export"

    # GPU device argument
    GPU_ARGS = ""
    if use_gpu:
        GPU_ARGS = "--gpu 0"

    for trial in seeds:
        seed = trial
        for dataset, dataset_type, split_type in zip(datasets, dataset_types, dataset_splits):

            save_dir_ = os.path.join(f"{save_dir}", split_type)
            for coeff, method in zip(reg_coefs, methods):
                METHOD_ARGS = ""
                method_name = method
                if method == "ensemble":
                    METHOD_ARGS=f"--ensemble_size {n_ensembles} --threads {ensemble_threads}"
                elif method == "dropout":
                    METHOD_ARGS=f"--ensemble_size {n_ensembles} --dropout {dropout} --no_dropout_inference"
                elif method == "evidence":
                    raise ValueError("Should only see new evidence with evidence_new_reg")
                elif method == "evidence_new":
                    raise ValueError("Should only see new evidence with evidence_new_reg")
                    METHOD_ARGS=f"--new_loss"
                    method_name= "evidence"
                elif method == "evidence_new_reg":
                    METHOD_ARGS=f"--new_loss --regularizer_coeff {coeff}"
                    method_name= "evidence"
                    # Rename method to have a name for the coefficient
                    method = f"{method}_{coeff}"
                else:
                    pass

                SPLIT_ARGS=f"--split_type {split_type} --split_sizes {split_sizes[0]} {split_sizes[1]} {split_sizes[2]}"

                # Add train split here
                LOG_ARGS = f"--save_dir {save_dir_}/{dataset}/{method}"

                # Set dataset args
                if dataset == "freesolv":
                    EPOCH_ARGS=f"--epochs {epochs}"
                elif dataset=="qm9":
                    EPOCH_ARGS=f"--epochs {epochs} --metric mae"
                elif dataset=="qm7":
                    EPOCH_ARGS=f"--epochs {epochs} --metric rmse"
                elif dataset=="delaney":
                    EPOCH_ARGS= f"--epochs {epochs}"
                else:
                    EPOCH_ARGS=f"--epochs {epochs}"

                if debug:
                    EPOCH_ARGS="--epochs 3"

                python_string=f"python train.py --confidence {method_name} {EPOCH_ARGS} {METHOD_ARGS} {LOG_ARGS} {BASE_ARGS} {SPLIT_ARGS} {GPU_ARGS} --seed {trial} --dataset_type {dataset_type} --data_path data/{dataset}.csv"

                if debug:
                    python_string = f"{python_string} --debug"

                print(f"{python_string}")
                subprocess.call(python_string, shell=True)

                time.sleep(3)

if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)
    args, exp_file  = get_args()
    main(experiment_file_name = exp_file, **args)