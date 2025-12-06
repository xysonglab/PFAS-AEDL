"""Combine all confidence files from regression predictions
Focus on RMSE-based regression analysis only
"""

import os
import numpy as np
import argparse
import json
import pandas as pd
from tqdm import tqdm
import scipy.stats as stats

CONF_ORDERING = "sets_by_error"
SUMMARY_LOC = "fold_0"

def get_args():
    """Get command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", action="store",
                        help="""Directory containing different datasets of
                        results. E.g. dataset_dir/lipo/method/trial/conf.txt
                        and dataset_dir/logs_freesolv/method/trial#/conf.txt
                        should both be accessible""")
    parser.add_argument("--results-name", action="store",
                        default="conf.txt",
                        help="Name of files to actually merge")
    parser.add_argument("--summary-names", action="store",
                        default=[], nargs="+",
                        help="""List of all summary file names that should
                        appear. For example: --summary-names spearman.txt
                        log_likelihood.txt""")
    parser.add_argument("--outfile", action="store",
                        help="Prefix of all outfiles to be stored",
                        default="consolidated")
    parser.add_argument("--ood-test", action="store_true",
                        help="If true, collect ood test split info",
                        default=False)
    parser.add_argument("--result-type", action="store",
                        default="low_n",
                        help=("Type of experiments being combined. This will"
                              "change how to compute the summary df marginally"
                              "by setting the binning"),
                        choices=["atomistic", "low_n", "high_n", "tdc"])

    args = parser.parse_args()
    return args


def extract_subdirs(path):
    """Extract both the subdir names and full paths; exclude .files"""

    # Avoid collecting .DS_Store and hidden files
    dataset_names = [i for i in os.listdir(path)
                     if i[0] != "." and os.path.isdir(os.path.join(path, i))]
    dataset_dirs = [os.path.join(path, i) for i in dataset_names]

    return dataset_names, dataset_dirs


def extract_summary(summary_file, extra_info):
    """Get dict from summary file

    Args:
        summary_file: name of summary file
        extra_info: dict of extra info to add to each entry

    Return:
        new dict
    """
    temp_results = json.load(open(summary_file, "r"))
    temp_results.update(extra_info)
    return temp_results


def extract_yaml_info(conf_file, extra_info):
    """Make one single list of dicts from the stored json.

    Args:
        conf_file: name of conf file
        extra_info: dict of extra info to add to each entry

    Return:
        list of new dicts
    """
    output_list = []
    temp_results = json.load(open(conf_file, "r"))

    # Iterate through "test" and "val"
    for data_split in temp_results:

        # Now get each subdataset
        for task_name in temp_results[data_split]:
            # Note: Each dataset actually has dataset name in it. Save this as well
            # Hard code selection of the ordering
            for entry in temp_results[data_split][task_name][CONF_ORDERING]:
                entry['partition'] = data_split
                entry['task_name'] = task_name
                # Add extra info
                entry.update(extra_info)
                output_list.append(entry)

    return output_list


############
### Regression RMSE functions
############

def ordered_rmse(df_subset, sort_factor, skip_factor=1):
    """Order df_subset by sort_factor and compute RMSE at each cutoff

    Args:
        df_subset: DataFrame subset to analyze
        sort_factor: Column name to sort by (typically 'confidence')
        skip_factor: Step size for sampling cutoff points

    Returns:
        Array of RMSE values at different confidence cutoffs
    """
    data = df_subset.to_dict('records')
    sorted_data = sorted(data,
                         key=lambda pair: pair[sort_factor],
                         reverse=True)

    cutoff, errors = [], []

    # Compute squared errors
    error_list = [set_['error']**2 for set_ in sorted_data]
    total_error = np.sum(error_list)

    for i in tqdm(range(0, len(error_list), skip_factor),
                  desc="Computing RMSE"):
        cutoff.append(sorted_data[i][sort_factor])
        errors.append(np.sqrt(total_error / len(error_list[i:])))
        total_error -= np.sum(error_list[i:i+skip_factor])

    return np.array(errors)


def regr_calibration_fn(df_subset, num_partitions=10):
    """Create regression calibration curves in the observed bins.

    Full explanation and code adapted from:
    https://github.com/uncertainty-toolbox/uncertainty-toolbox/blob/89c42138d3028c8573a1a007ea8bef80ad2ed8e6/uncertainty_toolbox/metrics_calibration.py#L182

    Args:
        df_subset: DataFrame subset for a specific experiment
        num_partitions: Number of probability bins

    Returns:
        Observed proportions at each expected probability level
    """
    expected_p = np.arange(num_partitions + 1) / num_partitions

    df_subset = df_subset.query('partition == "test"')
    data = df_subset.to_dict('records')
    predictions = np.array([i['prediction'] for i in data])
    confidence = np.array([i['confidence'] for i in data])
    targets = np.array([i['target'] for i in data])

    # Compute calibration using Gaussian assumption
    norm = stats.norm(loc=0, scale=1)
    gaussian_lower_bound = norm.ppf(0.5 - expected_p / 2.0)
    gaussian_upper_bound = norm.ppf(0.5 + expected_p / 2.0)

    residuals = predictions - targets
    normalized_residuals = (residuals.flatten() / confidence.flatten()).reshape(-1, 1)

    above_lower = normalized_residuals >= gaussian_lower_bound
    below_upper = normalized_residuals <= gaussian_upper_bound
    within_quantile = above_lower * below_upper

    obs_proportions = np.sum(within_quantile, axis=0).flatten() / len(residuals)

    return obs_proportions


def make_summary_df(df, summary_functions, summary_names):
    """Convert the full_df object into a summary df.

    Args:
        df: Full dataframe of all experiments
        summary_functions: Functions to be applied to each experiment run
        summary_names: Names of outputs in df for the summary functions

    Returns:
        Summary dataframe with computed metrics
    """
    df = df.query("partition == 'test'")

    # Group by experiment parameters
    subsetted = df.groupby(["dataset", "method_name", "trial_number",
                            "task_name"])

    merge_list = []
    for name, fn in tqdm(zip(summary_names, summary_functions),
                         total=len(summary_names),
                         desc="Computing summary metrics"):
        merge_list.append(subsetted.apply(fn).to_frame(name=name))

    summary_df = pd.concat(merge_list, axis=1).reset_index()

    return summary_df


def convert_to_std(full_df):
    """Convert confidence to std where applicable

    Args:
        full_df: DataFrame with 'confidence' and 'stds' columns
    """
    # Convert all the confidence values into standard deviations
    new_confidence = full_df["stds"].values

    # At all the locations that *don't* have an std value,
    # replace with the value in confidence column
    std_na = pd.isna(new_confidence)
    new_confidence[std_na] = full_df['confidence'][std_na].values
    full_df["confidence"] = new_confidence


def create_regression_summary(full_df, skip_factor=1, num_partitions=40):
    """Given the regression full df as input, create a smaller summary by
    collecting data across trials

    Args:
        full_df: Full dataframe with all regression predictions
        skip_factor: Binning factor for cutoff curves
        num_partitions: Number of partitions for calibration curves

    Returns:
        tuple: (summary_names, summary_df)
    """
    # Create cutoff RMSE and calibration plots
    summary_names = ["rmse", "Predicted Probability", "Expected Probability"]

    # Define lambda functions for each metric
    cutoff_rmse = lambda x: ordered_rmse(x, "confidence", skip_factor=skip_factor)
    calibration_fn_ = lambda x: regr_calibration_fn(x, num_partitions=num_partitions)
    expected_p = lambda x: np.arange(num_partitions + 1) / num_partitions

    summary_fns = [cutoff_rmse, calibration_fn_, expected_p]
    summary_df = make_summary_df(full_df, summary_fns, summary_names)

    # Fill NaN values with 0
    summary_df.fillna(0, inplace=True)

    return summary_names, summary_df


############
### Main execution
############

def main(saved_dir, data_file_name, outfile, summary_file_names,
         result_type, ood_test=False):
    """Main function to combine and analyze regression results

    Args:
        saved_dir: Root directory containing all results
        data_file_name: Name of confidence files to merge
        outfile: Output file prefix
        summary_file_names: List of summary file names to extract
        result_type: Type of experiment (atomistic, low_n, high_n, tdc)
        ood_test: Whether to collect out-of-distribution test information
    """
    # Initialize data containers
    full_data = []
    summary_data = []

    dataset_names, dataset_dirs = extract_subdirs(saved_dir)

    # Loop over different datasets
    for dataset_dir, dataset_name in zip(dataset_dirs, dataset_names):
        print(f"\nProcessing dataset: {dataset_name}")

        # Loop over all different methods in each dataset
        method_names, method_dirs = extract_subdirs(dataset_dir)

        # Loop over methods
        for method_dir, method_name in zip(method_dirs, method_names):
            print(f"  Method: {method_name}")

            # Loop over all trials in each directory
            trial_names, trial_dirs = extract_subdirs(method_dir)

            for trial_number, (trial_dir, trial_name) in tqdm(
                enumerate(zip(trial_dirs, trial_names)),
                desc=f"  Processing trials",
                total=len(trial_names)):

                extra_info = {
                    "method_name": method_name,
                    "trial_number": trial_number,
                    "dataset": dataset_name
                }

                # EXTRACT CONFIDENCE DATA
                results_file = os.path.join(trial_dir, data_file_name)
                if os.path.isfile(results_file):
                    new_data = extract_yaml_info(results_file, extra_info)

                    # Handle OOD test information if requested
                    if ood_test:
                        ood_info = os.path.join(trial_dir, "fold_0/ood_info.csv")

                        if not os.path.isfile(ood_info):
                            raise ValueError(f"OOD info file not found: {ood_info}")

                        ood_df = pd.read_csv(ood_info, index_col=0)

                        smi = ood_df["smiles"].values
                        part = ood_df["partition"]
                        tani_sim = ood_df["max_sim_to_train"]

                        smi_to_part = dict(zip(smi, part))
                        smi_to_tani = dict(zip(smi, tani_sim))

                        for new_entry in new_data:
                            smi = new_entry['smiles']
                            ood_part = smi_to_part.get(smi, None)
                            tani = smi_to_tani.get(smi, None)
                            new_entry.update({
                                'ood_partition': ood_part,
                                'closest_sim': tani
                            })

                    full_data.extend(new_data)

                # EXTRACT SUMMARY FILES
                summary_dir = os.path.join(trial_dir, SUMMARY_LOC)
                current_summary = {}

                for summary_file_name in summary_file_names:
                    summary_file = os.path.join(summary_dir, summary_file_name)
                    if os.path.isfile(summary_file):
                        current_summary.update(extract_summary(summary_file, extra_info))

                summary_data.append(current_summary)

    # Save full dataframe
    print(f"\nSaving full results to {outfile}.tsv")
    df_full = pd.DataFrame(full_data)
    df_full.to_csv(f"{outfile}.tsv", sep="\t")

    # Save summary dataframe
    print(f"Saving summary to {outfile}_summary.tsv")
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(f"{outfile}_summary.tsv", sep="\t")

    # Only select test set for detailed analysis
    full_df = df_full.query("partition == 'test'").reset_index()

    ### Compute summary dataframe based on result type
    print(f"\nComputing detailed summary for result_type: {result_type}")

    # Convert confidence values to standard deviations
    convert_to_std(full_df)

    # Set skip_factor based on result type
    if result_type == "atomistic":
        skip_factor = 30
    elif result_type == "high_n":
        skip_factor = 30
    elif result_type in ["low_n", "tdc"]:
        skip_factor = 1
    else:
        raise ValueError(f"Unknown result_type: {result_type}")

    # Create regression summary
    summary_names, summary_df = create_regression_summary(
        full_df,
        skip_factor=skip_factor
    )

    # Save calculated summary
    print(f"Saving calculated summary to {outfile}_summary_calc.tsv")
    summary_df.to_csv(f"{outfile}_summary_calc.tsv", sep="\t")

    print("\nProcessing complete!")
    print(f"  Total samples: {len(df_full)}")
    print(f"  Test samples: {len(full_df)}")
    print(f"  Summary experiments: {len(summary_df)}")


if __name__ == "__main__":
    args = get_args()
    saved_dir = args.dataset_dir
    data_file_name = args.results_name
    outfile = args.outfile
    summary_file_names = args.summary_names

    main(saved_dir, data_file_name, outfile, summary_file_names,
         args.result_type, args.ood_test)