"""
make_figs.py
Modified version - Ensures all plots display axis ticks and labels, and removes all grids
Retains only regression-related functionality
"""

import os
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import pandas as pd
import numpy as np
import argparse
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
import scipy.stats as stats
from scipy.interpolate import interp1d
from scipy.integrate import simps
from sklearn import metrics
from tqdm import tqdm, trange
from pathos import multiprocessing

from numpy import nan
from ast import literal_eval


METHOD_ORDER = ["evidence", "dropout", "ensemble", "sigmoid"]

METHOD_COLORS = {
    method: sns.color_palette()[index]
    for index, method in enumerate(METHOD_ORDER)
}

# Define fresh color palette for pfas plots
FRESH_COLORS = {
    "evidence": "#66C2A5",  # Teal
    "dropout": "#FC8D62",   # Coral
    "ensemble": "#8DA0CB",  # Light blue
    "sigmoid": "#E78AC3"    # Pink
}

DATASET_MAPPING = {
    "SI-2_KAW": r"$\mathrm{log}\ K_{\mathrm{AW}}$",
    "SI-2_KOW": r"$\mathrm{log}\ K_{\mathrm{OW}}$",
    "SI-2_KOA": r"$\mathrm{log}\ K_{\mathrm{OA}}$",
    "SI-2_KOC": r"$\mathrm{log}\ K_{\mathrm{OC}}$",
    "SI-2_W": r"$\mathrm{log}\ S_{\mathrm{W}}$",
}

DATASETS = DATASET_MAPPING.values()

REGR_SUMMARY_NAMES = [
    "rmse", "mae", "Predicted Probability",
    "Expected Probability"
]


def setup_axes(ax=None, show_x_ticks=True, show_y_ticks=True):
    """Setup axis style, control which borders show ticks

    Parameters:
    -----------
    ax : matplotlib axis object, optional
        The axis object to set up
    show_x_ticks : bool, default=True
        Whether to show x-axis (bottom) ticks and labels
    show_y_ticks : bool, default=True
        Whether to show y-axis (left) ticks and labels
    """
    if ax is None:
        ax = plt.gca()

    # Show all borders
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)

    # Set bottom and left ticks (with numbers)
    ax.tick_params(axis='x', which='both',
                   bottom=show_x_ticks, top=False,
                   labelbottom=show_x_ticks, labeltop=False,
                   length=5 if show_x_ticks else 0, width=1,
                   direction='in')

    ax.tick_params(axis='y', which='both',
                   left=show_y_ticks, right=False,
                   labelleft=show_y_ticks, labelright=False,
                   length=5 if show_y_ticks else 0, width=1,
                   direction='in')

    # Set font size and font
    ax.tick_params(labelsize=26)

    # Set axis label font
    if ax.get_xlabel():
        ax.set_xlabel(ax.get_xlabel(), fontsize=30, fontname='Arial', fontweight='bold')
    if ax.get_ylabel():
        ax.set_ylabel(ax.get_ylabel(), fontsize=30, fontname='Arial', fontweight='bold')

    # Set title font
    if ax.get_title():
        ax.set_title(ax.get_title(), fontsize=30, fontname='Arial', fontweight='bold')

    # Remove grid
    ax.grid(False)

    return ax


def rename_method_df_none(df_column, rename_key):
    """Transform method names"""
    return [rename_key.get(i, None) for i in df_column]


def convert_dataset_names(dataset_series):
    """Convert the dataset series into the desired labeling"""
    ret_ar = []
    for i in dataset_series:
        ret_ar.append(DATASET_MAPPING.get(i, i))
    return ret_ar


def convert_to_std(full_df):
    """Convert confidence to std where applicable"""
    new_confidence = full_df["stds"].values
    std_na = pd.isna(new_confidence)
    new_confidence[std_na] = full_df['confidence'][std_na].values
    full_df["confidence"] = new_confidence


def make_cutoff_table(df, outdir="results/figures", out_name="cutoff_table.txt",
                      export_stds=True, output_metric="rmse",
                      table_data_name="D-MPNN (RMSE)",
                      significant_best=True,
                      higher_better=False):
    """Create the output latex results table and save in a text file."""

    top_k = np.array([1, 0.5, 0.25, 0.1, 0.05])[::-1]

    df = df.copy()
    df["Method"] = df["method_name"]
    df["Data"] = convert_dataset_names(df["dataset"])

    uniq_methods = set(df["Method"].values)
    unique_datasets = set(df["Data"].values)

    data_order = [j for j in DATASETS if j in unique_datasets]
    method_order = [j for j in METHOD_ORDER if j in uniq_methods]

    table_items = []
    for data_group, data_df in df.groupby(["Data"]):
        for data_method, data_method_df in data_df.groupby(["Method"]):
            if data_method.lower() not in [method.lower() for method in method_order]:
                continue
            metric_sub = data_method_df[output_metric]
            num_tested = np.min([len(i) for i in metric_sub])

            for cutoff in top_k:
                num_items = int(cutoff * num_tested)
                temp = np.reshape([j[-num_items] for j in metric_sub], -1)
                metric_cutoff_mean = np.mean(temp)
                metric_cutoff_std = stats.sem(temp)

                table_items.append({
                    "Data": data_group,
                    "Method": data_method,
                    "Cutoff": cutoff,
                    "METRIC_MEAN": metric_cutoff_mean,
                    "METRIC_STD": metric_cutoff_std
                })

    metric_summary = pd.DataFrame(table_items)
    means_tbl = pd.pivot_table(metric_summary,
                                values="METRIC_MEAN",
                                columns=["Cutoff"],
                                index=["Data", "Method"])

    stds_tbl = pd.pivot_table(metric_summary,
                               values="METRIC_STD",
                               columns=["Cutoff"],
                               index=["Data", "Method"])

    means_tbl = means_tbl.reindex(sorted(means_tbl.columns)[::-1], axis=1)
    stds_tbl = stds_tbl.reindex(sorted(stds_tbl.columns)[::-1], axis=1)

    output_tbl = means_tbl.astype(str)
    for cutoff in means_tbl.keys():
        cutoff_means = means_tbl[cutoff]
        cutoff_stds = stds_tbl[cutoff]

        for dataset in data_order:
            means = cutoff_means[dataset].round(5)
            stds = cutoff_stds[dataset]
            str_repr = means.astype(str)

            if export_stds:
                str_repr += " $\\pm$ "
                str_repr += stds.round(5).astype(str)

            if higher_better:
                if significant_best:
                    METRIC_mins = means - stds
                    METRIC_maxs = means + stds
                    highest_metric_min = np.max(METRIC_mins)
                    best_methods = METRIC_maxs > highest_metric_min
                else:
                    best_methods = (means == means.max())
            else:
                if significant_best:
                    METRIC_mins = means - stds
                    METRIC_maxs = means + stds
                    smallest_metric_max = np.min(METRIC_maxs)
                    best_methods = METRIC_mins < smallest_metric_max
                else:
                    best_methods = (means == means.min())

            str_repr[best_methods] = "\\textbf{" + str_repr[best_methods] + "}"
            output_tbl[cutoff][dataset] = str_repr

    output_tbl = output_tbl.reindex(pd.MultiIndex.from_product([data_order, method_order]))

    assert(isinstance(table_data_name, str))
    output_tbl = output_tbl.set_index(pd.MultiIndex.from_product([[table_data_name],
                                                                  data_order,
                                                                  method_order]))
    with open(os.path.join(outdir, out_name), "w") as fp:
        fp.write(output_tbl.to_latex(escape=False))


def average_summary_df_tasks(df, avg_columns):
    """Create averages of the summary df across tasks."""
    new_df = []
    keep_cols = ["dataset", "method_name", "trial_number"]
    subsetted = df.groupby(keep_cols)

    for subset_indices, subset_df in subsetted:
        return_dict = {}
        return_dict.update(dict(zip(keep_cols, subset_indices)))

        for column in avg_columns:
            task_values = subset_df[column].values
            min_length = min([len(i) for i in task_values])

            new_task_values = []
            for j in task_values:
                j = np.array(j)
                if len(j) > min_length:
                    percentiles = np.linspace(0, len(j) - 1, min_length).astype(int)
                    new_task_values.append(j[percentiles])
                else:
                    new_task_values.append(j)
            avg_task = np.mean(np.array(new_task_values), axis=0).tolist()
            return_dict[column] = avg_task

        new_df.append(return_dict)

    return pd.DataFrame(new_df)


def evidence_tuning_plots(df, x_input="Mean Predicted Avg",
                          y_input="Empirical Probability",
                          x_name="Mean Predicted",
                          y_name="Empirical Probability"):
    """Plot the tuning plot at different evidence values"""

    def lineplot(x, y, trials, methods, **kwargs):
        uniq_methods = set(methods.values)
        method_order = sorted(uniq_methods)

        method_new_names = [f"$\lambda={i:0.4f}$" for i in method_order]
        method_df = []
        for method_idx, (method, method_new_name) in enumerate(zip(method_order, method_new_names)):
            lines_y = y[methods == method]
            lines_x = x[methods == method]
            for index, (xx, yy, trial) in enumerate(zip(lines_x, lines_y, trials)):
                to_append = [{x_name: x,
                              y_name: y,
                              "Method": method_new_name,
                              "Trial": trial}
                             for i, (x, y) in enumerate(zip(xx, yy))]
                method_df.extend(to_append)

        method_df = pd.DataFrame(method_df)
        x = np.linspace(0, 1, 100)
        plt.plot(x, x, linestyle='--', color="black")
        sns.lineplot(x=x_name, y=y_name, hue="Method",
                     alpha=0.8,
                     hue_order=method_new_names, data=method_df)

    df = df.copy()
    df = df[["evidence" in i for i in df['method_name']]].reset_index()

    coeff = [float(i.split("evidence_new_reg_")[1]) for i in df['method_name']]
    df["method_name"] = coeff
    df["Data"] = convert_dataset_names(df["dataset"])
    df["Method"] = df["method_name"]

    g = sns.FacetGrid(df, col="Data", height=6, sharex=False, sharey=False)
    g.map(lineplot, x_input, y_input, "trial_number",
          methods=df["Method"]).add_legend()

    # Set up axes for each subplot
    for ax in g.axes.flat:
        setup_axes(ax)


def plot_spearman_r(full_df, std=True, is_pfas=False, use_fresh_colors=False):
    """Plot spearman R summary stats"""

    if std:
        convert_to_std(full_df)
    full_df["Data"] = convert_dataset_names(full_df["dataset"])

    grouped_df = full_df.groupby(["dataset", "method_name", "trial_number", "task_name"])
    spearman_r = grouped_df.apply(lambda x: stats.spearmanr(x['confidence'].values,
                                                            np.abs(x['error'].values)).correlation)

    new_df = spearman_r.reset_index().rename({0: "Spearman Rho"}, axis=1)

    method_order = [i for i in METHOD_ORDER
                    if i in pd.unique(new_df['method_name'])]
    new_df['Method'] = new_df['method_name']

    if is_pfas:
        # Use LaTeX format for subscripts
        dataset_name_mapping = {
            "SI-2_KAW": r"$\mathrm{log}\ K_{\mathrm{AW}}$",
            "SI-2_KOW": r"$\mathrm{log}\ K_{\mathrm{OW}}$",
            "SI-2_KOA": r"$\mathrm{log}\ K_{\mathrm{OA}}$",
            "SI-2_KOC": r"$\mathrm{log}\ K_{\mathrm{OC}}$",
            "SI-2_W": r"$\mathrm{log}\ S_{\mathrm{W}}$"
        }
        new_df['Dataset_Display'] = new_df['dataset'].apply(
            lambda x: dataset_name_mapping.get(x, x)
        )
        x_column = "Dataset_Display"
    else:
        new_df['Dataset'] = new_df['dataset']
        x_column = "Dataset"

    plot_width = 2.6 * len(pd.unique(new_df[x_column]))
    plt.figure(figsize=(plot_width, 5))

    if use_fresh_colors:
        color_palette = FRESH_COLORS
    else:
        color_palette = METHOD_COLORS

    sns.barplot(data=new_df, x=x_column, y="Spearman Rho",
                hue="Method", hue_order=method_order, palette=color_palette)

    # Set up axes
    ax = setup_axes(show_x_ticks=True, show_y_ticks=True)

    if is_pfas:
        plt.xlabel("Dataset", fontsize=24, fontname='Arial', fontweight='bold')

    plt.ylabel("Spearman Rho", fontsize=24, fontname='Arial', fontweight='bold')

    # Use tick_params to set tick labels
    ax.tick_params(axis='x', labelsize=20, labelcolor='black')
    ax.tick_params(axis='y', labelsize=20, labelcolor='black')

    # Set tick label font
    for label in ax.get_xticklabels():
        label.set_fontname('Arial')
        label.set_fontweight('bold')

    for label in ax.get_yticklabels():
        label.set_fontname('Arial')
        label.set_fontweight('bold')

    # Set legend font
    legend = plt.legend()
    for text in legend.get_texts():
        text.set_fontname('Arial')
        text.set_fontsize(18)
        text.set_fontweight('bold')

    plt.tight_layout()

    spearman_r_summary = new_df.groupby(["dataset", "method_name"]).describe()['Spearman Rho'].reset_index()
    return spearman_r_summary


def make_tuning_plot_rmse(df, error_col_name="rmse",
                          error_title="Top 10% RMSE",
                          cutoff=0.10):
    """Create the tuning plot for different lambda evidence parameters"""

    df = df.copy()

    coeff = [float(i.split("evidence_new_reg_")[1]) if "evidence" in i else i for i in df['method_name']]
    df["method_name"] = coeff
    df["Data"] = convert_dataset_names(df["dataset"])
    df["Method"] = df["method_name"]

    trials = 'trial_number'
    methods = 'Method'

    uniq_methods = set(df["Method"].values)
    method_order = sorted(uniq_methods,
                          key=lambda x: x if isinstance(x, float) else -1)
    method_df = []
    datasets = set()

    for data, sub_df in df.groupby("Data"):
        datasets.add(data)
        rmse_sub = sub_df[error_col_name]
        methods_sub = sub_df["Method"]
        trials_sub = sub_df['trial_number']

        for method_idx, method in enumerate(method_order):
            bool_select = (methods_sub == method)
            rmse_method = rmse_sub[bool_select]
            trials_temp = trials_sub[bool_select]
            areas = []

            for trial, rmse_trial in zip(trials_sub, rmse_method):
                num_tested = len(rmse_trial)
                cutoff_index = int(cutoff * num_tested) - 1
                rmse_val = rmse_trial[-cutoff_index]
                to_append = {
                    error_title: rmse_val,
                    "Regularizer Coeff, $\lambda$": method,
                    "method_name": method,
                    "Data": data,
                    "Trial": trial
                }
                method_df.append(to_append)

    method_df = pd.DataFrame(method_df)
    method_df = method_df[[i != "dropout" for i in method_df['method_name']]].reset_index()

    for dataset in datasets:
        division_factor = np.ones(len(method_df))
        indices = (method_df["Data"] == dataset)
        max_val = method_df[indices].query("method_name == 'ensemble'").mean(numeric_only=True)[error_title]
        division_factor[indices] = max_val
        method_df[error_title] = method_df[error_title] / division_factor

    method_df_evidence = method_df[[isinstance(i, float) for i in method_df['method_name']]].reset_index()
    method_df_ensemble = method_df[["ensemble" in str(i) for i in method_df['method_name']]].reset_index()

    data_colors = {
        dataset: sns.color_palette()[index]
        for index, dataset in enumerate(datasets)
    }

    min_x = np.min(method_df_evidence["Regularizer Coeff, $\lambda$"])
    max_x = np.max(method_df_evidence["Regularizer Coeff, $\lambda$"])

    sns.lineplot(x="Regularizer Coeff, $\lambda$", y=error_title,
                 hue="Data", alpha=0.8, data=method_df_evidence,
                 palette=data_colors)

    for data, subdf in method_df_ensemble.groupby("Data"):
        color = data_colors[data]
        area = subdf[error_title].mean()
        std = subdf[error_title].std()
        plt.hlines(area, min_x, max_x, linestyle="--", color=color, alpha=0.8)

    ensemble_line = plt.plot([], [], color='black', linestyle="--", label="Ensemble")

    # Set legend font
    legend = plt.legend(bbox_to_anchor=(1.1, 1.05))
    for text in legend.get_texts():
        text.set_fontname('Arial')
        text.set_fontsize(18)
        text.set_fontweight('bold')

    # Set up axes
    setup_axes()


def make_area_plots(df, x_input="Mean Predicted Avg",
                    y_input="Empirical Probability"):
    """Make evidence tuning plots"""
    plt.figure(figsize=(13, 10))

    df = df.copy()

    coeff = [float(i.split("evidence_new_reg_")[1]) if "evidence" in i else i for i in df['method_name']]
    df["method_name"] = coeff
    df["Data"] = convert_dataset_names(df["dataset"])
    df["Method"] = df["method_name"]

    trials = 'trial_number'
    methods = 'Method'

    uniq_methods = set(df["Method"].values)
    method_order = sorted(uniq_methods,
                          key=lambda x: x if isinstance(x, float) else -1)
    method_df = []
    datasets = set()

    for data, sub_df in df.groupby("Data"):
        datasets.add(data)
        x_vals = sub_df[x_input]
        y_vals = sub_df[y_input]
        methods_sub = sub_df["Method"]
        trials_sub = sub_df['trial_number']

        for method_idx, method in enumerate(method_order):
            bool_select = (methods_sub == method)
            lines_y = y_vals[bool_select]
            lines_x = x_vals[bool_select]
            trials_temp = trials_sub[bool_select]
            areas = []

            for trial, line_x, line_y in zip(trials_sub, lines_x, lines_y):
                new_y = np.abs(np.array(line_y) - np.array(line_x))
                area = simps(new_y, line_x)
                to_append = {
                    "Area from parity": area,
                    "Regularizer Coeff, $\lambda$": method,
                    "method_name": method,
                    "Data": data,
                    "Trial": trial
                }
                method_df.append(to_append)

    method_df = pd.DataFrame(method_df)
    method_df_evidence = method_df[[isinstance(i, float) for i in method_df['method_name']]].reset_index()
    method_df_ensemble = method_df[["ensemble" in str(i) for i in method_df['method_name']]].reset_index()

    data_colors = {
        dataset: sns.color_palette()[index]
        for index, dataset in enumerate(datasets)
    }

    min_x = np.min(method_df_evidence["Regularizer Coeff, $\lambda$"])
    max_x = np.max(method_df_evidence["Regularizer Coeff, $\lambda$"])

    # Create main plot
    sns.lineplot(x="Regularizer Coeff, $\lambda$", y="Area from parity",
                 hue="Data", alpha=0.8, data=method_df_evidence,
                 palette=data_colors)

    # Add horizontal lines
    for data, subdf in method_df_ensemble.groupby("Data"):
        color = data_colors[data]
        area = subdf["Area from parity"].mean()
        std = subdf["Area from parity"].std()
        plt.hlines(area, min_x, max_x, linestyle="--", color=color, alpha=0.8)

    # Create Ensemble legend handle
    ensemble_line = mlines.Line2D([], [], color='black', linestyle="--", label="Ensemble")

    # Get current legend and add Ensemble
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    handles.append(ensemble_line)

    # Set up axes
    setup_axes()

    # Reset axis labels after setup_axes()
    plt.xlabel("Regularizer Coeff, $\lambda$", fontsize=55, fontname='Arial', fontweight='bold')
    plt.ylabel("Area from parity", fontsize=55, fontname='Arial', fontweight='bold')

    # Ensure axis ticks also use correct font
    plt.xticks(fontname='Arial', fontsize=50, fontweight='bold')
    plt.yticks(fontname='Arial', fontsize=50, fontweight='bold')

    # Set ticks to display at 0.5 intervals
    from matplotlib.ticker import MultipleLocator
    ax.xaxis.set_major_locator(MultipleLocator(0.5))


def save_plot(outdir, outname):
    """Save current plot"""
    plt.savefig(os.path.join(outdir, "png", outname + ".png"), bbox_inches="tight", dpi=300)
    plt.savefig(os.path.join(outdir, "pdf", outname + ".pdf"), bbox_inches="tight")
    plt.close()


def plot_calibration(df, x_input="Mean Predicted Avg",
                     y_input="Empirical Probability",
                     x_name="Mean Predicted",
                     y_name="Empirical Probability",
                     method_order=METHOD_ORDER,
                     avg_x=False):
    """Plot calibration"""

    methods = df['method_name']
    uniq_methods = pd.unique(methods)
    method_order = [j for j in METHOD_ORDER if j in uniq_methods]
    method_df = []

    if avg_x:
        df_copy = df.copy()
        new_list = [0]
        new_x_map = {}
        for method in uniq_methods:
            temp_vals = df[df['method_name'] == method][x_input]
            new_ar = np.vstack(temp_vals)
            new_ar = np.nanmean(new_ar, 0)
            new_x_map[method] = new_ar
        df_copy[x_input] = [new_x_map[method] for method in methods]
        df = df_copy

    x, y = df[x_input].values, df[y_input].values

    method_df = [{x_name: xx, y_name: yy, "Method": method}
                 for x_i, y_i, method in zip(x, y, methods)
                 for xx, yy in zip(x_i, y_i)]
    method_df = pd.DataFrame(method_df)

    sns.lineplot(x=x_name, y=y_name, hue="Method", alpha=0.8,
                 hue_order=method_order,
                 data=method_df,
                 palette=METHOD_COLORS)

    x = np.linspace(0, 1, 100)
    plt.plot(x, x, linestyle='--', color="black")

    # Set axis label font
    plt.xlabel(x_name, fontsize=30, fontname='Arial', fontweight='bold')
    plt.ylabel(y_name, fontsize=30, fontname='Arial', fontweight='bold')

    # Set up axes
    setup_axes()


def conf_percentile_lineplot(df, error_col_name="rmse", error_title="RMSE",
                              xlabel="Confidence Percentile",
                              y_points=1000,
                              truncate_tail=False):
    """Confidence percentile lineplot"""

    methods = df['method_name']
    y = df[error_col_name]

    uniq_methods = set(methods.values)
    method_order = [j for j in METHOD_ORDER if j in uniq_methods]
    method_df = []

    slice_obj = slice(None) if not truncate_tail else slice(-1)

    lines = [line if len(line) < y_points
             else np.array(line)[np.linspace(0, len(line) - 1, y_points).astype(int)]
             for line in y]

    method_df = [{"Ratio": x, error_title: y, "Method": method}
                 for line, method in zip(lines, methods)
                 for x, y in zip(np.linspace(0, 1, len(line))[slice_obj],
                                line[slice_obj])]
    method_df = pd.DataFrame(method_df)

    sns.lineplot(x="Ratio", y=error_title, hue="Method",
                 palette=METHOD_COLORS,
                 hue_order=method_order, data=method_df)

    # Set axis label font
    plt.xlabel(xlabel, fontsize=30, fontname='Arial', fontweight='bold')
    plt.ylabel(error_title, fontsize=30, fontname='Arial', fontweight='bold')

    # Set up axes
    setup_axes()


def distribute_task_plots(df, plot_fn):
    """Distribute task plots"""

    num_tasks = len(pd.unique(df['task_name']))
    num_cols = int(min(num_tasks, 4))
    num_rows = int(np.ceil(num_tasks / num_cols))

    figsize = (20, 6 * num_rows)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)

    for task_index, (task_name, task_df) in enumerate(df.groupby("task_name")):
        row_num = task_index // (num_cols)
        col_num = task_index % num_cols

        if num_rows > 1:
            ax = axes[row_num, col_num]
        else:
            ax = axes[col_num]

        plt.sca(ax)
        plot_fn(task_df)
        ax.set_title(f"{task_name}", fontsize=30, fontname='Arial', fontweight='bold')

        # Set up axes
        setup_axes(ax)

        if row_num != num_rows - 1:
            ax.set_xlabel("")

        if col_num != 0:
            ax.set_ylabel("")

        if col_num != num_cols - 1 or row_num != 0:
            ax.get_legend().remove()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-data-file", help="Location of full data file")
    parser.add_argument("--summary-data-file", help="Location of summary data file", default=None)
    parser.add_argument("--summary-data-task-file", help="Location of summary data file for tasks", default=None)
    parser.add_argument("--plot-type", type=str,
                        choices=["pfas"],
                        default="pfas")
    parser.add_argument("--outdir", default="results/figures", help="Directory to save figures")
    return parser.parse_args()


def make_pfas_plots(full_df, summary_df, results_dir):
    """Make pfas plots"""

    y_points = 100
    pfas_rename = {
        "ensemble": "ensemble",
        "dropout": "dropout",
        "evidence_new_reg_0.2": "evidence"
    }

    make_tuning_plot_rmse(summary_df, error_col_name="rmse",
                          error_title="Top 10% RMSE",
                          cutoff=0.10)

    # Set legend font
    legend = plt.legend(title="Method", loc='lower right', prop={'size': 12})
    for text in legend.get_texts():
        text.set_fontname('Arial')
        text.set_fontsize(18)
        text.set_fontweight('bold')
    if legend.get_title():
        legend.get_title().set_fontname('Arial')
        legend.get_title().set_fontsize(18)
        legend.get_title().set_fontweight('bold')

    plt.savefig(os.path.join(results_dir, "pfas_rmse_evidence_tuning.svg"),
                format="svg", dpi=300, bbox_inches='tight')
    plt.close()

    make_area_plots(summary_df, x_input="Expected Probability",
                    y_input="Predicted Probability")

    # Set axis border width to 1
    ax = plt.gca()
    ax.spines['top'].set_linewidth(2.5)
    ax.spines['right'].set_linewidth(2.5)
    ax.spines['bottom'].set_linewidth(2.5)
    ax.spines['left'].set_linewidth(2.5)

    # Set legend font
    legend = plt.legend(title="Dataset", loc='lower right',
                        prop={'size': 20, 'family': 'Arial', 'weight': 'bold'})
    for text in legend.get_texts():
        text.set_fontname('Arial')
        text.set_fontsize(28)
        text.set_fontweight('bold')
    if legend.get_title():
        legend.get_title().set_fontname('Arial')
        legend.get_title().set_fontsize(28)
        legend.get_title().set_fontweight('bold')

    plt.savefig(os.path.join(results_dir, "pfas_evidence_tuning_plot_summary.svg"),
                format="svg", dpi=300, bbox_inches='tight')
    plt.close()

    df_list = [full_df, summary_df]
    for df_index, df in enumerate(df_list):
        df['method_name'] = rename_method_df_none(df['method_name'], rename_key=pfas_rename)
        df_list[df_index] = df[[isinstance(i, str) for i in df['method_name']]].reset_index(drop=True)
    full_df, summary_df = df_list

    spearman_r_summary = plot_spearman_r(full_df, std=True, is_pfas=True,
                                          use_fresh_colors=True)

    # Set legend font
    legend = plt.legend(title="Method", loc='upper left', prop={'size': 12})
    for text in legend.get_texts():
        text.set_fontname('Arial')
        text.set_fontsize(18)
        text.set_fontweight('bold')
    if legend.get_title():
        legend.get_title().set_fontname('Arial')
        legend.get_title().set_fontsize(18)
        legend.get_title().set_fontweight('bold')

    spearman_r_summary.to_csv(os.path.join(results_dir, "spearman_r_pfas_summary_stats.csv"))
    plt.savefig(os.path.join(results_dir, "spearman_r_pfas.svg"),
                format="svg", dpi=300, bbox_inches='tight')
    plt.close()

    for dataset_name, summary_df_sub in summary_df.groupby("dataset"):
        plot_calibration(summary_df_sub,
                         x_input="Expected Probability",
                         y_input="Predicted Probability",
                         x_name="Expected Probability",
                         y_name="Predicted Probability")

        # Set legend font
        legend = plt.legend(title="Method", loc='lower right', prop={'size': 12})
        for text in legend.get_texts():
            text.set_fontname('Arial')
            text.set_fontsize(18)
            text.set_fontweight('bold')
        if legend.get_title():
            legend.get_title().set_fontname('Arial')
            legend.get_title().set_fontsize(18)
            legend.get_title().set_fontweight('bold')

        plt.savefig(os.path.join(results_dir, f"calibration_plot_pfas_{dataset_name}.svg"),
                    format="svg", dpi=300, bbox_inches='tight')
        plt.close()

        conf_percentile_lineplot(summary_df_sub, error_col_name="rmse", error_title="RMSE",
                                 y_points=y_points, truncate_tail=True)

        if dataset_name == "SI-2_KAW" or dataset_name == "SI-2_W":
            legend = plt.legend(title="Method", loc='upper left', prop={'size': 12})
        elif dataset_name == "SI-2_KOA":
            legend = plt.legend(title="Method", loc='lower left', prop={'size': 12})
        elif dataset_name == "SI-2_KOC":
            legend = plt.legend(title="Method", loc='upper right', prop={'size': 12})
        elif dataset_name == "SI-2_KOW":
            legend = plt.legend(title="Method", loc='upper right', prop={'size': 12})
        else:
            legend = plt.legend(title="Method", loc='lower right', prop={'size': 12})

        # Set legend font
        for text in legend.get_texts():
            text.set_fontname('Arial')
            text.set_fontsize(18)
            text.set_fontweight('bold')
        if legend.get_title():
            legend.get_title().set_fontname('Arial')
            legend.get_title().set_fontsize(18)
            legend.get_title().set_fontweight('bold')

        plt.savefig(os.path.join(results_dir, f"pfas_rmse_{dataset_name}.svg"),
                    format="svg", dpi=300, bbox_inches='tight')
        plt.close()

        conf_percentile_lineplot(summary_df_sub, error_col_name="mae", error_title="MAE",
                                 y_points=y_points, truncate_tail=True)

        # Set legend font
        legend = plt.legend(title="Method", loc='lower right', prop={'size': 12})
        for text in legend.get_texts():
            text.set_fontname('Arial')
            text.set_fontsize(18)
            text.set_fontweight('bold')
        if legend.get_title():
            legend.get_title().set_fontname('Arial')
            legend.get_title().set_fontsize(18)
            legend.get_title().set_fontweight('bold')

        plt.savefig(os.path.join(results_dir, f"pfas_mae_{dataset_name}.svg"),
                    format="svg", dpi=300, bbox_inches='tight')
        plt.close()

    make_cutoff_table(summary_df, outdir=results_dir,
                      table_data_name="PFAS (RMSE)",
                      output_metric="rmse",
                      out_name="cutoff_table_pfas_rmse.txt",
                      export_stds=True, significant_best=True,
                      higher_better=False)

    make_cutoff_table(summary_df, outdir=results_dir,
                      table_data_name="PFAS (MAE)",
                      output_metric="mae",
                      out_name="cutoff_table_pfas_mae.txt",
                      export_stds=True, significant_best=True,
                      higher_better=False)


if __name__ == "__main__":
    args = get_args()
    full_df = pd.read_csv(args.full_data_file, sep="\t", index_col=0)

    # Only process regression data
    summary_df = pd.read_csv(args.summary_data_file, sep="\t", index_col=0)
    new_columns = {}
    for col in REGR_SUMMARY_NAMES:
        new_col_vals = []
        for col_vals in summary_df[col]:
            if isinstance(col_vals, str):
                if "," not in col_vals:
                    eval_values = [float(i) for i in col_vals.strip()[1:-1].split()]
                else:
                    eval_values = eval(col_vals)
            else:
                eval_values = col_vals
            new_col_vals.append(eval_values)
        new_columns[col] = new_col_vals

    for col, val in new_columns.items():
        summary_df[col] = val

    results_dir = args.outdir
    os.makedirs(results_dir, exist_ok=True)

    # Modify seaborn style settings, remove grid
    sns.set(font_scale=1.3, style="white")

    # Set matplotlib global parameters to ensure font and tick display
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['xtick.labelsize'] = 26
    plt.rcParams['ytick.labelsize'] = 26
    plt.rcParams['legend.fontsize'] = 26
    plt.rcParams['axes.labelsize'] = 30
    plt.rcParams['axes.titlesize'] = 30
    plt.rcParams['xtick.major.size'] = 5
    plt.rcParams['xtick.minor.size'] = 3
    plt.rcParams['ytick.major.size'] = 5
    plt.rcParams['ytick.minor.size'] = 3
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['axes.grid'] = False  # Remove grid

    results_dir = os.path.join(results_dir, args.plot_type)

    for result_format in ["png", "pdf"]:
        os.makedirs(os.path.join(results_dir, result_format), exist_ok=True)

    full_df = full_df.query("partition == 'test'")

    if args.plot_type == "pfas":
        make_pfas_plots(full_df, summary_df, results_dir)
    else:
        raise NotImplementedError()