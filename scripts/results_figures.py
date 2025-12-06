import argparse
import os
import seaborn as sns
import pandas as pd
import glob
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Global font settings
plt.rcParams.update({
    'font.family': 'Arial',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'font.weight': 'bold',
})

# PFAS property configuration
PFAS_PROPERTIES = {
    'logSW': {
        'pred_col': 'logSW_pred',
        'true_col': 'logSW_true',
        'display_name': r'$\log S_{\mathrm{W}}$',
        'unit': 'log(mol/L)'
    },
    'logKOW': {
        'pred_col': 'logKOW_pred',
        'true_col': 'logKOW_true',
        'display_name': r'$\log K_{\mathrm{OW}}$',
        'unit': 'log(unitless)'
    },
    'logKAW': {
        'pred_col': 'logKAW_pred',
        'true_col': 'logKAW_true',
        'display_name': r'$\log K_{\mathrm{AW}}$',
        'unit': 'log(unitless)'
    },
    'logKOA': {
        'pred_col': 'logKOA_pred',
        'true_col': 'logKOA_true',
        'display_name': r'$\log K_{\mathrm{OA}}$',
        'unit': 'log(unitless)'
    },
    'logKOC': {
        'pred_col': 'logKOC_pred',
        'true_col': 'logKOC_true',
        'display_name': r'$\log K_{\mathrm{OC}}$',
        'unit': 'log(L/kg)'
    }
}


def calculate_regression_metrics(y_true, y_pred):
    """Calculate regression performance metrics"""
    # Remove NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]

    if len(y_true_clean) == 0:
        return {}

    rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    r2 = pearsonr(y_true_clean, y_pred_clean)[0] ** 2
    spearman_rho = spearmanr(y_true_clean, y_pred_clean)[0]

    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'spearman_rho': spearman_rho,
        'n_samples': len(y_true_clean)
    }


def set_font_properties(ax, xlabel_size=36, ylabel_size=36, tick_size=30):
    """Set axis font properties"""
    # Set axis labels
    ax.xaxis.label.set_fontsize(xlabel_size)
    ax.yaxis.label.set_fontsize(ylabel_size)
    ax.xaxis.label.set_fontweight('bold')
    ax.yaxis.label.set_fontweight('bold')
    ax.xaxis.label.set_fontname('Arial')
    ax.yaxis.label.set_fontname('Arial')

    # Set tick labels
    ax.tick_params(axis='both', which='major', labelsize=tick_size)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname('Arial')
        label.set_fontweight('bold')


def setup_axes(ax=None):
    """Set axis style, making left and bottom axis ticks point inward"""
    if ax is None:
        ax = plt.gca()

    # Show all spines
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)

    # Set spine width to 2
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)

    # Set bottom and left ticks pointing inward with bold tick lines
    ax.tick_params(axis='x', which='both',
                   bottom=True, top=False,
                   labelbottom=True, labeltop=False,
                   direction='in',
                   length=5,
                   width=2)  # Bold x-axis tick lines

    ax.tick_params(axis='y', which='both',
                   left=True, right=False,
                   labelleft=True, labelright=False,
                   direction='in',
                   length=5,
                   width=2)  # Bold y-axis tick lines

    return ax


def plot_primary(args):
    """
    Plot model performance on primary dataset (predicted vs true values)
    """
    if args.property not in PFAS_PROPERTIES:
        raise ValueError(f"Property {args.property} not supported. Choose from {list(PFAS_PROPERTIES.keys())}")

    prop_config = PFAS_PROPERTIES[args.property]

    for method in args.methods:
        files = glob.glob(os.path.join(args.preds_path, f"{method}/*.csv"))
        for fname in files:
            df = pd.read_csv(fname)

            # Get corresponding column names
            pred_col = prop_config['pred_col']
            true_col = prop_config['true_col']

            if pred_col not in df.columns or true_col not in df.columns:
                print(f"Warning: Required columns {pred_col}, {true_col} not found in {fname}")
                continue

            preds = df[pred_col]
            targets = df[true_col]

            # Calculate performance metrics
            metrics = calculate_regression_metrics(targets.values, preds.values)

            # Create regression plot - using viridis color scheme
            plt.figure(figsize=(10, 8))
            ax = plt.gca()

            # Use scatter plot instead of jointplot
            plt.scatter(preds, targets, alpha=0.5, color='#5F9EA0', s=60)

            # Add regression line
            z = np.polyfit(preds, targets, 1)
            p = np.poly1d(z)
            x_range = np.linspace(preds.min(), preds.max(), 100)
            plt.plot(x_range, p(x_range), color='#5F9EA0', linewidth=2)

            # Set up axes
            setup_axes(ax)

            # Add diagonal line
            lims = [
                np.min([ax.get_xlim(), ax.get_ylim()]),
                np.max([ax.get_xlim(), ax.get_ylim()]),
            ]
            ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0, linewidth=2)

            # Add performance metrics text
            if metrics:
                textstr = f'RMSE: {metrics["rmse"]:.3f}\nR²: {metrics["r2"]:.3f}\nn: {metrics["n_samples"]}'
                props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black')
                ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=30,
                        verticalalignment='top', bbox=props, fontname='Arial', fontweight='bold')

            # Set labels and fonts
            ax.set_xlabel(f'Predicted {prop_config["display_name"]}', fontsize=44, fontweight='bold')
            ax.set_ylabel(f'True {prop_config["display_name"]}', fontsize=44, fontweight='bold')
            ax.tick_params(axis='both', which='major', labelsize=36)

            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontname('Arial')
                label.set_fontweight('bold')

            save_name = fname.split('.csv')[0]
            save_path = f"{save_name}_{args.property}_performance{args.ext}"
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()


def plot_uncertainty_analysis(args):
    """
    Plot uncertainty analysis including:
    1. Predicted vs true values (colored by uncertainty) - viridis gradient
    2. Predicted vs uncertainty scatter plot - viridis color, only showing left and bottom axes
    3. Uncertainty filtering effect analysis - viridis color
    """
    if args.property not in PFAS_PROPERTIES:
        raise ValueError(f"Property {args.property} not supported. Choose from {list(PFAS_PROPERTIES.keys())}")

    prop_config = PFAS_PROPERTIES[args.property]

    for method in args.methods:
        files = glob.glob(os.path.join(args.preds_path, f"{method}/*.csv"))
        df_percentiles = pd.DataFrame()

        for i, fname in enumerate(files):
            save_name = fname.split('.csv')[0]
            df = pd.read_csv(fname)

            # Get column names
            pred_col = prop_config['pred_col']
            true_col = prop_config['true_col']

            if pred_col not in df.columns or true_col not in df.columns:
                print(f"Warning: Required columns not found in {fname}")
                continue

            predictions = df[pred_col]
            targets = df[true_col]

            # Get uncertainty column
            if args.use_stds and 'stds' in df.columns:
                uncertainties = df['stds']
                uncertainty_label = 'Standard Deviation'
            elif 'uncertainty' in df.columns:
                uncertainties = df['uncertainty']
                uncertainty_label = 'Uncertainty'
            else:
                print(f"Warning: No uncertainty column found in {fname}")
                continue

            #### 1. Predicted vs true values (colored by uncertainty) - using viridis gradient
            # Calculate performance metrics
            metrics = calculate_regression_metrics(targets.values, predictions.values)

            plt.figure(figsize=(10, 8))
            ax = plt.gca()

            scatter = plt.scatter(predictions, targets, c=np.log(uncertainties),
                                  s=60, alpha=0.7, cmap='viridis')
            cbar = plt.colorbar(scatter, label='log(Uncertainty)')
            cbar.ax.tick_params(labelsize=36)
            cbar.ax.yaxis.label.set_size(44)
            cbar.ax.yaxis.label.set_fontweight('bold')
            cbar.ax.yaxis.label.set_fontname('Arial')
            for label in cbar.ax.get_yticklabels():
                label.set_fontname('Arial')
                label.set_fontweight('bold')

            # Set up axes
            setup_axes(ax)

            # Set axis labels and fonts
            ax.set_xlabel(f'Predicted {prop_config["display_name"]}', fontsize=44, fontweight='bold',
                          fontname='Arial')
            ax.set_ylabel(f'True {prop_config["display_name"]}', fontsize=44, fontweight='bold', fontname='Arial')
            ax.tick_params(axis='both', which='major', labelsize=36)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontname('Arial')
                label.set_fontweight('bold')

            # Add diagonal line
            lims = [
                np.min([plt.xlim(), plt.ylim()]),
                np.max([plt.xlim(), plt.ylim()]),
            ]
            plt.plot(lims, lims, 'k--', alpha=0.75, zorder=0, linewidth=2)

            # Add performance metrics text box
            if metrics:
                textstr = f'RMSE: {metrics["rmse"]:.3f}\nR²: {metrics["r2"]:.3f}\nn: {metrics["n_samples"]}'
                props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black')
                plt.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=36,
                         verticalalignment='top', bbox=props, fontname='Arial', fontweight='bold')

            save_path_performance = f"{save_name}_{args.property}_uncertainty_colored{args.ext}"
            plt.savefig(save_path_performance, bbox_inches='tight', dpi=300)
            plt.close()

            #### 2. Predicted vs uncertainty scatter plot - viridis color, left and bottom axes ticks pointing inward
            plt.figure(figsize=(10, 8))
            ax = plt.gca()

            plt.scatter(predictions, uncertainties, alpha=0.7, color='#5F9EA0', s=60)
            plt.yscale('log')

            # Set up axes
            setup_axes(ax)

            ax.set_xlabel(f'Predicted {prop_config["display_name"]}', fontsize=44, fontweight='bold')
            ax.set_ylabel(uncertainty_label, fontsize=44, fontweight='bold')
            ax.tick_params(axis='both', which='major', labelsize=36)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontname('Arial')
                label.set_fontweight('bold')

            save_path_joint = f"{save_name}_{args.property}_pred_vs_uncertainty{args.ext}"
            plt.savefig(save_path_joint, bbox_inches='tight', dpi=300)
            plt.close()

            #### 3. Uncertainty filtering effect analysis
            # Remove NaN values
            mask = ~(np.isnan(predictions) | np.isnan(targets) | np.isnan(uncertainties))
            pred_clean = predictions[mask].values
            true_clean = targets[mask].values
            unc_clean = uncertainties[mask].values

            if len(pred_clean) == 0:
                continue

            # Select top-k predictions with best performance
            percentiles = range(0, 101, 5)  # 0%, 5%, 10%, ..., 100%

            for p in percentiles:
                unc_thresh = np.percentile(unc_clean, p)

                # Filter samples with low uncertainty
                low_unc_mask = unc_clean <= unc_thresh
                if np.sum(low_unc_mask) < 2:  # Need at least 2 samples
                    continue

                # Calculate performance after filtering
                pred_filtered = pred_clean[low_unc_mask]
                true_filtered = true_clean[low_unc_mask]

                metrics_filtered = calculate_regression_metrics(true_filtered, pred_filtered)
                metrics_all = calculate_regression_metrics(true_clean, pred_clean)

                if metrics_filtered and metrics_all:
                    df_percentiles = pd.concat([df_percentiles, pd.DataFrame({
                        'Percentile': [p],
                        'RMSE': [metrics_filtered['rmse']],
                        'MAE': [metrics_filtered['mae']],
                        'R2': [metrics_filtered['r2']],
                        'N_samples': [metrics_filtered['n_samples']],
                        'Method': ['uncertainty_filtered'],
                        'Trial': [i],
                    })], ignore_index=True)

                    # Also save all data performance as baseline
                    df_percentiles = pd.concat([df_percentiles, pd.DataFrame({
                        'Percentile': [p],
                        'RMSE': [metrics_all['rmse']],
                        'MAE': [metrics_all['mae']],
                        'R2': [metrics_all['r2']],
                        'N_samples': [metrics_all['n_samples']],
                        'Method': ['all_data'],
                        'Trial': [i],
                    })], ignore_index=True)

        # Plot uncertainty filtering effect - viridis color
        if not df_percentiles.empty:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))

            # Define viridis palette
            palette = {'uncertainty_filtered': '#5F9EA0', 'all_data': '#5F9EA0'}

            # RMSE vs Percentile
            sns.lineplot(data=df_percentiles, x='Percentile', y='RMSE',
                         hue='Method', ax=axes[0, 0], palette=palette, linewidth=3)
            axes[0, 0].set_title('RMSE vs Uncertainty Percentile', fontsize=36, fontweight='bold', fontname='Arial')
            axes[0, 0].set_xlabel('Percentile', fontsize=30, fontweight='bold', fontname='Arial')
            axes[0, 0].set_ylabel('RMSE', fontsize=30, fontweight='bold', fontname='Arial')
            setup_axes(axes[0, 0])
            set_font_properties(axes[0, 0])
            axes[0, 0].legend(fontsize=26)

            # R² vs Percentile
            sns.lineplot(data=df_percentiles, x='Percentile', y='R2',
                         hue='Method', ax=axes[0, 1], palette=palette, linewidth=3)
            axes[0, 1].set_title('R² vs Uncertainty Percentile', fontsize=30, fontweight='bold', fontname='Arial')
            axes[0, 1].set_xlabel('Percentile', fontsize=30, fontweight='bold', fontname='Arial')
            axes[0, 1].set_ylabel('R²', fontsize=30, fontweight='bold', fontname='Arial')
            setup_axes(axes[0, 1])
            set_font_properties(axes[0, 1])
            axes[0, 1].legend(fontsize=26)

            # Sample count vs Percentile
            sns.lineplot(data=df_percentiles, x='Percentile', y='N_samples',
                         hue='Method', ax=axes[1, 0], palette=palette, linewidth=3)
            axes[1, 0].set_title('Sample Count vs Uncertainty Percentile', fontsize=30, fontweight='bold',
                                 fontname='Arial')
            axes[1, 0].set_xlabel('Percentile', fontsize=30, fontweight='bold', fontname='Arial')
            axes[1, 0].set_ylabel('Sample Count', fontsize=30, fontweight='bold', fontname='Arial')
            setup_axes(axes[1, 0])
            set_font_properties(axes[1, 0])
            axes[1, 0].legend(fontsize=26)

            # MAE vs Percentile
            sns.lineplot(data=df_percentiles, x='Percentile', y='MAE',
                         hue='Method', ax=axes[1, 1], palette=palette, linewidth=3)
            axes[1, 1].set_title('MAE vs Uncertainty Percentile', fontsize=30, fontweight='bold', fontname='Arial')
            axes[1, 1].set_xlabel('Percentile', fontsize=30, fontweight='bold', fontname='Arial')
            axes[1, 1].set_ylabel('MAE', fontsize=30, fontweight='bold', fontname='Arial')
            setup_axes(axes[1, 1])
            set_font_properties(axes[1, 1])
            axes[1, 1].legend(fontsize=26)

            # Set fonts for all subplots
            for ax_row in axes:
                for ax in ax_row:
                    for item in ([ax.xaxis.label, ax.yaxis.label, ax.title] +
                                 ax.get_xticklabels() + ax.get_yticklabels()):
                        item.set_fontname('Arial')
                        item.set_fontweight('bold')
                    ax.title.set_fontsize(30)
                    ax.xaxis.label.set_fontsize(30)
                    ax.yaxis.label.set_fontsize(30)

            plt.tight_layout()
            save_path_analysis = f"{save_name}_{args.property}_uncertainty_analysis{args.ext}"
            plt.savefig(save_path_analysis, bbox_inches='tight', dpi=300)
            plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot PFAS property prediction results.')
    parser.add_argument('--methods', type=str, nargs='*', default=['evidence'],
                        help='Methods for which to plot results')
    parser.add_argument('--preds_path', type=str, default='./pfas_results',
                        help='Path to directory of prediction results')
    parser.add_argument('--property', type=str, required=True,
                        choices=list(PFAS_PROPERTIES.keys()),
                        help='PFAS property to analyze')
    parser.add_argument('--k', type=int, default=50,
                        help='Number of top predictions to consider for analysis')
    parser.add_argument('--use_stds', action='store_true', default=False,
                        help='Use standard deviations instead of uncertainty')
    parser.add_argument('--ext', type=str, default='.svg',
                        help='File extension for saved plots')
    parser.add_argument('--analysis_type', type=str, default='both',
                        choices=['primary', 'uncertainty', 'both'],
                        help='Type of analysis to perform')

    args = parser.parse_args()

    if args.analysis_type in ['primary', 'both']:
        plot_primary(args)

    if args.analysis_type in ['uncertainty', 'both']:
        plot_uncertainty_analysis(args)