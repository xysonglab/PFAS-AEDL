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

# PFAS性质配置
PFAS_PROPERTIES = {
    'logSW': {
        'pred_col': 'logSW_pred',
        'true_col': 'logSW_true',
        'display_name': 'Water Solubility (logSW)',
        'unit': 'log(mol/L)'
    },
    'logKOW': {
        'pred_col': 'logKOW_pred',
        'true_col': 'logKOW_true',
        'display_name': 'Octanol-Water Partition Coefficient (logKOW)',
        'unit': 'log(unitless)'
    },
    'logKAW': {
        'pred_col': 'logKAW_pred',
        'true_col': 'logKAW_true',
        'display_name': 'Air-Water Partition Coefficient (logKAW)',
        'unit': 'log(unitless)'
    },
    'logKOA': {
        'pred_col': 'logKOA_pred',
        'true_col': 'logKOA_true',
        'display_name': 'Octanol-Air Partition Coefficient (logKOA)',
        'unit': 'log(unitless)'
    },
    'logKOC': {
        'pred_col': 'logKOC_pred',
        'true_col': 'logKOC_true',
        'display_name': 'Organic Carbon-Water Partition Coefficient (logKOC)',
        'unit': 'log(L/kg)'
    }
}


def calculate_regression_metrics(y_true, y_pred):
    """计算回归性能指标"""
    # 移除NaN值
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


def plot_primary(args):
    """
    绘制主要数据集上的模型性能图（预测值vs真实值）
    """
    if args.property not in PFAS_PROPERTIES:
        raise ValueError(f"Property {args.property} not supported. Choose from {list(PFAS_PROPERTIES.keys())}")

    prop_config = PFAS_PROPERTIES[args.property]

    for method in args.methods:
        files = glob.glob(os.path.join(args.preds_path, f"{method}/*.csv"))
        for fname in files:
            df = pd.read_csv(fname)

            # 获取对应列名
            pred_col = prop_config['pred_col']
            true_col = prop_config['true_col']

            if pred_col not in df.columns or true_col not in df.columns:
                print(f"Warning: Required columns {pred_col}, {true_col} not found in {fname}")
                continue

            preds = df[pred_col]
            targets = df[true_col]

            # 计算性能指标
            metrics = calculate_regression_metrics(targets.values, preds.values)

            # 创建回归图
            g = sns.jointplot(x=pred_col, y=true_col, data=df, kind='reg')

            # 添加对角线
            ax = g.ax_joint
            lims = [
                np.min([ax.get_xlim(), ax.get_ylim()]),
                np.max([ax.get_xlim(), ax.get_ylim()]),
            ]
            ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)

            # 添加性能指标文本
            if metrics:
                textstr = f'RMSE: {metrics["rmse"]:.3f}\nR²: {metrics["r2"]:.3f}\nn: {metrics["n_samples"]}'
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
                        verticalalignment='top', bbox=props)

            # 设置标签
            ax.set_xlabel(f'Predicted {prop_config["display_name"]}')
            ax.set_ylabel(f'True {prop_config["display_name"]}')

            save_name = fname.split('.csv')[0]
            save_path = f"{save_name}_{args.property}_performance{args.ext}"
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()


def plot_uncertainty_analysis(args):
    """
    绘制不确定性分析图，包括：
    1. 预测值vs真实值（按不确定性着色）
    2. 预测值vs不确定性联合图
    3. 不确定性筛选效果分析
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

            # 获取列名
            pred_col = prop_config['pred_col']
            true_col = prop_config['true_col']

            if pred_col not in df.columns or true_col not in df.columns:
                print(f"Warning: Required columns not found in {fname}")
                continue

            predictions = df[pred_col]
            targets = df[true_col]

            # 获取不确定性列
            if args.use_stds and 'stds' in df.columns:
                uncertainties = df['stds']
                uncertainty_label = 'Standard Deviation'
            elif 'uncertainty' in df.columns:
                uncertainties = df['uncertainty']
                uncertainty_label = 'Uncertainty'
            else:
                print(f"Warning: No uncertainty column found in {fname}")
                continue

            #### 1. 预测值vs真实值（按不确定性着色）
            plt.figure(figsize=(8, 6))
            scatter = plt.scatter(predictions, targets, c=np.log(uncertainties),
                                  s=60, alpha=0.7, cmap='viridis')
            plt.colorbar(scatter, label='log(Uncertainty)')

            # 添加对角线
            lims = [
                np.min([plt.xlim(), plt.ylim()]),
                np.max([plt.xlim(), plt.ylim()]),
            ]
            plt.plot(lims, lims, 'k--', alpha=0.75, zorder=0)

            plt.xlabel(f'Predicted {prop_config["display_name"]}')
            plt.ylabel(f'True {prop_config["display_name"]}')
            plt.title(f'{method.title()} - {prop_config["display_name"]}')

            save_path_performance = f"{save_name}_{args.property}_uncertainty_colored{args.ext}"
            plt.savefig(save_path_performance, bbox_inches='tight')
            plt.close()

            #### 2. 预测值vs不确定性联合图
            g = sns.jointplot(x=predictions, y=uncertainties, alpha=0.7)
            g.ax_joint.set_yscale('log')
            g.ax_joint.set_xlabel(f'Predicted {prop_config["display_name"]}')
            g.ax_joint.set_ylabel(uncertainty_label)

            save_path_joint = f"{save_name}_{args.property}_pred_vs_uncertainty{args.ext}"
            plt.savefig(save_path_joint, bbox_inches='tight')
            plt.close()

            #### 3. 不确定性筛选效果分析
            # 移除NaN值
            mask = ~(np.isnan(predictions) | np.isnan(targets) | np.isnan(uncertainties))
            pred_clean = predictions[mask].values
            true_clean = targets[mask].values
            unc_clean = uncertainties[mask].values

            if len(pred_clean) == 0:
                continue

            # 选择top-k预测最好的化合物（对于大多数性质，较小的绝对值可能更好，但这取决于具体性质）
            # 这里我们基于不确定性进行筛选分析
            percentiles = range(0, 101, 5)  # 0%, 5%, 10%, ..., 100%

            for p in percentiles:
                unc_thresh = np.percentile(unc_clean, p)

                # 筛选低不确定性的样本
                low_unc_mask = unc_clean <= unc_thresh
                if np.sum(low_unc_mask) < 2:  # 至少需要2个样本
                    continue

                # 计算筛选后的性能
                pred_filtered = pred_clean[low_unc_mask]
                true_filtered = true_clean[low_unc_mask]

                metrics_filtered = calculate_regression_metrics(true_filtered, pred_filtered)
                metrics_all = calculate_regression_metrics(true_clean, pred_clean)

                if metrics_filtered and metrics_all:
                    df_percentiles = df_percentiles.append({
                        'Percentile': p,
                        'RMSE': metrics_filtered['rmse'],
                        'MAE': metrics_filtered['mae'],
                        'R2': metrics_filtered['r2'],
                        'N_samples': metrics_filtered['n_samples'],
                        'Method': 'uncertainty_filtered',
                        'Trial': i,
                    }, ignore_index=True)

                    # 也保存全部数据的性能作为基线
                    df_percentiles = df_percentiles.append({
                        'Percentile': p,
                        'RMSE': metrics_all['rmse'],
                        'MAE': metrics_all['mae'],
                        'R2': metrics_all['r2'],
                        'N_samples': metrics_all['n_samples'],
                        'Method': 'all_data',
                        'Trial': i,
                    }, ignore_index=True)

        # 绘制不确定性筛选效果
        if not df_percentiles.empty:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            # RMSE vs Percentile
            sns.lineplot(data=df_percentiles, x='Percentile', y='RMSE',
                         hue='Method', ax=axes[0, 0])
            axes[0, 0].set_title('RMSE vs Uncertainty Percentile')

            # R² vs Percentile
            sns.lineplot(data=df_percentiles, x='Percentile', y='R2',
                         hue='Method', ax=axes[0, 1])
            axes[0, 1].set_title('R² vs Uncertainty Percentile')

            # Sample count vs Percentile
            sns.lineplot(data=df_percentiles, x='Percentile', y='N_samples',
                         hue='Method', ax=axes[1, 0])
            axes[1, 0].set_title('Sample Count vs Uncertainty Percentile')

            # MAE vs Percentile
            sns.lineplot(data=df_percentiles, x='Percentile', y='MAE',
                         hue='Method', ax=axes[1, 1])
            axes[1, 1].set_title('MAE vs Uncertainty Percentile')

            plt.tight_layout()
            save_path_analysis = f"{save_name}_{args.property}_uncertainty_analysis{args.ext}"
            plt.savefig(save_path_analysis, bbox_inches='tight')
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
    parser.add_argument('--ext', type=str, default='.pdf',
                        help='File extension for saved plots')
    parser.add_argument('--analysis_type', type=str, default='both',
                        choices=['primary', 'uncertainty', 'both'],
                        help='Type of analysis to perform')

    args = parser.parse_args()

    if args.analysis_type in ['primary', 'both']:
        plot_primary(args)

    if args.analysis_type in ['uncertainty', 'both']:
        plot_uncertainty_analysis(args)