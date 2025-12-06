"""
XGBoost Model Training with Optuna Optimization and SHAP Feature Importance Analysis
用于PFAS分子性质预测的完整训练流程（包含SHAP特征重要性分析）
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List
import warnings

warnings.filterwarnings('ignore')

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

import optuna
from optuna.samplers import TPESampler

import shap
from shap import Explanation

# 导入特征生成器
from chemprop.features.features_generators import (
    get_features_generator,
    get_available_features_generators
)


class XGBoostModelTrainer:
    """XGBoost模型训练器,包含Optuna优化、交叉验证和SHAP特征重要性分析"""

    def __init__(
            self,
            csv_file: str,
            smiles_column: str = 'smiles',
            target_column: str = 'target',
            features_generator: str = 'pfas',
            output_dir: str = './results/XGB',
            random_state: int = 42
    ):
        """
        初始化训练器

        Args:
            csv_file: 输入CSV文件路径
            smiles_column: SMILES列名
            target_column: 目标性质列名
            features_generator: 特征生成器名称
            output_dir: 输出目录
            random_state: 随机种子
        """
        self.csv_file = csv_file
        self.smiles_column = smiles_column
        self.target_column = target_column
        self.features_generator_name = features_generator
        self.output_dir = output_dir
        self.random_state = random_state

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 初始化
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.best_model = None
        self.cv_results = []
        self.study = None
        self.feature_names = None  # 保存特征名称
        self.shap_values = None  # 保存SHAP值
        self.shap_explainer = None  # SHAP解释器

        print(f"可用的特征生成器: {get_available_features_generators()}")
        print(f"使用特征生成器: {features_generator}")

    def load_data(self) -> None:
        """加载CSV数据"""
        print(f"\n{'=' * 60}")
        print("1. 加载数据")
        print(f"{'=' * 60}")

        self.data = pd.read_csv(self.csv_file)
        print(f"数据集大小: {len(self.data)} 条")
        print(f"列名: {list(self.data.columns)}")

        # 检查必需的列
        if self.smiles_column not in self.data.columns:
            raise ValueError(f"未找到SMILES列: {self.smiles_column}")
        if self.target_column not in self.data.columns:
            raise ValueError(f"未找到目标列: {self.target_column}")

        # 删除缺失值
        initial_size = len(self.data)
        self.data = self.data.dropna(subset=[self.smiles_column, self.target_column])
        if len(self.data) < initial_size:
            print(f"删除 {initial_size - len(self.data)} 条缺失数据")

        print(f"最终数据集大小: {len(self.data)} 条")
        print(f"目标值统计:\n{self.data[self.target_column].describe()}")

    def generate_features(self) -> None:
        """生成分子特征"""
        print(f"\n{'=' * 60}")
        print("2. 生成分子特征")
        print(f"{'=' * 60}")

        features_generator = get_features_generator(self.features_generator_name)

        features_list = []
        valid_indices = []

        for idx, smiles in enumerate(self.data[self.smiles_column]):
            try:
                features = features_generator(smiles)
                features_list.append(features)
                valid_indices.append(idx)
            except Exception as e:
                print(f"警告: SMILES {smiles} 特征生成失败: {str(e)}")
                continue

        if len(features_list) == 0:
            raise ValueError("没有成功生成任何特征!")

        self.X = np.array(features_list)
        self.y = self.data[self.target_column].iloc[valid_indices].values

        # 生成特征名称
        n_features = self.X.shape[1]
        self.feature_names = [f"feature_{i}" for i in range(n_features)]

        print(f"特征矩阵形状: {self.X.shape}")
        print(f"目标向量形状: {self.y.shape}")
        print(f"特征维度: {n_features}")

    def split_data(self, train_ratio: float = 0.8, val_ratio: float = 0.1) -> None:
        """
        划分数据集为训练集、验证集和测试集

        Args:
            train_ratio: 训练集比例
            val_ratio: 验证集比例
        """
        print(f"\n{'=' * 60}")
        print("3. 划分数据集 (8:1:1)")
        print(f"{'=' * 60}")

        # 首先划分出测试集
        test_ratio = 1 - train_ratio - val_ratio
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            self.X, self.y,
            test_size=test_ratio,
            random_state=self.random_state
        )

        # 然后从剩余数据中划分训练集和验证集
        val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_ratio_adjusted,
            random_state=self.random_state
        )

        print(f"训练集: {len(self.X_train)} 条 ({len(self.X_train) / len(self.X) * 100:.1f}%)")
        print(f"验证集: {len(self.X_val)} 条 ({len(self.X_val) / len(self.X) * 100:.1f}%)")
        print(f"测试集: {len(self.X_test)} 条 ({len(self.X_test) / len(self.X) * 100:.1f}%)")

        # 标准化特征
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_val = self.scaler.transform(self.X_val)
        self.X_test = self.scaler.transform(self.X_test)

        print("特征标准化完成")

    def optuna_optimization(self, n_trials: int = 20) -> Dict[str, Any]:
        """
        使用Optuna优化寻找最佳超参数

        Args:
            n_trials: 优化试验次数

        Returns:
            最佳参数字典
        """
        print(f"\n{'=' * 60}")
        print("4. Optuna超参数优化")
        print(f"{'=' * 60}")

        def objective(trial):
            """Optuna目标函数"""
            # 定义XGBoost超参数搜索空间
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'random_state': self.random_state,
                'n_jobs': -1,
                'tree_method': 'hist',
                'verbosity': 0
            }

            # 创建模型
            model = XGBRegressor(**params)

            # 5折交叉验证
            cv_scores = cross_val_score(
                model, self.X_train, self.y_train,
                cv=5,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )

            # 返回平均负MSE(Optuna会最小化目标函数)
            return -cv_scores.mean()

        # 创建Optuna study
        print(f"开始Optuna优化 (试验次数: {n_trials})...")

        # 设置采样器
        sampler = TPESampler(seed=self.random_state)

        self.study = optuna.create_study(
            direction='minimize',
            sampler=sampler,
            study_name='xgboost_hyperparameter_optimization'
        )

        # 优化
        self.study.optimize(
            objective,
            n_trials=n_trials,
            show_progress_bar=True,
            n_jobs=1
        )

        best_params = self.study.best_params
        best_params['random_state'] = self.random_state
        best_params['n_jobs'] = -1
        best_params['tree_method'] = 'hist'
        best_params['verbosity'] = 0

        print(f"\n最佳参数:")
        for key, value in best_params.items():
            print(f"  {key}: {value}")
        print(f"\n最佳交叉验证MSE: {self.study.best_value:.6f}")
        print(f"最佳交叉验证RMSE: {np.sqrt(self.study.best_value):.6f}")

        return best_params

    def cross_validation(self, params: Dict[str, Any], n_folds: int = 10) -> None:
        """
        使用最佳参数进行交叉验证

        Args:
            params: 模型参数
            n_folds: 交叉验证折数
        """
        print(f"\n{'=' * 60}")
        print(f"5. {n_folds}折交叉验证")
        print(f"{'=' * 60}")

        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)

        self.cv_results = []

        for fold, (train_idx, val_idx) in enumerate(kfold.split(self.X_train), 1):
            # 划分折
            X_fold_train = self.X_train[train_idx]
            X_fold_val = self.X_train[val_idx]
            y_fold_train = self.y_train[train_idx]
            y_fold_val = self.y_train[val_idx]

            # 训练模型
            model = XGBRegressor(**params)
            model.fit(X_fold_train, y_fold_train)

            # 预测
            train_pred = model.predict(X_fold_train)
            val_pred = model.predict(X_fold_val)

            # 计算指标 - 训练集
            train_rmse = np.sqrt(mean_squared_error(y_fold_train, train_pred))
            train_r2 = r2_score(y_fold_train, train_pred)
            train_mae = mean_absolute_error(y_fold_train, train_pred)

            # 计算指标 - 验证集（每个fold的测试集）
            val_rmse = np.sqrt(mean_squared_error(y_fold_val, val_pred))
            val_r2 = r2_score(y_fold_val, val_pred)
            val_mae = mean_absolute_error(y_fold_val, val_pred)

            result = {
                'fold': fold,
                'train_rmse': train_rmse,
                'train_r2': train_r2,
                'train_mae': train_mae,
                'val_rmse': val_rmse,
                'val_r2': val_r2,
                'val_mae': val_mae
            }
            self.cv_results.append(result)

            print(f"Fold {fold:2d}: "
                  f"Train RMSE={train_rmse:.4f}, Val RMSE={val_rmse:.4f}, "
                  f"Train R²={train_r2:.4f}, Val R²={val_r2:.4f}")

        # 计算平均指标
        cv_df = pd.DataFrame(self.cv_results)
        print(f"\n{'=' * 60}")
        print("交叉验证平均结果:")
        print(f"{'=' * 60}")
        print(f"训练集 - RMSE: {cv_df['train_rmse'].mean():.4f} ± {cv_df['train_rmse'].std():.4f}")
        print(f"验证集 - RMSE: {cv_df['val_rmse'].mean():.4f} ± {cv_df['val_rmse'].std():.4f}")
        print(f"训练集 - R²:   {cv_df['train_r2'].mean():.4f} ± {cv_df['train_r2'].std():.4f}")
        print(f"验证集 - R²:   {cv_df['val_r2'].mean():.4f} ± {cv_df['val_r2'].std():.4f}")
        print(f"训练集 - MAE:  {cv_df['train_mae'].mean():.4f} ± {cv_df['train_mae'].std():.4f}")
        print(f"验证集 - MAE:  {cv_df['val_mae'].mean():.4f} ± {cv_df['val_mae'].std():.4f}")

        # 保存交叉验证结果
        cv_df.to_csv(
            os.path.join(self.output_dir, 'cross_validation_results.csv'),
            index=False
        )

    def train_final_model(self, params: Dict[str, Any]) -> None:
        """
        使用全部训练数据训练最终模型

        Args:
            params: 模型参数
        """
        print(f"\n{'=' * 60}")
        print("6. 训练最终模型")
        print(f"{'=' * 60}")

        # 合并训练集和验证集
        X_train_full = np.vstack([self.X_train, self.X_val])
        y_train_full = np.concatenate([self.y_train, self.y_val])

        print(f"使用 {len(X_train_full)} 条数据训练最终模型...")

        # 训练模型
        self.best_model = XGBRegressor(**params)
        self.best_model.fit(X_train_full, y_train_full)

        print("模型训练完成!")

    def compute_shap_values(self, n_samples: int = 1000) -> None:
        """
        计算SHAP值（使用TreeExplainer）

        Args:
            n_samples: 用于计算SHAP值的样本数（如果数据太多可以采样）
        """
        print(f"\n{'=' * 60}")
        print("7. 计算SHAP值")
        print(f"{'=' * 60}")

        if self.best_model is None:
            print("错误: 模型尚未训练!")
            return

        # 合并训练集和验证集作为背景数据
        X_background = np.vstack([self.X_train, self.X_val])

        # 如果数据太多，采样一部分作为背景
        if len(X_background) > n_samples:
            indices = np.random.choice(len(X_background), n_samples, replace=False)
            X_background = X_background[indices]
            print(f"背景数据采样: {n_samples} 个样本")

        print(f"使用 {len(X_background)} 个样本作为背景数据计算SHAP值...")

        # 创建TreeExplainer
        print("创建SHAP TreeExplainer...")
        self.shap_explainer = shap.TreeExplainer(
            self.best_model,
            data=X_background,
            feature_names=self.feature_names,
            model_output='raw'
        )

        # 计算测试集的SHAP值
        print(f"计算 {len(self.X_test)} 个测试样本的SHAP值...")
        self.shap_values = self.shap_explainer.shap_values(self.X_test)

        print("SHAP值计算完成!")

        # 打印SHAP值的基本信息
        print(f"SHAP值形状: {self.shap_values.shape}")
        print(f"SHAP值范围: [{self.shap_values.min():.4f}, {self.shap_values.max():.4f}]")
        print(f"平均绝对SHAP值: {np.abs(self.shap_values).mean():.4f}")

    def analyze_shap_feature_importance(self) -> pd.DataFrame:
        """
        分析SHAP特征重要性并保存到CSV

        Returns:
            包含特征重要性的DataFrame
        """
        print(f"\n{'=' * 60}")
        print("8. SHAP特征重要性分析")
        print(f"{'=' * 60}")

        if self.shap_values is None:
            print("错误: SHAP值尚未计算!")
            return None

        # 1. 计算全局重要性（平均绝对SHAP值）
        shap_abs_mean = np.abs(self.shap_values).mean(axis=0)

        # 2. 计算全局重要性（SHAP值平方的均值）
        shap_squared_mean = (self.shap_values ** 2).mean(axis=0)

        # 3. 计算正负贡献统计
        shap_pos_mean = np.where(self.shap_values > 0, self.shap_values, 0).mean(axis=0)
        shap_neg_mean = np.where(self.shap_values < 0, self.shap_values, 0).mean(axis=0)

        # 4. 计算特征贡献频率（非零SHAP值的比例）
        non_zero_freq = (self.shap_values != 0).sum(axis=0) / len(self.shap_values)

        # 5. 计算特征贡献的强度（SHAP值的标准差）
        shap_std = self.shap_values.std(axis=0)

        # 6. 计算平均SHAP值（原始方向）
        shap_mean = self.shap_values.mean(axis=0)

        # 创建包含所有重要性指标的DataFrame
        importance_data = {
            'feature_name': self.feature_names,
            'shap_mean_abs': shap_abs_mean,
            'shap_mean_squared': shap_squared_mean,
            'shap_mean': shap_mean,
            'shap_std': shap_std,
            'shap_pos_mean': shap_pos_mean,
            'shap_neg_mean': shap_neg_mean,
            'non_zero_freq': non_zero_freq
        }

        shap_importance_df = pd.DataFrame(importance_data)

        # 按平均绝对SHAP值排序（这是最常用的SHAP重要性指标）
        shap_importance_df = shap_importance_df.sort_values(
            'shap_mean_abs',
            ascending=False
        ).reset_index(drop=True)

        # 添加排名
        shap_importance_df.insert(0, 'rank', range(1, len(shap_importance_df) + 1))

        # 计算累积重要性
        total_importance = shap_importance_df['shap_mean_abs'].sum()
        if total_importance > 0:
            shap_importance_df['cumulative_importance'] = (
                shap_importance_df['shap_mean_abs'].cumsum() / total_importance
            )
        else:
            shap_importance_df['cumulative_importance'] = 0.0

        # 计算贡献方向指标
        shap_importance_df['pos_neg_ratio'] = (
            shap_importance_df['shap_pos_mean'] /
            (np.abs(shap_importance_df['shap_neg_mean']) + 1e-10)
        )

        shap_importance_df['dominant_effect'] = np.where(
            np.abs(shap_importance_df['shap_pos_mean']) > np.abs(shap_importance_df['shap_neg_mean']),
            'Positive',
            'Negative'
        )

        # 重新排列列顺序
        column_order = [
            'rank', 'feature_name', 'shap_mean_abs', 'shap_mean_squared',
            'cumulative_importance', 'shap_mean', 'shap_std',
            'shap_pos_mean', 'shap_neg_mean', 'pos_neg_ratio',
            'dominant_effect', 'non_zero_freq'
        ]

        shap_importance_df = shap_importance_df[column_order]

        # 保存到CSV
        shap_importance_path = os.path.join(self.output_dir, 'shap_feature_importance.csv')
        shap_importance_df.to_csv(shap_importance_path, index=False, float_format='%.6f')
        print(f"SHAP特征重要性已保存到: {shap_importance_path}")

        # 打印前20个最重要的特征
        print(f"\n前20个最重要的特征 (按平均绝对SHAP值排序):")
        print(f"{'=' * 120}")
        top_20 = shap_importance_df.head(20)
        for _, row in top_20.iterrows():
            print(f"  {row['rank']:2d}. {row['feature_name']:20s} | "
                  f"SHAP均值: {row['shap_mean']:8.4f} | "
                  f"SHAP绝对值均值: {row['shap_mean_abs']:8.4f} | "
                  f"方向: {row['dominant_effect']:8s} | "
                  f"累积: {row['cumulative_importance']:.4f}")

        # 打印统计信息
        print(f"\nSHAP特征重要性统计:")
        print(f"{'=' * 60}")
        print(f"  总特征数: {len(shap_importance_df)}")
        print(f"  非零重要性特征数: {(shap_importance_df['shap_mean_abs'] > 0).sum()}")
        print(f"  最大绝对SHAP值: {shap_importance_df['shap_mean_abs'].max():.6f}")
        print(f"  平均绝对SHAP值: {shap_importance_df['shap_mean_abs'].mean():.6f}")
        print(f"  中位数绝对SHAP值: {shap_importance_df['shap_mean_abs'].median():.6f}")
        print(f"  总绝对SHAP值: {total_importance:.6f}")

        # 正负贡献统计
        total_pos = shap_importance_df['shap_pos_mean'].sum()
        total_neg = shap_importance_df['shap_neg_mean'].sum()
        print(f"\n贡献方向统计:")
        print(f"  总正贡献: {total_pos:.6f}")
        print(f"  总负贡献: {total_neg:.6f}")
        print(f"  净贡献: {(total_pos + total_neg):.6f}")
        print(f"  正主导特征数: {(shap_importance_df['dominant_effect'] == 'Positive').sum()}")
        print(f"  负主导特征数: {(shap_importance_df['dominant_effect'] == 'Negative').sum()}")

        # 累积重要性分析
        if total_importance > 0:
            thresholds = [0.5, 0.8, 0.9, 0.95, 0.99]
            print(f"\n累积重要性分析:")
            for threshold in thresholds:
                n_features = (shap_importance_df['cumulative_importance'] <= threshold).sum() + 1
                n_features = min(n_features, len(shap_importance_df))
                print(f"  达到 {threshold*100:.0f}% 累积重要性需要: {n_features} 个特征")

        # 保存SHAP值到文件（可选，如果需要进一步分析）
        shap_values_path = os.path.join(self.output_dir, 'shap_values.npy')
        np.save(shap_values_path, self.shap_values)
        print(f"\n原始SHAP值已保存到: {shap_values_path}")

        # 保存SHAP解释器
        if self.shap_explainer is not None:
            explainer_path = os.path.join(self.output_dir, 'shap_explainer.pkl')
            with open(explainer_path, 'wb') as f:
                pickle.dump(self.shap_explainer, f)
            print(f"SHAP解释器已保存到: {explainer_path}")

        return shap_importance_df

    def compute_xgboost_feature_importance(self) -> pd.DataFrame:
        """
        计算XGBoost内置的特征重要性（用于比较）

        Returns:
            包含XGBoost特征重要性的DataFrame
        """
        if self.best_model is None:
            return None

        # XGBoost提供三种重要性类型
        importance_types = {
            'weight': '特征被使用的次数',
            'gain': '特征带来的平均增益',
            'cover': '特征覆盖的样本数'
        }

        # 获取所有类型的特征重要性
        all_importances = {}
        for imp_type in importance_types.keys():
            try:
                all_importances[imp_type] = self.best_model.get_booster().get_score(
                    importance_type=imp_type
                )
            except:
                all_importances[imp_type] = {}

        # 创建特征重要性DataFrame
        xgb_importance_data = []

        for i, feature_name in enumerate(self.feature_names):
            # XGBoost内部使用f0, f1, f2...作为特征名
            xgb_feature_name = f"f{i}"

            row = {'feature_name': feature_name}

            # 添加所有类型的重要性
            for imp_type in importance_types.keys():
                row[f'xgb_importance_{imp_type}'] = all_importances[imp_type].get(xgb_feature_name, 0.0)

            xgb_importance_data.append(row)

        xgb_importance_df = pd.DataFrame(xgb_importance_data)

        # 按gain降序排列
        if 'xgb_importance_gain' in xgb_importance_df.columns:
            xgb_importance_df = xgb_importance_df.sort_values(
                'xgb_importance_gain',
                ascending=False
            ).reset_index(drop=True)

        return xgb_importance_df

    def compare_feature_importance_methods(self) -> None:
        """比较不同特征重要性方法的结果"""
        print(f"\n{'=' * 60}")
        print("9. 特征重要性方法比较")
        print(f"{'=' * 60}")

        # 获取SHAP重要性
        shap_importance_df = self.analyze_shap_feature_importance()
        if shap_importance_df is None:
            return

        # 获取XGBoost重要性
        xgb_importance_df = self.compute_xgboost_feature_importance()

        if xgb_importance_df is not None:
            # 合并两种重要性
            comparison_df = pd.merge(
                shap_importance_df[['feature_name', 'shap_mean_abs', 'rank']],
                xgb_importance_df[['feature_name', 'xgb_importance_gain']],
                on='feature_name',
                how='left'
            )

            # 计算XGBoost重要性的排名
            if 'xgb_importance_gain' in comparison_df.columns:
                comparison_df['xgb_rank'] = comparison_df['xgb_importance_gain'].rank(
                    ascending=False, method='min'
                ).astype(int)

                # 计算排名差异
                comparison_df['rank_difference'] = comparison_df['rank'] - comparison_df['xgb_rank']

                # 计算相关性
                valid_mask = (comparison_df['shap_mean_abs'] > 0) & (comparison_df['xgb_importance_gain'] > 0)
                if valid_mask.sum() > 5:  # 至少有5个有效特征
                    valid_data = comparison_df[valid_mask]

                    # 计算斯皮尔曼等级相关系数
                    from scipy.stats import spearmanr
                    corr, p_value = spearmanr(
                        valid_data['shap_mean_abs'],
                        valid_data['xgb_importance_gain']
                    )

                    print(f"\nSHAP vs XGBoost Gain 相关性:")
                    print(f"  斯皮尔曼等级相关系数: {corr:.4f}")
                    print(f"  P值: {p_value:.4f}")

                    if p_value < 0.05:
                        print(f"  相关性在5%水平上显著")

                    # 保存相关性结果
                    correlation_result = {
                        'spearman_correlation': float(corr),
                        'p_value': float(p_value),
                        'n_features_compared': int(valid_mask.sum()),
                        'interpretation': 'Spearman rank correlation between SHAP mean absolute values and XGBoost gain importance'
                    }

                    corr_path = os.path.join(self.output_dir, 'importance_correlation.json')
                    with open(corr_path, 'w') as f:
                        json.dump(correlation_result, f, indent=4)
                    print(f"  相关性结果已保存到: {corr_path}")

            # 保存比较结果
            comparison_path = os.path.join(self.output_dir, 'feature_importance_comparison.csv')
            comparison_df.to_csv(comparison_path, index=False, float_format='%.6f')
            print(f"特征重要性比较结果已保存到: {comparison_path}")

            # 打印排名差异最大的特征
            if 'rank_difference' in comparison_df.columns:
                print(f"\n排名差异最大的特征 (SHAP排名 - XGBoost Gain排名):")
                print(f"{'=' * 80}")

                # 正向差异（SHAP排名更靠前）
                top_positive_diff = comparison_df.nlargest(5, 'rank_difference')
                print(f"\nSHAP排名显著更靠前的特征:")
                for _, row in top_positive_diff.iterrows():
                    print(f"  {row['feature_name']:20s} | "
                          f"SHAP排名: {row['rank']:3d} | "
                          f"XGB排名: {row['xgb_rank']:3d} | "
                          f"差异: {row['rank_difference']:+3d}")

                # 负向差异（XGBoost排名更靠前）
                top_negative_diff = comparison_df.nsmallest(5, 'rank_difference')
                print(f"\nXGBoost排名显著更靠前的特征:")
                for _, row in top_negative_diff.iterrows():
                    print(f"  {row['feature_name']:20s} | "
                          f"SHAP排名: {row['rank']:3d} | "
                          f"XGB排名: {row['xgb_rank']:3d} | "
                          f"差异: {row['rank_difference']:+3d}")

    def evaluate_model(self) -> None:
        """在测试集上评估模型"""
        print(f"\n{'=' * 60}")
        print("10. 测试集评估")
        print(f"{'=' * 60}")

        # 预测
        y_pred = self.best_model.predict(self.X_test)

        # 计算指标
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        r2 = r2_score(self.y_test, y_pred)
        mae = mean_absolute_error(self.y_test, y_pred)

        print(f"\n测试集结果:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R²:   {r2:.4f}")
        print(f"  MAE:  {mae:.4f}")

        # 计算交叉验证的统计量
        cv_df = pd.DataFrame(self.cv_results)

        # 提取每个fold的验证集（测试集）指标
        fold_test_rmse_values = cv_df['val_rmse'].tolist()
        fold_test_r2_values = cv_df['val_r2'].tolist()
        fold_test_mae_values = cv_df['val_mae'].tolist()

        # 保存评估指标
        metrics = {
            # 最终测试集指标
            'test_rmse': float(rmse),
            'test_r2': float(r2),
            'test_mae': float(mae),

            # 交叉验证 - 训练集指标
            'cv_train_rmse_mean': float(cv_df['train_rmse'].mean()),
            'cv_train_rmse_std': float(cv_df['train_rmse'].std()),
            'cv_train_r2_mean': float(cv_df['train_r2'].mean()),
            'cv_train_r2_std': float(cv_df['train_r2'].std()),
            'cv_train_mae_mean': float(cv_df['train_mae'].mean()),
            'cv_train_mae_std': float(cv_df['train_mae'].std()),

            # 交叉验证 - 每个fold测试集指标 - 汇总统计
            'cv_fold_test_rmse_mean': float(cv_df['val_rmse'].mean()),
            'cv_fold_test_rmse_std': float(cv_df['val_rmse'].std()),
            'cv_fold_test_r2_mean': float(cv_df['val_r2'].mean()),
            'cv_fold_test_r2_std': float(cv_df['val_r2'].std()),
            'cv_fold_test_mae_mean': float(cv_df['val_mae'].mean()),
            'cv_fold_test_mae_std': float(cv_df['val_mae'].std()),

            # 交叉验证 - 每个fold测试集指标 - 详细数值
            'cv_fold_test_rmse_values': [float(x) for x in fold_test_rmse_values],
            'cv_fold_test_r2_values': [float(x) for x in fold_test_r2_values],
            'cv_fold_test_mae_values': [float(x) for x in fold_test_mae_values],

            # 数据集折数
            'n_folds': int(len(cv_df))
        }

        with open(os.path.join(self.output_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)

        print(f"所有评估指标已保存到: metrics.json")

        # 打印完整的metrics摘要
        print(f"\n{'=' * 60}")
        print("完整评估指标摘要:")
        print(f"{'=' * 60}")
        print("\n【最终测试集】")
        print(f"  RMSE: {metrics['test_rmse']:.4f}")
        print(f"  R²:   {metrics['test_r2']:.4f}")
        print(f"  MAE:  {metrics['test_mae']:.4f}")

        print("\n【交叉验证 - 训练集】")
        print(f"  RMSE: {metrics['cv_train_rmse_mean']:.4f} ± {metrics['cv_train_rmse_std']:.4f}")
        print(f"  R²:   {metrics['cv_train_r2_mean']:.4f} ± {metrics['cv_train_r2_std']:.4f}")
        print(f"  MAE:  {metrics['cv_train_mae_mean']:.4f} ± {metrics['cv_train_mae_std']:.4f}")

        print("\n【交叉验证 - 每个Fold测试集】")
        print(f"  RMSE: {metrics['cv_fold_test_rmse_mean']:.4f} ± {metrics['cv_fold_test_rmse_std']:.4f}")
        print(f"  R²:   {metrics['cv_fold_test_r2_mean']:.4f} ± {metrics['cv_fold_test_r2_std']:.4f}")
        print(f"  MAE:  {metrics['cv_fold_test_mae_mean']:.4f} ± {metrics['cv_fold_test_mae_std']:.4f}")

        print("\n【每个Fold详细RMSE】")
        for i, rmse_val in enumerate(fold_test_rmse_values, 1):
            print(f"  Fold {i}: {rmse_val:.4f}")

        print("\n【每个Fold详细R²】")
        for i, r2_val in enumerate(fold_test_r2_values, 1):
            print(f"  Fold {i}: {r2_val:.4f}")

    def save_model(self) -> None:
        """保存模型和预处理器"""
        print(f"\n{'=' * 60}")
        print("11. 保存模型")
        print(f"{'=' * 60}")

        # 保存模型
        model_path = os.path.join(self.output_dir, 'xgboost_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(self.best_model, f)
        print(f"已保存模型: {model_path}")

        # 保存标准化器
        scaler_path = os.path.join(self.output_dir, 'scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"已保存标准化器: {scaler_path}")

        # 保存特征名称
        feature_names_path = os.path.join(self.output_dir, 'feature_names.json')
        with open(feature_names_path, 'w') as f:
            json.dump(self.feature_names, f, indent=4)
        print(f"已保存特征名称: {feature_names_path}")

        # 保存Optuna study
        if self.study is not None:
            study_path = os.path.join(self.output_dir, 'optuna_study.pkl')
            with open(study_path, 'wb') as f:
                pickle.dump(self.study, f)
            print(f"已保存Optuna study: {study_path}")

    def run_full_pipeline(
            self,
            n_trials: int = 20,
            n_folds_cv: int = 10,
            shap_n_samples: int = 1000
    ) -> None:
        """
        运行完整的训练流程

        Args:
            n_trials: Optuna优化试验次数
            n_folds_cv: 交叉验证折数
            shap_n_samples: SHAP计算使用的背景样本数
        """
        print("=" * 60)
        print("XGBoost模型训练流程 (含SHAP特征重要性分析)")
        print("=" * 60)

        # 1. 加载数据
        self.load_data()

        # 2. 生成特征
        self.generate_features()

        # 3. 划分数据集
        self.split_data()

        # 4. Optuna优化
        best_params = self.optuna_optimization(n_trials=n_trials)

        # 5. 交叉验证
        self.cross_validation(best_params, n_folds=n_folds_cv)

        # 6. 训练最终模型
        self.train_final_model(best_params)

        # 7. 计算SHAP值
        self.compute_shap_values(n_samples=shap_n_samples)

        # 8. 分析SHAP特征重要性
        self.analyze_shap_feature_importance()

        # 9. 比较不同特征重要性方法
        self.compare_feature_importance_methods()

        # 10. 评估模型
        self.evaluate_model()

        # 11. 保存模型
        self.save_model()

        print(f"\n{'=' * 60}")
        print("训练完成!")
        print(f"{'=' * 60}")
        print(f"所有结果已保存到: {self.output_dir}")
        print(f"\n输出文件清单:")
        print(f"  - xgboost_model.pkl                   : 训练好的模型")
        print(f"  - scaler.pkl                          : 特征标准化器")
        print(f"  - feature_names.json                  : 特征名称列表")
        print(f"  - shap_feature_importance.csv         : SHAP特征重要性分析结果")
        print(f"  - feature_importance_comparison.csv   : 特征重要性方法比较")
        print(f"  - importance_correlation.json         : 重要性方法相关性分析")
        print(f"  - shap_values.npy                     : 原始SHAP值")
        print(f"  - shap_explainer.pkl                  : SHAP解释器")
        print(f"  - optuna_study.pkl                    : Optuna study对象")
        print(f"  - cross_validation_results.csv        : 交叉验证详细结果")
        print(f"  - metrics.json                        : 所有评估指标")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='XGBoost模型训练 (Optuna优化 + SHAP特征重要性分析)')
    parser.add_argument('--csv_file', type=str, required=True,
                        help='输入CSV文件路径')
    parser.add_argument('--smiles_column', type=str, default='smiles',
                        help='SMILES列名 (默认: smiles)')
    parser.add_argument('--target_column', type=str, default='target',
                        help='目标性质列名 (默认: target)')
    parser.add_argument('--features_generator', type=str, default='pfas',
                        help='特征生成器名称 (默认: pfas)')
    parser.add_argument('--output_dir', type=str, default='./results/XGB',
                        help='输出目录 (默认: ./results/XGB)')
    parser.add_argument('--n_trials', type=int, default=10,
                        help='Optuna优化试验次数 (默认: 10)')
    parser.add_argument('--n_folds_cv', type=int, default=10,
                        help='交叉验证折数 (默认: 10)')
    parser.add_argument('--shap_n_samples', type=int, default=1000,
                        help='SHAP计算使用的背景样本数 (默认: 1000)')
    parser.add_argument('--random_state', type=int, default=42,
                        help='随机种子 (默认: 42)')

    args = parser.parse_args()

    # 创建训练器并运行
    trainer = XGBoostModelTrainer(
        csv_file=args.csv_file,
        smiles_column=args.smiles_column,
        target_column=args.target_column,
        features_generator=args.features_generator,
        output_dir=args.output_dir,
        random_state=args.random_state
    )

    trainer.run_full_pipeline(
        n_trials=args.n_trials,
        n_folds_cv=args.n_folds_cv,
        shap_n_samples=args.shap_n_samples
    )


if __name__ == '__main__':
    main()