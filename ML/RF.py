"""
Random Forest Model Training with Optuna Optimization and Feature Importance Analysis
用于PFAS分子性质预测的完整训练流程（包含特征重要性分析）
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
import warnings

warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

import optuna
from optuna.samplers import TPESampler

# 导入特征生成器
from chemprop.features.features_generators import (
    get_features_generator,
    get_available_features_generators
)


class RandomForestModelTrainer:
    """Random Forest模型训练器,包含Optuna优化、交叉验证和特征重要性分析"""

    def __init__(
            self,
            csv_file: str,
            smiles_column: str = 'smiles',
            target_column: str = 'target',
            features_generator: str = 'pfas',
            output_dir: str = './results/RF',
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
        self.feature_names = None  # 新增：保存特征名称

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
            # 定义Random Forest超参数搜索空间
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'max_samples': trial.suggest_float('max_samples', 0.5, 1.0),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                'random_state': self.random_state,
                'n_jobs': -1
            }

            # 创建模型
            model = RandomForestRegressor(**params)

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
            study_name='random_forest_hyperparameter_optimization'
        )

        # 优化
        self.study.optimize(
            objective,
            n_trials=n_trials,
            show_progress_bar=True,
            n_jobs=1  # Optuna本身用单线程,模型内部用多线程
        )

        best_params = self.study.best_params
        best_params['random_state'] = self.random_state
        best_params['n_jobs'] = -1

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
            model = RandomForestRegressor(**params)
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
        self.best_model = RandomForestRegressor(**params)
        self.best_model.fit(X_train_full, y_train_full)

        print("模型训练完成!")

    def analyze_feature_importance(self) -> None:
        """分析并保存特征重要性"""
        print(f"\n{'=' * 60}")
        print("7. 特征重要性分析")
        print(f"{'=' * 60}")

        if self.best_model is None:
            print("错误: 模型尚未训练!")
            return

        # 获取特征重要性
        importances = self.best_model.feature_importances_

        # 创建特征重要性DataFrame
        feature_importance_df = pd.DataFrame({
            'feature_name': self.feature_names,
            'importance': importances
        })

        # 按重要性降序排列
        feature_importance_df = feature_importance_df.sort_values(
            'importance',
            ascending=False
        ).reset_index(drop=True)

        # 添加排名列
        feature_importance_df.insert(0, 'rank', range(1, len(feature_importance_df) + 1))

        # 添加累积重要性
        feature_importance_df['cumulative_importance'] = feature_importance_df['importance'].cumsum()

        # 保存为CSV
        importance_path = os.path.join(self.output_dir, 'feature_importance.csv')
        feature_importance_df.to_csv(importance_path, index=False)
        print(f"特征重要性已保存到: {importance_path}")

        # 打印前20个最重要的特征
        print(f"\n前20个最重要的特征:")
        print(f"{'=' * 60}")
        top_20 = feature_importance_df.head(20)
        for _, row in top_20.iterrows():
            print(f"  {row['rank']:2d}. {row['feature_name']:20s} | "
                  f"重要性: {row['importance']:.6f} | "
                  f"累积: {row['cumulative_importance']:.4f}")

        # 统计信息
        print(f"\n特征重要性统计:")
        print(f"{'=' * 60}")
        print(f"  总特征数: {len(feature_importance_df)}")
        print(f"  非零重要性特征数: {(importances > 0).sum()}")
        print(f"  最大重要性: {importances.max():.6f}")
        print(f"  平均重要性: {importances.mean():.6f}")
        print(f"  中位数重要性: {np.median(importances):.6f}")

        # 计算达到不同累积重要性阈值所需的特征数
        thresholds = [0.5, 0.8, 0.9, 0.95, 0.99]
        print(f"\n累积重要性分析:")
        for threshold in thresholds:
            n_features = (feature_importance_df['cumulative_importance'] <= threshold).sum() + 1
            print(f"  达到 {threshold*100:.0f}% 累积重要性需要: {n_features} 个特征")

    def evaluate_model(self) -> None:
        """在测试集上评估模型"""
        print(f"\n{'=' * 60}")
        print("8. 测试集评估")
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
        print("9. 保存模型")
        print(f"{'=' * 60}")

        # 保存模型
        model_path = os.path.join(self.output_dir, 'random_forest_model.pkl')
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
            n_folds_cv: int = 10
    ) -> None:
        """
        运行完整的训练流程

        Args:
            n_trials: Optuna优化试验次数
            n_folds_cv: 交叉验证折数
        """
        print("=" * 60)
        print("Random Forest模型训练流程 (含特征重要性分析)")
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

        # 7. 特征重要性分析
        self.analyze_feature_importance()

        # 8. 评估模型
        self.evaluate_model()

        # 9. 保存模型
        self.save_model()

        print(f"\n{'=' * 60}")
        print("训练完成!")
        print(f"{'=' * 60}")
        print(f"所有结果已保存到: {self.output_dir}")
        print(f"\n输出文件清单:")
        print(f"  - random_forest_model.pkl       : 训练好的模型")
        print(f"  - scaler.pkl                    : 特征标准化器")
        print(f"  - feature_names.json            : 特征名称列表")
        print(f"  - feature_importance.csv        : 特征重要性分析结果")
        print(f"  - optuna_study.pkl              : Optuna study对象")
        print(f"  - cross_validation_results.csv  : 交叉验证详细结果")
        print(f"  - metrics.json                  : 所有评估指标")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='Random Forest模型训练 (Optuna优化 + 特征重要性分析)')
    parser.add_argument('--csv_file', type=str, required=True,
                        help='输入CSV文件路径')
    parser.add_argument('--smiles_column', type=str, default='smiles',
                        help='SMILES列名 (默认: smiles)')
    parser.add_argument('--target_column', type=str, default='target',
                        help='目标性质列名 (默认: target)')
    parser.add_argument('--features_generator', type=str, default='pfas',
                        help='特征生成器名称 (默认: pfas)')
    parser.add_argument('--output_dir', type=str, default='./results/RF',
                        help='输出目录 (默认: ./results/RF)')
    parser.add_argument('--n_trials', type=int, default=10,
                        help='Optuna优化试验次数 (默认: 10)')
    parser.add_argument('--n_folds_cv', type=int, default=10,
                        help='交叉验证折数 (默认: 10)')
    parser.add_argument('--random_state', type=int, default=42,
                        help='随机种子 (默认: 42)')

    args = parser.parse_args()

    # 创建训练器并运行
    trainer = RandomForestModelTrainer(
        csv_file=args.csv_file,
        smiles_column=args.smiles_column,
        target_column=args.target_column,
        features_generator=args.features_generator,
        output_dir=args.output_dir,
        random_state=args.random_state
    )

    trainer.run_full_pipeline(
        n_trials=args.n_trials,
        n_folds_cv=args.n_folds_cv
    )


if __name__ == '__main__':
    main()