# bp_lgbm_global_weighted.py
# 需要: pip install lightgbm

import os
import re
import time
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

import lightgbm as lgb

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class BloodPressureLGBMGlobalWeighted:
    """
    单一全局 LightGBM 模型 + 区间样本权重:
    - 所有区间数据合在一起训练一个 LGBMRegressor
    - 在训练时对不同 pressure_group 使用不同 sample_weight:
        样本量越少的区间, 权重越大
    - 目标: 在整体性能与各区间公平性之间取得折中
    """

    def __init__(self, input_dir, test_size=0.2, random_state=42, alpha=1.0):
        """
        alpha: 权重指数, weight_group ∝ 1 / (n_group^alpha)
               alpha 越大 => 越照顾少数区间
               你之前实验表明 alpha=1.0 在公平性上表现最好
        """
        self.input_dir = input_dir
        self.test_size = test_size
        self.random_state = random_state
        self.alpha = alpha

        self.features = ['Age', 'alt', 'Height', 'Weight', 'Pulse', 'BMI']
        self.target = 'systolic pressure'

        # 数据
        self.group_data = {}   # {'90-100': df, ...}
        self.train_data = None
        self.test_data = None

        # 模型相关
        self.scaler = None
        self.model = None
        self.sample_weights_train = None

        # 指标
        self.performance_metrics = {}

    # ========= 1. 文件读取与分组 =========
    def list_all_data_files(self, input_dir):
        files = []
        for file in os.listdir(input_dir):
            if file.endswith('.csv'):
                files.append(os.path.join(input_dir, file))
        return files

    def group_files_by_pressure_range(self, files):
        """根据文件名中的血压范围，将文件分组"""
        groups = {
            '90-100': [],
            '100-110': [],
            '110-120': [],
            '120-130': [],
            '130-140': []
        }

        patterns = {
            '90-100': re.compile(r"90\D*100"),
            '100-110': re.compile(r"100\D*110"),
            '110-120': re.compile(r"110\D*120"),
            '120-130': re.compile(r"120\D*130"),
            '130-140': re.compile(r"130\D*140")
        }

        for f in files:
            name = os.path.basename(f)
            for group_name, pattern in patterns.items():
                if pattern.search(name):
                    groups[group_name].append(f)
                    break

        for group_name in groups:
            groups[group_name].sort()

        return groups

    def load_and_combine_files(self, file_list, group_name):
        """加载并合并某个血压区间的所有文件"""
        if not file_list:
            print(f"警告: {group_name} 组没有找到文件")
            return None

        dfs = []
        for file in file_list:
            try:
                df = pd.read_csv(file)
                if all(col in df.columns for col in self.features + [self.target]):
                    dfs.append(df)
                    print(f"  已加载: {os.path.basename(file)} - {len(df)} 行")
                else:
                    missing_cols = [col for col in self.features + [self.target] if col not in df.columns]
                    print(f"警告: 文件 {os.path.basename(file)} 缺少列: {missing_cols}")
            except Exception as e:
                print(f"加载文件 {file} 时出错: {e}")

        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)
            print(f"  {group_name} 组合并后: {len(combined_df)} 行")
            return combined_df
        else:
            print(f"错误: {group_name} 组没有有效数据")
            return None

    def prepare_data(self):
        """加载所有文件，并按血压区间分组"""
        print("正在加载和分组数据文件...")
        files = self.list_all_data_files(self.input_dir)
        file_groups = self.group_files_by_pressure_range(files)

        print("\n文件分组结果:")
        for group_name, file_list in file_groups.items():
            print(f"{group_name}: {len(file_list)} 个文件")
            for file in file_list:
                print(f"  - {os.path.basename(file)}")

        self.group_data = {}
        for group_name, file_list in file_groups.items():
            print(f"\n正在处理 {group_name} 组...")
            data = self.load_and_combine_files(file_list, group_name)
            if data is not None and len(data) > 0:
                self.group_data[group_name] = data

        print(f"\n有效数据分组: {list(self.group_data.keys())}")
        for group_name, data in self.group_data.items():
            print(f"{group_name}: {len(data)} 个样本")

    # ========= 2. 统一训练/测试集 + 样本权重 =========
    def create_train_test_and_weights(self):
        """
        将各区间数据合并, 分层划分训练/测试集,
        并在训练集上根据各区间样本数计算 sample_weight
        """
        print("\n正在合并数据并创建统一训练/测试集...")

        all_data = []
        for group_name, data in self.group_data.items():
            data = data.copy()
            data['pressure_group'] = group_name
            all_data.append(data)

        combined_data = pd.concat(all_data, ignore_index=True)

        train_data, test_data = train_test_split(
            combined_data,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=combined_data['pressure_group']
        )

        self.train_data = train_data.reset_index(drop=True)
        self.test_data = test_data.reset_index(drop=True)

        print(f"训练集大小: {len(self.train_data)}")
        print(f"测试集大小: {len(self.test_data)}")

        print("\n训练集各区间样本数:")
        train_counts = self.train_data['pressure_group'].value_counts().sort_index()
        for g, n in train_counts.items():
            print(f"  {g}: {n}")

        # ===== 关键: 计算区间样本权重 =====
        group_counts = train_counts.to_dict()
        # weight_group ∝ 1 / (n_group^alpha)
        inv_counts = {g: 1.0 / (c ** self.alpha) for g, c in group_counts.items()}
        total_inv = sum(inv_counts.values())
        # 让各组权重和 ≈ num_groups
        group_weights_raw = {g: inv_counts[g] / total_inv * len(inv_counts) for g in inv_counts.keys()}

        print(f"\n区间样本权重 (alpha={self.alpha}, 样本越少权重越大):")
        for g in sorted(group_weights_raw.keys()):
            print(f"  {g}: weight = {group_weights_raw[g]:.4f}")

        # 为每个训练样本赋权重
        w = self.train_data['pressure_group'].map(group_weights_raw).values.astype(float)
        # 再归一化一次，让平均权重 = 1（可选）
        w = w * (len(w) / np.sum(w))
        self.sample_weights_train = w

        print(f"\n训练样本平均权重: {self.sample_weights_train.mean():.4f}")

    # ========= 3. 训练 LightGBM 模型 =========
    def train_model(self):
        print("\n开始训练 LightGBM 全局加权模型...")

        X_train = self.train_data[self.features].values
        y_train = self.train_data[self.target].values
        w_train = self.sample_weights_train

        # 标准化特征(对树模型不是必须, 但保持和之前一致)
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        self.scaler = scaler

        # 配置 LightGBM 回归模型
        self.model = lgb.LGBMRegressor(
            objective='regression_l1',  # 近似 MAE
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=self.random_state,
            n_jobs=-1
        )

        self.model.fit(
            X_train_scaled,
            y_train,
            sample_weight=w_train
        )

        print("LightGBM 模型训练完成。")

    # ========= 4. 预测 =========
    def predict(self, X):
        if self.model is None or self.scaler is None:
            raise ValueError("模型尚未训练，请先调用 train_model()")

        X_scaled = self.scaler.transform(X)
        y_pred = self.model.predict(X_scaled)
        return y_pred

    # ========= 5. 评估 =========
    def evaluate_model(self, save_dir='./results_lgbm_global_weighted'):
        """在统一测试集上评估模型，并按区间给出指标"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        print("\n正在评估 LightGBM 全局加权模型性能...")

        X_test = self.test_data[self.features].values
        y_test = self.test_data[self.target].values

        start_time = time.time()
        y_pred = self.predict(X_test)
        prediction_time = time.time() - start_time

        residuals = y_test - y_pred
        me = np.mean(residuals)
        sd = np.std(residuals)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        abs_errors = np.abs(residuals)
        p5 = np.mean(abs_errors <= 5) * 100
        p10 = np.mean(abs_errors <= 10) * 100

        self.performance_metrics = {
            'ME': me, 'SD': sd, 'MAE': mae, 'R2': r2,
            'P5': p5, 'P10': p10, 'Prediction_Time': prediction_time
        }

        print("\n=== LightGBM 全局加权模型性能指标 ===")
        print(f"平均误差 (ME): {me:.4f} mmHg")
        print(f"残差标准差 (SD): {sd:.4f} mmHg")
        print(f"平均绝对误差 (MAE): {mae:.4f} mmHg")
        print(f"R² Score: {r2:.4f}")
        print(f"P5 (±5 mmHg): {p5:.2f}%")
        print(f"P10 (±10 mmHg): {p10:.2f}%")
        print(f"预测耗时: {prediction_time:.4f} 秒")

        # 按血压区间拆分指标
        self.evaluate_by_group(y_pred)

        # 保存结果 & 图
        self.save_results(save_dir, X_test, y_test, y_pred, residuals)

        return self.performance_metrics

    def evaluate_by_group(self, y_pred):
        df = self.test_data.copy()
        df = df.reset_index(drop=True)
        df['pred'] = y_pred
        df['abs_err'] = (df[self.target] - df['pred']).abs()

        print("\n=== 按血压区间分组的误差统计（LightGBM 全局加权） ===")
        for group in df['pressure_group'].unique():
            sub = df[df['pressure_group'] == group]
            mae = sub['abs_err'].mean()
            p10 = (sub['abs_err'] <= 10).mean() * 100
            print(f"{group}: MAE={mae:.2f}, P10={p10:.2f}% (n={len(sub)})")

    # ========= 6. 结果保存与可视化 =========
    def save_results(self, save_dir, X_test, y_test, y_pred, residuals):
        # 1. 性能指标
        metrics_df = pd.DataFrame([self.performance_metrics])
        metrics_df.to_csv(os.path.join(save_dir, 'performance_metrics.csv'), index=False)

        # 2. 预测结果
        results_df = pd.DataFrame({
            'Actual': y_test,
            'Predicted': y_pred,
            'Residual': residuals
        })
        results_df.to_csv(os.path.join(save_dir, 'predictions.csv'), index=False)

        # 3. 训练样本权重分布
        train_weights_df = pd.DataFrame({
            'pressure_group': self.train_data['pressure_group'],
            'sample_weight': self.sample_weights_train
        })
        train_weights_df.to_csv(os.path.join(save_dir, 'train_sample_weights.csv'), index=False)

        # 4. 保存模型对象
        with open(os.path.join(save_dir, 'lgbm_global_weighted_model.pkl'), 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'performance_metrics': self.performance_metrics,
                'features': self.features,
                'target': self.target
            }, f)

        # 5. 画图
        self.create_plots(save_dir, y_test, y_pred, residuals)

    def create_plots(self, save_dir, y_test, y_pred, residuals):
        # Bland-Altman
        plt.figure(figsize=(10, 6))
        mean = (y_test + y_pred) / 2
        diff = y_test - y_pred
        mean_diff = np.mean(diff)
        std_diff = np.std(diff)

        plt.scatter(mean, diff, alpha=0.5)
        plt.axhline(mean_diff, color='red', linestyle='--', label=f'Mean: {mean_diff:.2f}')
        plt.axhline(mean_diff + 1.96 * std_diff, color='gray', linestyle='--',
                    label=f'+1.96SD: {mean_diff + 1.96 * std_diff:.2f}')
        plt.axhline(mean_diff - 1.96 * std_diff, color='gray', linestyle='--',
                    label=f'-1.96SD: {mean_diff - 1.96 * std_diff:.2f}')

        plt.xlabel('Mean (Actual + Predicted) / 2')
        plt.ylabel('Difference (Actual - Predicted)')
        plt.title('Bland-Altman Plot (LGBM Global Weighted)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'bland_altman_plot.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # R² 图
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'Predicted vs Actual (R² = {self.performance_metrics["R2"]:.4f})')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'r2_plot.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 残差图
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(0, color='red', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residual Plot (LGBM Global Weighted)')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'residual_plot.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 特征重要性
        self.plot_feature_importance(save_dir)

    def plot_feature_importance(self, save_dir):
        if self.model is None:
            return

        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        feature_names = np.array(self.features)[indices]

        plt.figure(figsize=(8, 6))
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), feature_names, rotation=45)
        plt.ylabel('Importance')
        plt.title('Feature Importances (LightGBM)')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'feature_importances.png'), dpi=300, bbox_inches='tight')
        plt.close()


def main():
    # 改成你的数据目录
    input_dir = r"E:\Pycharm project\MedLLM\男高压"
    results_dir = "./blood_pressure_results_lgbm_global_weighted"

    print("=== LightGBM Global Weighted Model for Blood Pressure Prediction ===")
    bp_model = BloodPressureLGBMGlobalWeighted(
        input_dir=input_dir,
        test_size=0.2,
        random_state=42,
        alpha=1.0   # 建议先用 1.0, 和你 GAM 实验保持一致
    )

    bp_model.prepare_data()
    bp_model.create_train_test_and_weights()
    bp_model.train_model()
    bp_model.evaluate_model(results_dir)

    print(f"\nTraining completed! Results saved to: {results_dir}")


if __name__ == "__main__":
    main()
