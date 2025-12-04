import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, r2_score
from pygam import LinearGAM, s
import pickle
import time
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class BloodPressureGAMEnsembleSimple:
    """
    简化版集成模型：
    - 不再使用 gating 网络和分类器
    - 每个血压区间训练一个 GAM 子模型
    - 集成权重只由该子模型的验证 MAE 决定：MAE 越小，权重越大
    - 统一测试集 + 按区间评估
    """

    def __init__(self, input_dir, test_size=0.2, random_state=42):
        self.input_dir = input_dir
        self.test_size = test_size
        self.random_state = random_state

        self.features = ['Age', 'alt', 'Height', 'Weight', 'Pulse', 'BMI']
        self.target = 'systolic pressure'

        self.sub_models = {}          # {group_name: {'model': gam, 'scaler': scaler, ...}}
        self.model_weights = {}       # {group_name: weight}
        self.performance_metrics = {}
        self.group_data = {}
        self.train_groups = {}
        self.common_test_set = None

    # ========= 1. 文件读取与分组 =========
    def list_all_data_files(self, input_dir):
        files = []
        for file in os.listdir(input_dir):
            if file.endswith('.csv'):
                files.append(os.path.join(input_dir, file))
        return files

    def group_files_by_pressure_range(self, files):
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
        """加载所有文件并按血压区间分组"""
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

    # ========= 2. 统一测试集 =========
    def create_common_test_set(self):
        """合并所有组数据，分层抽样划分统一测试集，训练集按组拆分"""
        print("\n正在创建统一测试集...")

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

        self.common_test_set = test_data
        self.train_groups = {}

        for group_name in self.group_data.keys():
            group_train_data = train_data[train_data['pressure_group'] == group_name]
            if len(group_train_data) > 0:
                self.train_groups[group_name] = group_train_data.drop('pressure_group', axis=1)

        print(f"统一测试集大小: {len(self.common_test_set)}")
        for group_name, data in self.train_groups.items():
            print(f"{group_name} 训练集: {len(data)} 个样本")

    # ========= 3. 子模型训练 =========
    def train_sub_model(self, group_name, train_data):
        print(f"\n正在训练 {group_name} 子模型...")

        X = train_data[self.features].values
        y = train_data[self.target].values

        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)

        kf = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        cv_maes = []
        cv_predictions = []
        cv_true_values = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled)):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            fold_gam = LinearGAM(s(0) + s(1) + s(2) + s(3) + s(4) + s(5))
            fold_gam.fit(X_train, y_train)

            y_val_pred = fold_gam.predict(X_val)
            mae = mean_absolute_error(y_val, y_val_pred)
            cv_maes.append(mae)

            cv_predictions.extend(y_val_pred)
            cv_true_values.extend(y_val)

            print(f"  {group_name} - 折 {fold + 1}: MAE = {mae:.4f}")

        final_gam = LinearGAM(s(0) + s(1) + s(2) + s(3) + s(4) + s(5))
        final_gam.fit(X_scaled, y)

        avg_mae = np.mean(cv_maes)
        print(f"{group_name} 平均验证MAE: {avg_mae:.4f}")

        return final_gam, avg_mae, scaler, cv_predictions, cv_true_values

    # ========= 4. 权重计算：只按 MAE =========
    def calculate_weights(self, validation_maes):
        """
        使用“误差越小权重越大”的原则：
        weight[group] = (1 / MAE[group]) / Σ(1 / MAE[k])
        不再使用样本量做权重（避免多数组自动权重大）。
        """
        inv_mae = {group: 1.0 / mae for group, mae in validation_maes.items()}
        total = sum(inv_mae.values())
        weights = {group: w / total for group, w in inv_mae.items()}

        print("\n基于 MAE 的模型权重:")
        for group in validation_maes.keys():
            print(f"  {group}: MAE={validation_maes[group]:.4f}, 权重={weights[group]:.4f}")

        return weights

    # ========= 5. 训练集成模型 =========
    def train_ensemble(self):
        print("开始训练基于 MAE 加权的 GAM 集成模型...")

        self.prepare_data()
        self.create_common_test_set()

        validation_maes = {}
        sub_model_info = {}

        for group_name, train_data in self.train_groups.items():
            gam_model, mae, scaler, cv_pred, cv_true = self.train_sub_model(group_name, train_data)

            sub_model_info[group_name] = {
                'model': gam_model,
                'scaler': scaler,
                'train_data': train_data,
                'cv_predictions': cv_pred,
                'cv_true_values': cv_true
            }
            validation_maes[group_name] = mae

        self.model_weights = self.calculate_weights(validation_maes)
        self.sub_models = sub_model_info

    # ========= 6. 集成预测 =========
    def predict_ensemble(self, X):
        """
        所有子模型都参与预测；
        权重为全局常数（基于各自验证 MAE），
        不依赖 gating，不做分类。
        """
        if not self.sub_models:
            raise ValueError("模型尚未训练，请先调用 train_ensemble() 方法")

        predictions = []
        weights = []

        for group_name, model_info in self.sub_models.items():
            model = model_info['model']
            scaler = model_info['scaler']

            X_scaled = scaler.transform(X)
            y_pred = model.predict(X_scaled)
            predictions.append(y_pred)
            weights.append(self.model_weights[group_name])

        predictions = np.array(predictions)          # [n_groups, n_samples]
        weights = np.array(weights).reshape(-1, 1)   # [n_groups, 1]

        weighted_predictions = np.sum(predictions * weights, axis=0)
        return weighted_predictions

    # ========= 7. 评估与按区间分析 =========
    def evaluate_model(self, save_dir='./results_mae_only'):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        print("\n正在评估模型性能...")

        X_test = self.common_test_set[self.features].values
        y_test = self.common_test_set[self.target].values

        start_time = time.time()
        y_pred = self.predict_ensemble(X_test)
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

        print("\n=== 基于 MAE 加权的集成模型性能指标 ===")
        print(f"平均误差 (ME): {me:.4f} mmHg")
        print(f"残差标准差 (SD): {sd:.4f} mmHg")
        print(f"平均绝对误差 (MAE): {mae:.4f} mmHg")
        print(f"R² Score: {r2:.4f}")
        print(f"P5 (±5 mmHg): {p5:.2f}%")
        print(f"P10 (±10 mmHg): {p10:.2f}%")
        print(f"预测耗时: {prediction_time:.4f} 秒")

        # 按血压区间看误差
        self.evaluate_by_group(y_pred)

        self.save_results(save_dir, X_test, y_test, y_pred, residuals)
        return self.performance_metrics

    def evaluate_by_group(self, y_pred):
        df = self.common_test_set.copy()
        df = df.reset_index(drop=True)
        df['pred'] = y_pred
        df['abs_err'] = (df[self.target] - df['pred']).abs()

        print("\n=== 按血压区间分组的误差统计（MAE 加权集成） ===")
        for group in df['pressure_group'].unique():
            sub = df[df['pressure_group'] == group]
            mae = sub['abs_err'].mean()
            p10 = (sub['abs_err'] <= 10).mean() * 100
            print(f"{group}: MAE={mae:.2f}, P10={p10:.2f}% (n={len(sub)})")

    # ========= 8. 保存结果与可视化 =========
    def save_results(self, save_dir, X_test, y_test, y_pred, residuals):
        metrics_df = pd.DataFrame([self.performance_metrics])
        metrics_df.to_csv(os.path.join(save_dir, 'performance_metrics.csv'), index=False)

        results_df = pd.DataFrame({
            'Actual': y_test,
            'Predicted': y_pred,
            'Residual': residuals
        })
        results_df.to_csv(os.path.join(save_dir, 'predictions.csv'), index=False)

        weights_df = pd.DataFrame(list(self.model_weights.items()), columns=['Pressure_Group', 'Weight'])
        weights_df.to_csv(os.path.join(save_dir, 'model_weights_mae_only.csv'), index=False)

        with open(os.path.join(save_dir, 'gam_ensemble_mae_only.pkl'), 'wb') as f:
            pickle.dump({
                'sub_models': self.sub_models,
                'model_weights': self.model_weights,
                'performance_metrics': self.performance_metrics,
                'features': self.features,
                'target': self.target
            }, f)

        self.create_plots(save_dir, y_test, y_pred, residuals)

        for group_name, model_info in self.sub_models.items():
            model_dir = os.path.join(save_dir, 'sub_models')
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)

            with open(os.path.join(model_dir, f'{group_name}_model.pkl'), 'wb') as f:
                pickle.dump(model_info, f)

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
        plt.title('Bland-Altman Plot (MAE-weighted Ensemble)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'bland_altman_plot.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # R²
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
        plt.title('Residual Plot (MAE-weighted Ensemble)')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'residual_plot.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # GAM 特征贡献曲线
        self.plot_gam_contributions(save_dir)

        # 权重分布
        plt.figure(figsize=(10, 6))
        groups = list(self.model_weights.keys())
        weights = list(self.model_weights.values())
        plt.bar(groups, weights)
        plt.xlabel('Pressure Groups')
        plt.ylabel('Weight (1/MAE normalized)')
        plt.title('Model Weights Based on MAE')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'weight_distribution_mae_only.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_gam_contributions(self, save_dir):
        gam_dir = os.path.join(save_dir, 'gam_contributions')
        if not os.path.exists(gam_dir):
            os.makedirs(gam_dir)

        for group_name, model_info in self.sub_models.items():
            gam_model = model_info['model']

            plt.figure(figsize=(15, 10))
            for i, feature in enumerate(self.features):
                plt.subplot(2, 3, i + 1)
                XX = gam_model.generate_X_grid(term=i)
                plt.plot(XX[:, i], gam_model.partial_dependence(term=i, X=XX))
                plt.plot(
                    XX[:, i],
                    gam_model.partial_dependence(term=i, X=XX, width=0.95)[1],
                    c='r', ls='--'
                )
                plt.title(f'{feature} - {group_name}')
                plt.xlabel(feature)
                plt.ylabel('Contribution to SBP')

            plt.tight_layout()
            plt.savefig(os.path.join(gam_dir, f'{group_name}_feature_contributions.png'),
                        dpi=300, bbox_inches='tight')
            plt.close()


def main():
    input_dir = r"E:\Pycharm project\MedLLM\男高压"  # 改成你的路径
    results_dir = "./blood_pressure_results_mae_only"

    print("=== Blood Pressure Prediction GAM Ensemble (MAE-weighted) ===")
    bp_model = BloodPressureGAMEnsembleSimple(input_dir)

    bp_model.train_ensemble()
    metrics = bp_model.evaluate_model(results_dir)

    print(f"\nTraining completed! Results saved to: {results_dir}")


if __name__ == "__main__":
    main()
