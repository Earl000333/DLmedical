import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier  # NEW: gating 网络用的分类器
from pygam import LinearGAM, s
import pickle
import time
from scipy import stats
import warnings
from sklearn.metrics import confusion_matrix

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class BloodPressureGAMEnsembleMOE:
    """
    使用 Mixture-of-Experts 思路的血压预测模型：
    - expert: 各个血压区间的 GAM 子模型
    - gate:   一个分类器，根据特征 X 预测 pressure_group 的概率，作为每个样本的权重
    """

    def __init__(self, input_dir, test_size=0.2, random_state=42):
        self.input_dir = input_dir
        self.test_size = test_size
        self.random_state = random_state

        # 特征与目标
        self.features = ['Age', 'alt', 'Height', 'Weight', 'Pulse', 'BMI']
        self.target = 'systolic pressure'

        # 各组件
        self.sub_models = {}          # {'90-100': {'model': GAM, 'scaler': RobustScaler, ...}, ...}
        self.model_weights = {}       # 仍然计算全局权重，仅作参考/分析，不再用于预测
        self.performance_metrics = {}
        self.group_data = {}
        self.train_groups = {}
        self.common_test_set = None
        self.train_data_combined = None  # NEW: gating 网络训练用的合并训练集

        # NEW: gating 网络相关
        self.gating_model = None
        self.gating_scaler = None
        self.gating_classes_ = None   # 保存分类器的类别顺序（即各 pressure_group 的顺序）

    # =========================
    # 一、数据读取与分组
    # =========================
    def list_all_data_files(self, input_dir):
        """列出所有 CSV 文件"""
        files = []
        for file in os.listdir(input_dir):
            if file.endswith('.csv'):
                files.append(os.path.join(input_dir, file))
        return files

    def group_files_by_pressure_range(self, files):
        """按血压范围分组文件"""
        groups = {
            '90-100': [],
            '100-110': [],
            '110-120': [],
            '120-130': [],
            '130-140': []
        }

        # 更灵活的正则表达式：匹配文件名中的 "90...100" 等
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
        """加载并合并同一血压区间的文件"""
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
        """准备各血压区间的数据"""
        print("正在加载和分组数据文件...")
        files = self.list_all_data_files(self.input_dir)
        file_groups = self.group_files_by_pressure_range(files)

        print(f"\n文件分组结果:")
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

    # =========================
    # 二、统一测试集 + 训练集按组拆分
    # =========================
    def create_common_test_set(self):
        """创建统一测试集，并按组划分训练集"""
        print("\n正在创建统一测试集...")

        all_data = []
        for group_name, data in self.group_data.items():
            data = data.copy()
            data['pressure_group'] = group_name
            all_data.append(data)

        combined_data = pd.concat(all_data, ignore_index=True)

        # 分层抽样，保证各组在测试集中比例大致相同
        train_data, test_data = train_test_split(
            combined_data,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=combined_data['pressure_group']
        )

        self.common_test_set = test_data
        self.train_data_combined = train_data  # NEW: gating 网络训练要用
        self.train_groups = {}

        for group_name in self.group_data.keys():
            group_train_data = train_data[train_data['pressure_group'] == group_name]
            if len(group_train_data) > 0:
                self.train_groups[group_name] = group_train_data.drop('pressure_group', axis=1)

        print(f"统一测试集大小: {len(self.common_test_set)}")
        for group_name, data in self.train_groups.items():
            print(f"{group_name} 训练集: {len(data)} 个样本")

    # =========================
    # 三、训练各个 GAM 子模型（experts）
    # =========================
    def train_sub_model(self, group_name, train_data):
        """训练单个血压区间的 GAM 子模型"""
        print(f"\n正在训练 {group_name} 子模型...")

        X = train_data[self.features].values
        y = train_data[self.target].values

        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)

        # 五折交叉验证，估计该子模型的 MAE
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

        # 在全部训练数据上重新训练最终模型
        final_gam = LinearGAM(s(0) + s(1) + s(2) + s(3) + s(4) + s(5))
        final_gam.fit(X_scaled, y)

        avg_mae = np.mean(cv_maes)
        print(f"{group_name} 平均验证MAE: {avg_mae:.4f}")

        return final_gam, avg_mae, scaler, cv_predictions, cv_true_values

    def calculate_weights(self, validation_maes, sample_sizes):
        """
        保留原来的“全局常数权重”计算，仅作参考/分析用。
        Mixture-of-Experts 预测阶段不再使用这些权重。
        """
        loss_weights = {group: 1.0 / mae for group, mae in validation_maes.items()}

        total_samples = sum(sample_sizes.values())
        sample_weights = {group: size / total_samples for group, size in sample_sizes.items()}

        combined_weights = {}
        for group in validation_maes.keys():
            combined_weights[group] = 0.5 * loss_weights[group] + 0.5 * sample_weights[group]

        total_weight = sum(combined_weights.values())
        normalized_weights = {group: weight / total_weight for group, weight in combined_weights.items()}

        print("\n[仅作参考] 全局权重计算详情:")
        for group in validation_maes.keys():
            print(f"  {group}:")
            print(f"    MAE: {validation_maes[group]:.4f}")
            print(f"    Loss权重: {loss_weights[group]:.4f}")
            print(f"    样本量权重: {sample_weights[group]:.4f}")
            print(f"    组合权重: {combined_weights[group]:.4f}")
            print(f"    归一化权重: {normalized_weights[group]:.4f}")

        return normalized_weights

    # =========================
    # 四、训练 gating 网络（gate）
    # =========================
    def train_gating_network(self):
        """
        训练一个分类器：输入特征 X，输出 belongs to 哪个 pressure_group 的概率分布。
        这里用 RandomForestClassifier，并设置 class_weight='balanced' 来缓解各组不均衡。
        """
        print("\n开始训练 gating 网络（压力区间分类器）...")

        df = self.train_data_combined.copy()
        X = df[self.features].values
        y = df['pressure_group'].values

        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)

        clf = RandomForestClassifier(
            n_estimators=200,
            random_state=self.random_state,
            class_weight='balanced',  # 样本不均衡时会自动给少数类更大权重
            n_jobs=-1
        )
        clf.fit(X_scaled, y)

        # 简单在训练集上看一下分类效果（只是参考）
        y_pred_train = clf.predict(X_scaled)
        acc = accuracy_score(y, y_pred_train)
        print(f"gating 网络在训练集上的准确率: {acc:.4f}")
        print("gating 网络训练集分类报告(仅供参考):")
        print(classification_report(y, y_pred_train))

        self.gating_model = clf
        self.gating_scaler = scaler
        self.gating_classes_ = clf.classes_  # 类别顺序，后面做加权时要用

    def debug_gating_on_test(self):
        """在统一测试集上评估 gating 网络分类性能，用来诊断不均衡问题"""
        if self.gating_model is None or self.common_test_set is None:
            print("gating 网络或测试集尚未准备好")
            return

        df = self.common_test_set.copy()
        X = df[self.features].values
        y_true = df['pressure_group'].values

        X_scaled = self.gating_scaler.transform(X)
        y_pred = self.gating_model.predict(X_scaled)

        print("\n=== gating 网络在测试集上的分类报告 ===")
        print(classification_report(y_true, y_pred))
        print("混淆矩阵（行是真实，列是预测）:")
        print(confusion_matrix(y_true, y_pred, labels=self.gating_classes_))
        print("类别顺序:", self.gating_classes_)


    # =========================
    # 五、训练整个 Mixture-of-Experts 模型
    # =========================
    def train_ensemble(self):
        """训练所有子模型 + gating 网络"""
        print("开始训练 GAM Mixture-of-Experts 模型...")

        self.prepare_data()
        self.create_common_test_set()

        validation_maes = {}
        sample_sizes = {}
        sub_model_info = {}

        # 先训练各个子模型（experts）
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
            sample_sizes[group_name] = len(train_data)

        # 计算“全局常数权重”（仅作参考，不参与最终预测）
        self.model_weights = self.calculate_weights(validation_maes, sample_sizes)
        self.sub_models = sub_model_info

        print("\n[仅作参考] 最终全局权重（预测时不再使用）:")
        for group, weight in self.model_weights.items():
            print(f"  {group}: {weight:.4f}")

        # NEW: 再训练 gating 网络
        self.train_gating_network()

    # =========================
    # 六、Mixture-of-Experts 预测
    # =========================
    def predict_ensemble(self, X):
        """
        Mixture-of-Experts 预测：
        1）gating 网络输出每个样本属于各 pressure_group 的概率分布 p(group | X)
        2）各子模型给出 y_hat_group(X)
        3）最终预测 = 对所有 group 的 y_hat_group(X) 按对应概率加权求和
        """
        if not self.sub_models:
            raise ValueError("子模型尚未训练，请先调用 train_ensemble() 方法")
        if self.gating_model is None:
            raise ValueError("gating 网络尚未训练，请先调用 train_ensemble() 方法")

        # 1. 通过 gating 网络得到每个样本的组概率
        X_gate_scaled = self.gating_scaler.transform(X)
        proba = self.gating_model.predict_proba(X_gate_scaled)  # 形状: [n_samples, n_groups]
        classes = list(self.gating_classes_)   # 类别顺序，例如 ['100-110', '110-120', ...]

        # === 关键改动：软化 + 掺一点均匀分布，防止单一专家一家独大 ===
        # 1) 温度缩放（temperature > 1 会让分布变“更平”）
        temperature = 2.0  # 可以调，比如 1.5 ~ 3
        proba = proba ** (1.0 / temperature)
        proba = proba / proba.sum(axis=1, keepdims=True)

        # 2) 和均匀分布混合，避免任何一个专家权重变成几乎 1
        alpha = 0.7  # 主观：gating 占 70%，均匀 prior 占 30%
        n_groups = proba.shape[1]
        uniform = np.ones_like(proba) / n_groups
        proba = alpha * proba + (1 - alpha) * uniform
        # === 改动结束 ===

        # 2. 各个子模型对同一批 X 给出预测
        group_preds = []  # 将来形状 [n_groups, n_samples]，顺序与 classes 一致
        for group_name in classes:
            if group_name not in self.sub_models:
                # 理论上不会出现，如果某组没有数据就不会作为类出现
                # 防御性编程：用全零预测占位
                print(f"警告: gating 输出的类别 {group_name} 在 sub_models 中不存在，使用 0 占位")
                group_preds.append(np.zeros(X.shape[0]))
                continue

            model_info = self.sub_models[group_name]
            model = model_info['model']
            scaler = model_info['scaler']

            X_scaled_group = scaler.transform(X)
            y_pred_group = model.predict(X_scaled_group)
            group_preds.append(y_pred_group)

        group_preds = np.vstack(group_preds)  # [n_groups, n_samples]
        proba_T = proba.T                     # [n_groups, n_samples]

        # 3. 按每个样本的概率分布进行加权平均
        weighted_predictions = np.sum(group_preds * proba_T, axis=0)
        return weighted_predictions

    # =========================
    # 七、评估与可视化
    # =========================
    def evaluate_model(self, save_dir='./results'):
        """评估模型性能（在统一测试集上）"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 先看 gating 在测试集上的效果
        self.debug_gating_on_test()
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

        print("\n=== Mixture-of-Experts 集成模型性能指标 ===")
        print(f"平均误差 (ME): {me:.4f} mmHg")
        print(f"残差标准差 (SD): {sd:.4f} mmHg")
        print(f"平均绝对误差 (MAE): {mae:.4f} mmHg")
        print(f"R² Score: {r2:.4f}")
        print(f"P5 (±5 mmHg): {p5:.2f}%")
        print(f"P10 (±10 mmHg): {p10:.2f}%")
        print(f"预测耗时: {prediction_time:.4f} 秒")

        # 附加：按区间看一下误差（方便你判断不均衡有没有改善）
        self.evaluate_by_group(y_pred)

        self.save_results(save_dir, X_test, y_test, y_pred, residuals)

        return self.performance_metrics

    def evaluate_by_group(self, y_pred):
        """在统一测试集上，按 pressure_group 拆分指标，查看少数区间是否被改善"""
        df = self.common_test_set.copy()
        df = df.reset_index(drop=True)
        df['pred'] = y_pred
        df['abs_err'] = (df[self.target] - df['pred']).abs()

        print("\n=== 按血压区间分组的误差统计（Mixture-of-Experts） ===")
        for group in df['pressure_group'].unique():
            sub = df[df['pressure_group'] == group]
            mae = sub['abs_err'].mean()
            p10 = (sub['abs_err'] <= 10).mean() * 100
            print(f"{group}: MAE={mae:.2f}, P10={p10:.2f}% (n={len(sub)})")

    def save_results(self, save_dir, X_test, y_test, y_pred, residuals):
        """保存各种结果与图表"""

        metrics_df = pd.DataFrame([self.performance_metrics])
        metrics_df.to_csv(os.path.join(save_dir, 'performance_metrics.csv'), index=False)

        results_df = pd.DataFrame({
            'Actual': y_test,
            'Predicted': y_pred,
            'Residual': residuals
        })
        results_df.to_csv(os.path.join(save_dir, 'predictions.csv'), index=False)

        # 这里保存的是“全局常数权重”，仅供对比原版用
        weights_df = pd.DataFrame(list(self.model_weights.items()), columns=['Pressure_Group', 'Global_Weight_Reference'])
        weights_df.to_csv(os.path.join(save_dir, 'model_weights_reference.csv'), index=False)

        with open(os.path.join(save_dir, 'gam_moe_model.pkl'), 'wb') as f:
            pickle.dump({
                'sub_models': self.sub_models,
                'model_weights_reference': self.model_weights,
                'performance_metrics': self.performance_metrics,
                'features': self.features,
                'target': self.target,
                'gating_model': self.gating_model,
                'gating_scaler': self.gating_scaler,
                'gating_classes_': self.gating_classes_
            }, f)

        self.create_plots(save_dir, y_test, y_pred, residuals)

        for group_name, model_info in self.sub_models.items():
            model_dir = os.path.join(save_dir, 'sub_models')
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)

            with open(os.path.join(model_dir, f'{group_name}_model.pkl'), 'wb') as f:
                pickle.dump(model_info, f)

    def create_plots(self, save_dir, y_test, y_pred, residuals):
        """创建常规统计图与 GAM 特征贡献图"""

        # 1. Bland-Altman
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
        plt.title('Bland-Altman Plot (Mixture-of-Experts)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'bland_altman_plot.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 2. R² 图
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'Predicted vs Actual (R² = {self.performance_metrics["R2"]:.4f})')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'r2_plot.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 3. 残差图
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(0, color='red', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residual Plot (Mixture-of-Experts)')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'residual_plot.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 4. GAM 特征贡献曲线
        self.plot_gam_contributions(save_dir)

        # 5. 参考用的“全局权重”分布
        plt.figure(figsize=(10, 6))
        groups = list(self.model_weights.keys())
        weights = list(self.model_weights.values())
        plt.bar(groups, weights)
        plt.xlabel('Pressure Groups')
        plt.ylabel('Global Weight (Reference Only)')
        plt.title('Reference Global Weight Distribution (Not Used in MoE Prediction)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'weight_distribution_reference.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_gam_contributions(self, save_dir):
        """绘制各血压区间中 GAM 对每个特征的贡献曲线"""
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


# 主执行函数
def main():
    # 修改为你自己的数据目录
    input_dir = r"E:\Pycharm project\MedLLM\男高压"
    results_dir = "./blood_pressure_results_moe"

    print("=== Blood Pressure Prediction GAM Mixture-of-Experts Model ===")
    bp_model = BloodPressureGAMEnsembleMOE(input_dir)

    # 训练集成模型（包含子模型 + gating 网络）
    bp_model.train_ensemble()

    # 评估性能
    metrics = bp_model.evaluate_model(results_dir)

    print(f"\nTraining completed! Results saved to: {results_dir}")


if __name__ == "__main__":
    main()
