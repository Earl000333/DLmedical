import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split, KFold
import warnings
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

warnings.filterwarnings('ignore')

# 设置样式
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['font.size'] = 12

# 1. 加载和预处理数据
print("Loading data...")
df = pd.read_csv('男1201.csv')

# 选择特征和目标变量
features = ['age', 'alt', 'height', 'weight', 'pulse', 'BMI', 'blood type']
target = 'diastolic pressure'

print(f"Original data shape: {df.shape}")
df = df[features + [target]].dropna()
print(f"Data shape after removing missing values: {df.shape}")

# 保存原始血型数据用于可视化
blood_type_original = df['blood type'].copy()

# 2. 编码分类特征（血型）
label_encoder = LabelEncoder()
df['blood_type_encoded'] = label_encoder.fit_transform(df['blood type'])

# 3. 鲁棒缩放（仅对连续特征）
continuous_features = ['age', 'alt', 'height', 'weight', 'pulse', 'BMI']
categorical_features = ['blood_type_encoded']

scaler = RobustScaler()
X_continuous = scaler.fit_transform(df[continuous_features])
X_continuous_df = pd.DataFrame(X_continuous, columns=continuous_features, index=df.index)

# 合并特征
X = pd.concat([X_continuous_df, df[categorical_features]], axis=1)
y = df[target]

# 4. 5折交叉验证
print("\n=== 5-FOLD CROSS VALIDATION WITH CATBOOST ===")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
results = []

fold = 1
for train_val_idx, test_idx in kf.split(X):
    print(f"\n--- Fold {fold} ---")

    # 分割数据
    X_train_val = X.iloc[train_val_idx]
    y_train_val = y.iloc[train_val_idx]

    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]

    # 进一步分割
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.125, random_state=42
    )

    print(f"Train: {X_train.shape[0]}, Validation: {X_val.shape[0]}, Test: {X_test.shape[0]}")

    try:
        # 导入CatBoost
        from catboost import CatBoostRegressor, Pool

        print("Training CatBoost model...")

        # 准备数据 - CatBoost可以自动处理类别特征
        # 我们需要指定哪些列是类别特征
        cat_features_indices = [X.columns.get_loc('blood_type_encoded')]

        # 创建训练和验证Pool
        train_pool = Pool(X_train.values, y_train.values, cat_features=cat_features_indices)
        val_pool = Pool(X_val.values, y_val.values, cat_features=cat_features_indices)

        # 设置CatBoost参数（优化速度）
        params = {
            'iterations': 500,  # 减少迭代次数以加速
            'learning_rate': 0.05,
            'depth': 6,
            'l2_leaf_reg': 3,
            'random_seed': 42,
            'verbose': 100,  # 每100次迭代输出一次
            'early_stopping_rounds': 50,
            'task_type': 'CPU',  # 明确使用CPU
            'bootstrap_type': 'Bernoulli',  # 加速训练
            'subsample': 0.8,
            'loss_function': 'MAE',  # 使用MAE作为损失函数
        }

        # 训练CatBoost模型
        cat_model = CatBoostRegressor(**params)
        cat_model.fit(
            train_pool,
            eval_set=val_pool,
            use_best_model=True,
            plot=False
        )

        # 获取最佳迭代次数
        best_iteration = cat_model.get_best_iteration()
        print(f"  Best iteration: {best_iteration}")
        print(f"  Best validation score: {cat_model.get_best_score():.4f}")

        # 预测测试集
        test_pool = Pool(X_test.values, cat_features=cat_features_indices)
        y_pred_test = cat_model.predict(test_pool)

        # 计算指标
        errors = y_test - y_pred_test
        me = np.mean(errors)
        sd = np.std(errors)
        mae = np.mean(np.abs(errors))
        abs_errors = np.abs(errors)
        p5 = np.percentile(abs_errors, 5)
        p10 = np.percentile(abs_errors, 10)

        # 存储结果
        fold_results = {
            'fold': fold,
            'me': me,
            'sd': sd,
            'mae': mae,
            'p5': p5,
            'p10': p10,
            'model': cat_model,
            'model_type': 'CatBoost',
            'y_test': y_test,
            'y_pred': y_pred_test,
            'X_test': X_test,
            'blood_type_test': blood_type_original.iloc[test_idx],
            'errors': errors.values if hasattr(errors, 'values') else errors,
            'best_iteration': best_iteration,
            'feature_importances': cat_model.get_feature_importance()
        }
        results.append(fold_results)

        print(f"CatBoost - ME: {me:.4f}, SD: {sd:.4f}, MAE: {mae:.4f}, P5: {p5:.4f}, P10: {p10:.4f}")

    except Exception as e:
        print(f"CatBoost failed: {e}")
        print("Falling back to Random Forest...")

        from sklearn.ensemble import RandomForestRegressor

        rf = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train.values, y_train.values)
        y_pred_test = rf.predict(X_test.values)

        errors = y_test - y_pred_test
        me = np.mean(errors)
        sd = np.std(errors)
        mae = np.mean(np.abs(errors))
        abs_errors = np.abs(errors)
        p5 = np.percentile(abs_errors, 5)
        p10 = np.percentile(abs_errors, 10)

        fold_results = {
            'fold': fold,
            'me': me,
            'sd': sd,
            'mae': mae,
            'p5': p5,
            'p10': p10,
            'model': rf,
            'model_type': 'Random Forest',
            'y_test': y_test,
            'y_pred': y_pred_test,
            'X_test': X_test,
            'blood_type_test': blood_type_original.iloc[test_idx],
            'errors': errors.values if hasattr(errors, 'values') else errors,
            'is_fallback': True
        }
        results.append(fold_results)
        print(f"Random Forest - ME: {me:.4f}, SD: {sd:.4f}, MAE: {mae:.4f}")

    fold += 1

# 5. 改进的可视化（包含3D误差图）
print("\n=== GENERATING IMPROVED VISUALIZATIONS ===")

# 使用最佳性能的fold
if len(results) > 0:
    best_fold_idx = np.argmin([r['mae'] for r in results])
    demo_results = results[best_fold_idx]
    y_test_demo = demo_results['y_test']
    y_pred_demo = demo_results['y_pred']

    fig = plt.figure(figsize=(20, 16))

    # 5.1 散点图
    plt.subplot(3, 3, 1)
    plt.scatter(y_test_demo, y_pred_demo, alpha=0.7, color='steelblue', s=30, edgecolor='white', linewidth=0.3)
    plt.plot([y_test_demo.min(), y_test_demo.max()], [y_test_demo.min(), y_test_demo.max()],
             'r--', lw=2, alpha=0.8)
    plt.xlabel('True Diastolic Pressure', fontsize=12, fontweight='bold')
    plt.ylabel('Predicted Diastolic Pressure', fontsize=12, fontweight='bold')
    plt.title('True vs Predicted Values', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    # 添加R²
    from sklearn.metrics import r2_score

    r2 = r2_score(y_test_demo, y_pred_demo)
    plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes,
             fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # 5.2 残差图
    plt.subplot(3, 3, 2)
    residuals = y_test_demo - y_pred_demo
    plt.scatter(y_pred_demo, residuals, alpha=0.7, color='forestgreen', s=30, edgecolor='white', linewidth=0.3)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.8)
    plt.xlabel('Predicted Values', fontsize=12, fontweight='bold')
    plt.ylabel('Residuals', fontsize=12, fontweight='bold')
    plt.title('Residual Plot', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    # 添加残差统计
    residual_mean = np.mean(residuals)
    residual_std = np.std(residuals)
    plt.text(0.05, 0.95, f'Mean: {residual_mean:.3f}\nStd: {residual_std:.3f}',
             transform=plt.gca().transAxes, fontsize=11,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # 5.3 血型对预测的影响
    plt.subplot(3, 3, 3)
    blood_type_test = demo_results['blood_type_test']
    blood_types = sorted(blood_type_test.unique())
    blood_type_errors = []

    for bt in blood_types:
        mask = blood_type_test == bt
        if mask.sum() > 0:
            bt_errors = np.abs(y_test_demo[mask] - y_pred_demo[mask])
            blood_type_errors.append((bt, np.mean(bt_errors), bt_errors))

    # 箱线图显示不同血型的误差
    if blood_type_errors:
        error_data = [errors for _, _, errors in blood_type_errors]
        blood_type_labels = [f"{label}\n(n={len(errors)})" for label, _, errors in blood_type_errors]

        box = plt.boxplot(error_data, labels=blood_type_labels, patch_artist=True)
        colors = plt.cm.Set3(np.linspace(0, 1, len(blood_type_labels)))
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)

        plt.xlabel('Blood Type', fontsize=12, fontweight='bold')
        plt.ylabel('Absolute Error', fontsize=12, fontweight='bold')
        plt.title('Prediction Error by Blood Type', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')

    # 5.4 模型类型分布
    plt.subplot(3, 3, 4)
    model_types = [r.get('model_type', 'Unknown') for r in results]
    model_counts = pd.Series(model_types).value_counts()

    colors = plt.cm.Paired(np.linspace(0, 1, len(model_counts)))
    plt.pie(model_counts.values, labels=model_counts.index, autopct='%1.1f%%',
            colors=colors, startangle=90, textprops={'fontsize': 10})
    plt.title('Model Types Used in 5-Fold CV', fontsize=14, fontweight='bold')

    # 5.5 性能指标对比
    plt.subplot(3, 3, 5)
    fold_numbers = [r['fold'] for r in results]
    me_values = [r['me'] for r in results]
    mae_values = [r['mae'] for r in results]

    x = np.arange(len(fold_numbers))
    width = 0.35

    plt.bar(x - width / 2, me_values, width, label='Mean Error (ME)', alpha=0.8, color='lightcoral')
    plt.bar(x + width / 2, mae_values, width, label='Mean Absolute Error (MAE)', alpha=0.8, color='lightblue')

    plt.xlabel('Fold Number', fontsize=12, fontweight='bold')
    plt.ylabel('Error Value', fontsize=12, fontweight='bold')
    plt.title('Performance Comparison Across Folds', fontsize=14, fontweight='bold')
    plt.xticks(x, fold_numbers)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')

    # 5.6 误差分布图
    plt.subplot(3, 3, 6)
    abs_errors = np.abs(y_test_demo - y_pred_demo)
    plt.hist(abs_errors, bins=30, alpha=0.8, color='orange', density=True, edgecolor='black')
    plt.axvline(np.mean(abs_errors), color='red', linestyle='--', linewidth=2,
                label=f'MAE: {np.mean(abs_errors):.2f}')

    # 添加误差分布信息
    error_stats = f'Mean: {np.mean(abs_errors):.2f}\nStd: {np.std(abs_errors):.2f}\n95th: {np.percentile(abs_errors, 95):.2f}'
    plt.text(0.95, 0.95, error_stats, transform=plt.gca().transAxes,
             fontsize=10, ha='right', va='top',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.xlabel('Absolute Error', fontsize=12, fontweight='bold')
    plt.ylabel('Density', fontsize=12, fontweight='bold')
    plt.title('Absolute Error Distribution', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 5.7 累积误差分布
    plt.subplot(3, 3, 7)
    sorted_errors = np.sort(abs_errors)
    cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    plt.plot(sorted_errors, cumulative, linewidth=3, color='green', alpha=0.8)
    plt.axhline(0.9, color='red', linestyle='--', alpha=0.7, label='90%')
    plt.axhline(0.95, color='blue', linestyle='--', alpha=0.7, label='95%')

    # 找到90%和95%分位数对应的误差
    error_90 = np.percentile(abs_errors, 90)
    error_95 = np.percentile(abs_errors, 95)
    plt.axvline(error_90, color='red', linestyle=':', alpha=0.5)
    plt.axvline(error_95, color='blue', linestyle=':', alpha=0.5)

    plt.text(error_90, 0.5, f'90%: {error_90:.2f}', rotation=90, va='center', fontsize=10)
    plt.text(error_95, 0.5, f'95%: {error_95:.2f}', rotation=90, va='center', fontsize=10)

    plt.xlabel('Absolute Error', fontsize=12, fontweight='bold')
    plt.ylabel('Cumulative Probability', fontsize=12, fontweight='bold')
    plt.title('Cumulative Error Distribution', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # # 5.8 3D误差分布图（替代Bland-Altman图）
    # plt.subplot(3, 3, 8, projection='3d')
    # ax3d = plt.gca()
    #
    # # 使用实际误差数据
    # errors = demo_results['errors']
    # mean_error = np.mean(errors)
    # std_error = np.std(errors)
    # abs_errors = np.abs(errors)
    #
    # # 准备3D数据
    # # 我们将创建一个误差密度山
    # n_bins = 30
    # error_range = np.linspace(errors.min(), errors.max(), n_bins)
    # pred_range = np.linspace(y_pred_demo.min(), y_pred_demo.max(), n_bins)
    #
    # # 创建网格
    # X_grid, Y_grid = np.meshgrid(error_range, pred_range)
    #
    # # 计算核密度估计
    # from scipy.stats import gaussian_kde
    #
    # # 为可视化效率，使用部分数据
    # sample_size = min(1000, len(errors))
    # sample_idx = np.random.choice(len(errors), sample_size, replace=False)
    # errors_sample = errors[sample_idx]
    # pred_sample = y_pred_demo.values[sample_idx] if hasattr(y_pred_demo, 'values') else y_pred_demo[sample_idx]
    #
    # # 计算2D核密度估计
    # positions = np.vstack([X_grid.ravel(), Y_grid.ravel()])
    # values = np.vstack([errors_sample, pred_sample])
    # kernel = gaussian_kde(values)
    # Z = kernel(positions).reshape(X_grid.shape)
    #
    # # 绘制3D曲面
    # surf = ax3d.plot_surface(X_grid, Y_grid, Z, cmap=cm.viridis,
    #                          linewidth=0, antialiased=True, alpha=0.8)
    #
    # # 添加等高线到底部
    # ax3d.contour(X_grid, Y_grid, Z, zdir='z', offset=Z.min() * 0.5,
    #              cmap=cm.viridis, alpha=0.5)
    #
    # # 设置坐标轴标签
    # ax3d.set_xlabel('Prediction Error', fontsize=10, fontweight='bold', labelpad=10)
    # ax3d.set_ylabel('Predicted Value', fontsize=10, fontweight='bold', labelpad=10)
    # ax3d.set_zlabel('Error Density', fontsize=10, fontweight='bold', labelpad=10)
    #
    # # 添加标题
    # ax3d.set_title('3D Error Distribution Analysis', fontsize=13, fontweight='bold')
    #
    # # 调整视角
    # ax3d.view_init(elev=25, azim=-60)
    #
    # # 添加颜色条
    # fig.colorbar(surf, ax=ax3d, shrink=0.5, aspect=5, pad=0.1)
    #
    # # 添加统计信息注解
    # stats_text = f'Mean Error: {mean_error:.2f}\nStd Dev: {std_error:.2f}\nMAE: {mae:.2f}'
    # ax3d.text2D(0.05, 0.95, stats_text, transform=ax3d.transAxes,
    #             fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # 5.8 改进版 Bland-Altman：Hexbin 密度图
    plt.subplot(3, 3, 8)

    # Bland-Altman 核心变量
    true_vals = y_test_demo.values if hasattr(y_test_demo, 'values') else np.array(y_test_demo)
    pred_vals = y_pred_demo.values if hasattr(y_pred_demo, 'values') else np.array(y_pred_demo)

    mean_vals = (true_vals + pred_vals) / 2.0           # 横轴：真值与预测值的平均
    diff_vals = true_vals - pred_vals                   # 纵轴：差值（残差）

    mean_diff = np.mean(diff_vals)
    sd_diff = np.std(diff_vals)
    upper = mean_diff + 1.96 * sd_diff
    lower = mean_diff - 1.96 * sd_diff

    # Hexbin 密度图：解决点过于密集的问题
    hb = plt.hexbin(
        mean_vals,
        diff_vals,
        gridsize=40,          # 网格细一点/粗一点可以自己调
        cmap='viridis',
        mincnt=1
    )
    cbar = plt.colorbar(hb)
    cbar.set_label('Count', fontsize=10)

    # 画出均值线和 95% 一致性界限
    plt.axhline(mean_diff, color='red', linestyle='--', linewidth=1.5, label=f'Mean: {mean_diff:.2f}')
    plt.axhline(upper, color='orange', linestyle='--', linewidth=1.2, label=f'+1.96 SD ({upper:.2f})')
    plt.axhline(lower, color='orange', linestyle='--', linewidth=1.2, label=f'-1.96 SD ({lower:.2f})')

    # 计算在 95% 区间内/外的比例
    within_mask = (diff_vals >= lower) & (diff_vals <= upper)
    within_pct = within_mask.mean() * 100
    outside_pct = 100 - within_pct

    # 在图中注释比例信息
    text_str = (
        f'Within 95% LoA: {within_pct:.1f}%\n'
        f'Outside 95% LoA: {outside_pct:.1f}%\n'
        f'n = {len(diff_vals)}'
    )
    plt.text(
        0.02, 0.98, text_str,
        transform=plt.gca().transAxes,
        va='top', ha='left',
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8)
    )

    plt.xlabel('Mean of True and Predicted DP', fontsize=10, fontweight='bold')
    plt.ylabel('True - Predicted', fontsize=10, fontweight='bold')
    plt.title('Bland-Altman with Hexbin Density', fontsize=13, fontweight='bold')
    plt.legend(loc='lower left', fontsize=8)
    plt.grid(True, alpha=0.3)


    # 5.9 整体性能指标
    plt.subplot(3, 3, 9)
    overall_results = {
        'me': np.mean([r['me'] for r in results]),
        'mae': np.mean([r['mae'] for r in results]),
        'sd': np.mean([r['sd'] for r in results]),
        'p5': np.mean([r['p5'] for r in results]),
        'p10': np.mean([r['p10'] for r in results])
    }

    metrics = ['ME', 'MAE', 'SD', 'P5', 'P10']
    values = [overall_results['me'], overall_results['mae'], overall_results['sd'],
              overall_results['p5'], overall_results['p10']]
    colors = ['lightcoral', 'lightblue', 'lightgreen', 'lightyellow', 'lightpink']

    bars = plt.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black')
    plt.ylabel('Value', fontsize=12, fontweight='bold')
    plt.title('Overall Performance Metrics', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')

    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{value:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig('CatBoost_Analysis_with_3D_Errors.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

    # 6. 打印详细结果
    print("\n=== DETAILED FOLD-WISE PERFORMANCE ===")
    performance_df = pd.DataFrame({
        'Fold': [r['fold'] for r in results],
        'Model_Type': [r.get('model_type', 'Unknown') for r in results],
        'ME': [r['me'] for r in results],
        'SD': [r['sd'] for r in results],
        'MAE': [r['mae'] for r in results],
        'P5': [r['p5'] for r in results],
        'P10': [r['p10'] for r in results],
        'Best_Iteration': [r.get('best_iteration', 'N/A') for r in results]
    })

    print(performance_df.round(4))

    print("\n=== OVERALL PERFORMANCE SUMMARY ===")
    print(f"Mean Error (ME): {overall_results['me']:.4f}")
    print(f"Standard Deviation (SD): {overall_results['sd']:.4f}")
    print(f"Mean Absolute Error (MAE): {overall_results['mae']:.4f}")
    print(f"5th Percentile of Abs Error (P5): {overall_results['p5']:.4f}")
    print(f"10th Percentile of Abs Error (P10): {overall_results['p10']:.4f}")

    print("\n=== DATA SUMMARY ===")
    print(f"Total samples: {len(df)}")
    print(f"Continuous features: {continuous_features}")
    print(f"Categorical feature: blood_type")
    print(f"Blood types: {sorted(blood_type_original.unique())}")

    # 保存结果
    performance_df.to_csv('CatBoost_Model_Performance_Detailed.csv', index=False)
    print("\nResults saved to 'CatBoost_Model_Performance_Detailed.csv'")
    print("Visualization saved as 'CatBoost_Analysis_with_3D_Errors.png'")

    # 7. CatBoost特定分析
    print("\n=== CATBOOST SPECIFIC ANALYSIS ===")

    # 选择最佳模型进行分析
    best_fold_result = results[best_fold_idx]
    best_model = best_fold_result['model']

    if 'CatBoost' in best_fold_result.get('model_type', ''):
        try:
            # 7.1 特征重要性分析
            print("\n1. Feature Importance Analysis:")

            importances = best_fold_result.get('feature_importances', [])
            if len(importances) > 0:
                feature_names = list(X.columns)

                # 创建特征重要性DataFrame
                feature_importance = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances
                }).sort_values('Importance', ascending=False)

                print(f"Feature Importance (Fold {best_fold_result['fold']}):")
                print(feature_importance.to_string(index=False))

                # 可视化特征重要性
                plt.figure(figsize=(12, 6))
                bars = plt.barh(range(len(feature_importance)),
                                feature_importance['Importance'][::-1],
                                color=plt.cm.Blues(np.linspace(0.3, 0.8, len(feature_importance))))

                plt.yticks(range(len(feature_importance)), feature_importance['Feature'][::-1])
                plt.xlabel('Feature Importance (PredictionValuesChange)', fontsize=12, fontweight='bold')
                plt.title('CatBoost Feature Importance', fontsize=14, fontweight='bold')
                plt.grid(True, alpha=0.3, axis='x')

                # 添加重要性数值
                for i, (bar, importance) in enumerate(zip(bars, feature_importance['Importance'][::-1])):
                    plt.text(bar.get_width(), i, f'{importance:.1f}',
                             ha='left', va='center', fontsize=9, fontweight='bold')

                plt.tight_layout()
                plt.savefig('CatBoost_Feature_Importance.png', dpi=300, bbox_inches='tight')
                plt.show()

            # 7.2 学习曲线分析
            print("\n2. Learning Curve Analysis:")

            # 获取CatBoost的训练历史
            if hasattr(best_model, 'get_evals_result'):
                evals_result = best_model.get_evals_result()

                if evals_result and 'validation' in evals_result:
                    plt.figure(figsize=(10, 6))

                    # 提取训练过程中的MAE
                    train_mae = evals_result.get('learn', {}).get('MAE', [])
                    val_mae = evals_result.get('validation', {}).get('MAE', [])

                    if val_mae:
                        epochs = range(1, len(val_mae) + 1)
                        plt.plot(epochs, val_mae, 'r-', linewidth=2, label='Validation MAE')

                    if train_mae:
                        epochs = range(1, len(train_mae) + 1)
                        plt.plot(epochs, train_mae, 'b-', linewidth=2, alpha=0.5, label='Training MAE')

                    plt.xlabel('Iterations', fontsize=12, fontweight='bold')
                    plt.ylabel('Mean Absolute Error (MAE)', fontsize=12, fontweight='bold')
                    plt.title('CatBoost Learning Curve', fontsize=14, fontweight='bold')
                    plt.legend()
                    plt.grid(True, alpha=0.3)

                    plt.tight_layout()
                    plt.savefig('CatBoost_Learning_Curve.png', dpi=300, bbox_inches='tight')
                    plt.show()

                    if val_mae:
                        print(f"Final validation MAE: {val_mae[-1]:.4f}")
                        print(f"Best validation MAE: {min(val_mae):.4f} at iteration {np.argmin(val_mae) + 1}")

            # 7.3 创建高级3D误差可视化
            print("\n3. Advanced 3D Error Visualization:")

            # 创建独立的3D误差半球图
            fig = plt.figure(figsize=(14, 10))

            # 3D半球图 - 展示误差分布
            ax1 = fig.add_subplot(121, projection='3d')

            # 创建误差的极坐标表示
            errors = demo_results['errors']
            n_points = min(200, len(errors))  # 限制点数以提高性能
            sample_indices = np.random.choice(len(errors), n_points, replace=False)
            errors_sample = errors[sample_indices]
            abs_errors_sample = np.abs(errors_sample)

            # 转换为极坐标
            theta = np.linspace(0, 2 * np.pi, n_points)  # 角度
            r = (abs_errors_sample - abs_errors_sample.min()) / (
                        abs_errors_sample.max() - abs_errors_sample.min() + 1e-10)
            z = r * 0.5  # 高度

            # 转换为笛卡尔坐标
            x = r * np.cos(theta)
            y = r * np.sin(theta)

            # 根据误差大小和方向着色
            colors = np.where(errors_sample >= 0, 'red', 'blue')  # 红色为正误差，蓝色为负误差
            sizes = 20 + 50 * r  # 大小与误差绝对值成正比

            # 绘制3D散点图
            scatter = ax1.scatter(x, y, z, c=colors, s=sizes, alpha=0.7, depthshade=True)

            # 添加参考平面
            xx, yy = np.meshgrid(np.linspace(-1, 1, 10), np.linspace(-1, 1, 10))
            zz = np.zeros_like(xx)
            ax1.plot_surface(xx, yy, zz, alpha=0.2, color='gray')

            # 设置坐标轴
            ax1.set_xlabel('X (Error Magnitude)', fontsize=10, labelpad=10)
            ax1.set_ylabel('Y (Error Magnitude)', fontsize=10, labelpad=10)
            ax1.set_zlabel('Z (Error Height)', fontsize=10, labelpad=10)
            ax1.set_title('3D Error Hemisphere', fontsize=12, fontweight='bold')

            # 添加图例
            from matplotlib.lines import Line2D

            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                       markersize=10, label='Positive Error (Pred < True)'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='blue',
                       markersize=10, label='Negative Error (Pred > True)')
            ]
            ax1.legend(handles=legend_elements, loc='upper right', fontsize=9)

            # 3D误差密度图
            ax2 = fig.add_subplot(122, projection='3d')

            # 创建误差的网格数据
            n_grid = 20
            x_grid = np.linspace(-std_error * 3, std_error * 3, n_grid)
            y_grid = np.linspace(y_pred_demo.min(), y_pred_demo.max(), n_grid)
            X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

            # 计算2D核密度
            positions = np.vstack([X_grid.ravel(), Y_grid.ravel()])
            errors_sample_2d = errors[:min(500, len(errors))]
            pred_sample_2d = y_pred_demo.values[:min(500, len(errors))] if hasattr(y_pred_demo,
                                                                                   'values') else y_pred_demo[:min(500,
                                                                                                                   len(errors))]
            values_2d = np.vstack([errors_sample_2d, pred_sample_2d])
            kernel_2d = gaussian_kde(values_2d)
            Z_grid = kernel_2d(positions).reshape(X_grid.shape)

            # 绘制3D曲面
            surf = ax2.plot_surface(X_grid, Y_grid, Z_grid, cmap=cm.coolwarm,
                                    linewidth=0, antialiased=True, alpha=0.8)

            # 设置坐标轴
            ax2.set_xlabel('Prediction Error', fontsize=10, labelpad=10)
            ax2.set_ylabel('Predicted Value', fontsize=10, labelpad=10)
            ax2.set_zlabel('Density', fontsize=10, labelpad=10)
            ax2.set_title('3D Error Density Surface', fontsize=12, fontweight='bold')

            # 调整视角
            ax1.view_init(elev=20, azim=-60)
            ax2.view_init(elev=25, azim=-45)

            plt.suptitle('Advanced 3D Error Analysis for CatBoost Model', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig('CatBoost_Advanced_3D_Error_Analysis.png', dpi=300, bbox_inches='tight')
            plt.show()

            print("Advanced 3D visualizations saved as 'CatBoost_Advanced_3D_Error_Analysis.png'")

        except Exception as e:
            print(f"CatBoost specific analysis failed: {e}")

    print("\n=== ANALYSIS COMPLETE ===")
    print("CatBoost model has been successfully implemented with 3D error visualization!")

else:
    print("\nNo successful results to visualize. Please check your data and model.")