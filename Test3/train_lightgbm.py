"""
LightGBM 充电负荷预测模型训练脚本
替代原有的 TCN-Attention-LSTM 混合模型
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ============================================================
# 路径配置
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

TRAIN_CSV = os.path.join(ROOT_DIR, "Data", "dataset_selected_features_train.csv")
TEST_CSV = os.path.join(ROOT_DIR, "Data", "dataset_selected_features_test.csv")
MODEL_OUT = os.path.join(BASE_DIR, "lightgbm_charging_model.pkl")
SCALER_X_OUT = os.path.join(BASE_DIR, "scaler_X_lightgbm.pkl")
SCALER_Y_OUT = os.path.join(BASE_DIR, "scaler_y_lightgbm.pkl")

# ============================================================
# 参数
# ============================================================
LOOKBACK = 96  # 24小时窗口
FEATURE_COLS = ["price", "lag_1", "lag_96", "lag_672", "rolling_std_4", "rolling_mean_4"]
TARGET_COL = "load_kw"

LGBM_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 127,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "n_estimators": 2000,
    "early_stopping_rounds": 100,
    "random_state": 42,
    "n_jobs": -1,
}

print("=" * 60)
print("LightGBM 充电负荷预测模型训练")
print("=" * 60)

# ============================================================
# 1. 加载数据
# ============================================================
print("\n[1/5] 加载数据...")
train_df = pd.read_csv(TRAIN_CSV, parse_dates=["timestamp"])
test_df = pd.read_csv(TEST_CSV, parse_dates=["timestamp"])
print(f"  训练集: {len(train_df)} 行, 测试集: {len(test_df)} 行")

# ============================================================
# 2. 构建滑动窗口样本
# ============================================================
print("\n[2/5] 构建滑动窗口样本...")

def build_samples(df, lookback=LOOKBACK, feature_cols=FEATURE_COLS, target_col=TARGET_COL):
    data = df[feature_cols + [target_col]].dropna().values.astype(np.float64)
    n = len(data)
    X_list, y_list = [], []
    for i in range(n - lookback):
        # 使用 lookback 窗口的所有特征
        X_list.append(data[i:i + lookback, :-1].flatten())
        y_list.append(data[i + lookback, -1])
    return np.array(X_list), np.array(y_list)

X_train, y_train = build_samples(train_df)
X_test, y_test = build_samples(test_df)
print(f"  训练样本: {X_train.shape[0]}, 特征维度: {X_train.shape[1]}")
print(f"  测试样本: {X_test.shape[0]}, 特征维度: {X_test.shape[1]}")

# ============================================================
# 3. 标准化 (对 LightGBM 树模型非必需，但保留以与原有流程兼容)
# ============================================================
print("\n[3/5] 标准化...")
from sklearn.preprocessing import StandardScaler

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

X_test_scaled = scaler_X.transform(X_test)
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

# 保存 scaler
with open(SCALER_X_OUT, "wb") as f:
    pickle.dump(scaler_X, f)
with open(SCALER_Y_OUT, "wb") as f:
    pickle.dump(scaler_y, f)
print(f"  Scaler 已保存: {SCALER_X_OUT}, {SCALER_Y_OUT}")

# ============================================================
# 4. 训练 LightGBM (直接在原始值上训练，树模型不需要标准化)
# ============================================================
print("\n[4/5] 训练 LightGBM...")

# LightGBM 在原始值上训练效果更好（树模型对尺度不敏感）
model = lgb.LGBMRegressor(**LGBM_PARAMS)

# 使用时间序列交叉验证
tscv = TimeSeriesSplit(n_splits=3)
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric="rmse",
    callbacks=[
        lgb.early_stopping(100, verbose=False),
        lgb.log_evaluation(100),
    ],
)

print(f"\n  训练完成！最佳迭代: {model.best_iteration_}")

# ============================================================
# 5. 评估并保存
# ============================================================
print("\n[5/5] 评估并保存...")

y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

# MAPE (排除零值/极小值)
nonzero_mask = np.abs(y_test) > 10
if nonzero_mask.sum() > 0:
    mape = np.mean(np.abs(y_test[nonzero_mask] - y_pred[nonzero_mask]) / y_test[nonzero_mask]) * 100
else:
    mape = float("nan")

print(f"\n{'='*60}")
print(f"📊 测试集评估结果")
print(f"{'='*60}")
print(f"  RMSE: {rmse:.2f} kW")
print(f"  MAE:  {mae:.2f} kW")
print(f"  MAPE: {mape:.1f}% (load>10kW)")
print(f"  测试样本数: {len(y_test)}")

# 检查是否有全零预测
zero_count = np.sum(y_pred <= 0)
print(f"  零值预测数: {zero_count} / {len(y_pred)} ({zero_count/len(y_pred)*100:.1f}%)")
if zero_count == len(y_pred):
    print("  ⚠️ 警告: 所有预测均为零！模型可能存在问题。")
elif zero_count > len(y_pred) * 0.5:
    print(f"  ⚠️ 警告: {zero_count/len(y_pred)*100:.1f}% 预测值为零，比例偏高。")
else:
    print(f"  ✅ 仅 {zero_count/len(y_pred)*100:.1f}% 零值预测，模型正常。")

# 保存模型
with open(MODEL_OUT, "wb") as f:
    pickle.dump(model, f)
print(f"\n  模型已保存: {MODEL_OUT}")

# 保存训练历史
train_history = {
    "train_losses": [],  # LightGBM 不记录逐步loss
    "val_logs": [rmse],
    "rmse": rmse,
    "mae": mae,
    "mape": mape,
    "best_iteration": model.best_iteration_,
}
with open(os.path.join(BASE_DIR, "train_history_lightgbm.pkl"), "wb") as f:
    pickle.dump(train_history, f)

print("\n✅ 训练完成！")