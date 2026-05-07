"""诊断充电预测输出为 0 的问题"""
import sys, os
import pickle
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
from config import ROOT_DIR, CHARGING_FEATURE_DIM

print("=" * 60)
print("1. 加载 scaler")
scaler_X_path = os.path.join(ROOT_DIR, "Charging_Forecast", "scaler_X.pkl")
scaler_y_path = os.path.join(ROOT_DIR, "Charging_Forecast", "scaler_y.pkl")

with open(scaler_X_path, 'rb') as f:
    scaler_X = pickle.load(f)
with open(scaler_y_path, 'rb') as f:
    scaler_y = pickle.load(f)

print(f"   scaler_X type: {type(scaler_X).__name__}")
print(f"   scaler_y type: {type(scaler_y).__name__}")
if hasattr(scaler_X, 'data_min_'):
    print(f"   scaler_X.data_min_: {scaler_X.data_min_}")
    print(f"   scaler_X.data_max_: {scaler_X.data_max_}")
    print(f"   scaler_X.feature_names_in_: {getattr(scaler_X, 'feature_names_in_', 'N/A')}")
if hasattr(scaler_y, 'data_min_'):
    print(f"   scaler_y.data_min_: {scaler_y.data_min_}")
    print(f"   scaler_y.data_max_: {scaler_y.data_max_}")
    print(f"   scaler_y.scale_: {getattr(scaler_y, 'scale_', 'N/A')}")
    print(f"   scaler_y.min_: {getattr(scaler_y, 'min_', 'N/A')}")

print("\n2. 测试 scaler_y.inverse_transform")
test_vals = np.array([[0.0], [0.5], [1.0]])
inv = scaler_y.inverse_transform(test_vals)
print(f"   输入: {test_vals.ravel()}")
print(f"   逆变换: {inv.ravel()}")
print(f"   输入 0.0 → {scaler_y.inverse_transform([[0.0]])[0,0]:.4f}")

print("\n3. 加载测试数据")
test_path = os.path.join(ROOT_DIR, "Data", "dataset_selected_features_test.csv")
df = pd.read_csv(test_path, parse_dates=['timestamp'])
df.sort_values('timestamp', inplace=True)
print(f"   shape: {df.shape}")
print(f"   列: {list(df.columns)}")
print(f"   日期范围: {df['timestamp'].min()} ~ {df['timestamp'].max()}")
print(f"   load_kw 统计: min={df['load_kw'].min():.1f}, max={df['load_kw'].max():.1f}, mean={df['load_kw'].mean():.1f}")

print("\n4. 测试特征标准化")
CHARGING_FEATURE_COLS = ['price', 'lag_1', 'lag_96', 'lag_672', 'rolling_std_4', 'rolling_mean_4']
raw = df[CHARGING_FEATURE_COLS].tail(96).values.astype(np.float32)
print(f"   原始特征统计:")
for i, col in enumerate(CHARGING_FEATURE_COLS):
    print(f"     {col}: min={raw[:, i].min():.4f}, max={raw[:, i].max():.4f}, mean={raw[:, i].mean():.4f}")

scaled = scaler_X.transform(raw)
print(f"\n   Scaled 特征统计:")
for i, col in enumerate(CHARGING_FEATURE_COLS):
    print(f"     {col}: min={scaled[:, i].min():.4f}, max={scaled[:, i].max():.4f}, mean={scaled[:, i].mean():.4f}")

# 查找可能的零值或异常值
print(f"\n   任何 scaled 列全为 0?")
for i, col in enumerate(CHARGING_FEATURE_COLS):
    if np.all(scaled[:, i] == 0):
        print(f"     ⚠️ {col} 全为 0!")
    elif np.std(scaled[:, i]) < 1e-6:
        print(f"     ⚠️ {col} 方差接近 0!")

print("\n5. 加载充电模型")
charge_pth = os.path.join(ROOT_DIR, "Charging_Forecast", "best_pth", "final_best_hybrid_model.pth")
from prediction_service import HybridModel, _infer_charging_hidden_size

hs = _infer_charging_hidden_size(charge_pth)
print(f"   hidden_size: {hs}")
model = HybridModel(input_dim=CHARGING_FEATURE_DIM, hidden_dim=hs, output_dim=1)
state = torch.load(charge_pth, map_location="cpu", weights_only=True)
model.load_state_dict(state)
model.eval()
print("   模型加载成功!")

print("\n6. 单步推理测试 (使用最后 96 步历史)")
with torch.no_grad():
    inp = torch.FloatTensor(scaled[-96:]).unsqueeze(0)
    print(f"   输入 shape: {inp.shape}")
    print(f"   输入统计: min={inp.min().item():.4f}, max={inp.max().item():.4f}, mean={inp.mean().item():.4f}")
    out = model(inp)
    print(f"   模型输出 (scaled): {out.item():.6f}")
    
print(f"\n   逆变换后: {scaler_y.inverse_transform([[out.item()]])[0,0]:.4f}")

print("\n7. 测试多个输入 (不同起始位置)")
for offset in [0, 1, 10, 50, 96, 200]:
    if len(df) < offset + 96:
        break
    win = df[CHARGING_FEATURE_COLS].iloc[-(offset+96):-(offset) if offset > 0 else None].values.astype(np.float32)
    if len(win) < 96:
        win = df[CHARGING_FEATURE_COLS].head(96).values.astype(np.float32)
    win_s = scaler_X.transform(win)
    with torch.no_grad():
        o = model(torch.FloatTensor(win_s[-96:]).unsqueeze(0))
    inv = scaler_y.inverse_transform([[o.item()]])[0,0]
    true = df['load_kw'].iloc[-(offset)] if offset > 0 else df['load_kw'].iloc[-1]
    print(f"   offset={offset:>4d}: scaled_out={o.item():.6f}, inverse={inv:.2f}, true_load={true:.2f}")

print("\n8. 检查模型内部各层输出")
with torch.no_grad():
    inp = torch.FloatTensor(scaled[-96:]).unsqueeze(0)
    x_tcn = model.tcn(inp.transpose(1,2)).transpose(1,2)
    print(f"   TCN 输出: min={x_tcn.min().item():.4f}, max={x_tcn.max().item():.4f}")
    x_attn = model.attention(x_tcn)
    print(f"   Attention 输出: min={x_attn.min().item():.4f}, max={x_attn.max().item():.4f}")
    _, (hn, _) = model.lstm(x_attn)
    lstm_out = hn[-1]
    print(f"   LSTM 最后隐状态: min={lstm_out.min().item():.4f}, max={lstm_out.max().item():.4f}")
    raw_last = inp[:, -1, :]
    combined = torch.cat([lstm_out, raw_last], dim=1)
    print(f"   Combined: min={combined.min().item():.4f}, max={combined.max().item():.4f}")
    fc_out = model.fc(combined)
    print(f"   FC 输出 (ReLU前): {fc_out.item():.6f}")
    relu_out = torch.relu(fc_out)
    print(f"   ReLU 后: {relu_out.item():.6f}")

print("\n" + "=" * 60)
print("诊断完成")