"""
预测服务：加载模型、执行预测、生成策略建议
=============================
充电与光伏预测共用此服务。v3 改动:
  - 移除上传依赖，直接使用内置数据集作为"后台实时同步数据"
  - 充电预测改为用户输入当前电价+负荷，追加到历史窗口末尾后迭代预测
  - 光伏预测保持原有逻辑
"""
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")


# PyTorch
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
    print("[INFO] PyTorch 可用")
except ImportError:
    TORCH_AVAILABLE = False
    print("[WARN] PyTorch 不可用，将使用模拟预测")

from config import (
    SOLAR_MODEL_PTH, CHARGING_MODEL_PTH,
    SOLAR_LOOKBACK, CHARGING_LOOKBACK,
    SOLAR_FEATURE_COLS, MC_DROPOUT_SAMPLES,
    SOLAR_FEATURE_DIM, CHARGING_FEATURE_DIM,
    BASE_DIR, ROOT_DIR,
    DATA_ALIGNED, DATA_SELECTED_TEST,
)

# ============================================================
# 充电模型类定义 (与 Bys-TCN-Attention-LSTM_model.ipynb 一致)
# ============================================================
if TORCH_AVAILABLE:
    class TCNBlock(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
            super().__init__()
            self.padding = (kernel_size - 1) * dilation
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                                   padding=self.padding, dilation=dilation)
            self.dropout = nn.Dropout(dropout)
            self.relu = nn.ReLU()
            self.residual = nn.Conv1d(in_channels, out_channels, 1) \
                if in_channels != out_channels else nn.Identity()

        def forward(self, x):
            out = self.conv(x)
            out = out[:, :, :-(self.conv.padding[0])]
            out = self.relu(out)
            out = self.dropout(out)
            return out + self.residual(x)

    class SimpleAttention(nn.Module):
        def __init__(self, hidden_dim):
            super().__init__()
            self.attn_weights = nn.Parameter(torch.randn(hidden_dim, 1))

        def forward(self, x):
            scores = torch.matmul(x, self.attn_weights).squeeze(-1)
            weights = torch.softmax(scores, dim=1).unsqueeze(-1)
            return x * weights

    class HybridModel(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim=1):
            super().__init__()
            channels = [input_dim, 64, 64, 64, 64, 64]
            dilations = [1, 2, 4, 8, 16]
            tcn_layers = []
            for i in range(len(dilations)):
                tcn_layers.append(
                    TCNBlock(channels[i], channels[i+1], kernel_size=3,
                             dilation=dilations[i])
                )
            self.tcn = nn.Sequential(*tcn_layers)
            self.attention = SimpleAttention(64)
            self.lstm = nn.LSTM(64, hidden_dim, batch_first=True,
                                 num_layers=1, dropout=0.1)
            self.fc = nn.Linear(hidden_dim + input_dim, output_dim)

        def forward(self, x):
            raw_last_step = x[:, -1, :]
            x_tcn = x.transpose(1, 2)
            x_tcn = self.tcn(x_tcn).transpose(1, 2)
            x_attn = self.attention(x_tcn)
            _, (hn, _) = self.lstm(x_attn)
            lstm_out = hn[-1]
            combined = torch.cat([lstm_out, raw_last_step], dim=1)
            return torch.relu(self.fc(combined))


# ============================================================
# 光伏模型类定义 (与 Solar_Forecast/NN.py 一致)
# ============================================================
if TORCH_AVAILABLE:
    class LSTMPredictor(nn.Module):
        def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1,
                     dropout=0.2, bidirectional=False):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.bidirectional = bidirectional
            self.lstm = nn.LSTM(
                input_size=input_size, hidden_size=hidden_size,
                num_layers=num_layers, batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional
            )
            direction_factor = 2 if bidirectional else 1
            self.fc = nn.Linear(hidden_size * direction_factor, output_size)

        def forward(self, x):
            out, _ = self.lstm(x)
            last_step = out[:, -1, :]
            return self.fc(last_step)

    class GeneratorWithFeatures(nn.Module):
        def __init__(self, lstm_model):
            super().__init__()
            self.lstm_model = lstm_model

        def forward(self, x):
            return self.lstm_model(x)


# ============================================================
# Scaler 加载
# ============================================================

def _load_charging_scalers():
    """直接加载训练时保存的 scaler，不做重拟合。"""
    import pickle
    scaler_X_path = os.path.join(ROOT_DIR, "Charging_Retraining", "scaler_X.pkl")
    scaler_y_path = os.path.join(ROOT_DIR, "Charging_Retraining", "scaler_y.pkl")
    if not os.path.exists(scaler_X_path) or not os.path.exists(scaler_y_path):
        print("[WARN] 充电 scaler 文件不存在, 将使用模拟预测")
        return None, None
    with open(scaler_X_path, 'rb') as f:
        scaler_X = pickle.load(f)
    with open(scaler_y_path, 'rb') as f:
        scaler_y = pickle.load(f)
    print("[OK] 充电 scaler 加载成功 (训练时)")
    return scaler_X, scaler_y


def _load_solar_scaler():
    """加载光伏模型 scaler (MinMaxScaler)"""
    import joblib
    scaler_path = os.path.join(ROOT_DIR, "Solar_Forecast", "simple_scaler_1.pkl")
    if not os.path.exists(scaler_path):
        print("[WARN] 光伏 scaler 文件不存在, 将使用模拟预测")
        return None
    scaler = joblib.load(scaler_path)
    print("[OK] 光伏 scaler 加载成功")
    return scaler


# ============================================================
# 模型缓存与参数
# ============================================================
_solar_model = None
_solar_loaded = False
_charging_model = None
_charging_loaded = False
DEVICE = "cpu"
_CHARGING_INPUT_DIM = 6
_CHARGING_HIDDEN_SIZE = 121
_SOLAR_HIDDEN_SIZE = 128
_SOLAR_NUM_LAYERS = 2

# 充电特征列(训练时使用)
CHARGING_FEATURE_COLS = ['price', 'lag_1', 'lag_96', 'lag_672', 'rolling_std_4', 'rolling_mean_4']

# 光伏特征列(训练时使用)
SOLAR_FEATURE_COLS_7 = [
    'power', 'hour_sin', 'hour_cos',
    'shortwave_radiation (W/m2)', 'direct_radiation (W/m2)',
    'diffuse_radiation (W/m2)', 'direct_normal_irradiance (W/m2)'
]

if TORCH_AVAILABLE:
    DEVICE = "cpu"
    print(f"[INFO] 预测设备: {DEVICE}")


# ============================================================
# 模型加载
# ============================================================
def _infer_charging_hidden_size(pth_path):
    if not TORCH_AVAILABLE or not os.path.exists(pth_path):
        return _CHARGING_HIDDEN_SIZE
    state = torch.load(pth_path, map_location="cpu", weights_only=True)
    for k in state:
        if "lstm.weight_ih_l0" in k:
            hidden = state[k].shape[0] // 4
            print(f"[INFO] 充电模型 hidden_size 推断为: {hidden}")
            return hidden
    return _CHARGING_HIDDEN_SIZE


def load_solar_model():
    global _solar_model, _solar_loaded
    if _solar_loaded:
        return _solar_model
    if not TORCH_AVAILABLE:
        return None
    try:
        if not os.path.exists(SOLAR_MODEL_PTH):
            print(f"[WARN] 光伏模型文件不存在: {SOLAR_MODEL_PTH}")
            return None
        state = torch.load(SOLAR_MODEL_PTH, map_location=DEVICE, weights_only=True)
        lstm = LSTMPredictor(
            input_size=SOLAR_FEATURE_DIM, hidden_size=_SOLAR_HIDDEN_SIZE,
            num_layers=_SOLAR_NUM_LAYERS, output_size=1,
            dropout=0.2, bidirectional=False,
        )
        model = GeneratorWithFeatures(lstm)
        model.load_state_dict(state)
        model.to(DEVICE)
        model.eval()
        _solar_model = model
        _solar_loaded = True
        print("[OK] 光伏模型加载成功 (LSTM h128)")
        return _solar_model
    except Exception as e:
        print(f"[WARN] 光伏模型加载失败: {e}")
        return None


def load_charging_model():
    global _charging_model, _charging_loaded
    if _charging_loaded:
        return _charging_model
    if not TORCH_AVAILABLE:
        return None
    try:
        if not os.path.exists(CHARGING_MODEL_PTH):
            print(f"[WARN] 充电模型文件不存在: {CHARGING_MODEL_PTH}")
            return None
        hs = _infer_charging_hidden_size(CHARGING_MODEL_PTH)
        model = HybridModel(input_dim=_CHARGING_INPUT_DIM, hidden_dim=hs, output_dim=1)
        state = torch.load(CHARGING_MODEL_PTH, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state)
        model.to(DEVICE)
        model.eval()
        _charging_model = model
        _charging_loaded = True
        print(f"[OK] 充电模型加载成功 (TCN-Attention-LSTM h{hs})")
        return _charging_model
    except Exception as e:
        print(f"[WARN] 充电模型加载失败: {e}")
        return None


# ============================================================
# 历史窗口构建 (v3: 移除上传依赖，直接使用内置数据)
# ============================================================
def _load_aligned_df():
    """加载 aligned CSV 并做特征工程 — 直接使用内置数据作为"后台同步"数据"""
    aligned_path = os.path.join(ROOT_DIR, "Data", "aligned_2026_01_02.csv")
    if not os.path.exists(aligned_path):
        return None
    df = pd.read_csv(aligned_path, parse_dates=['datetime'])
    df.sort_values('datetime', inplace=True)
    df['hour'] = df['datetime'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    diffuse = df['diffuse_radiation (W/m2)'].values
    dni = df['direct_normal_irradiance (W/m2)'].values
    dni_safe = np.where(np.abs(dni) < 1e-6, 1.0, dni)
    cos_theta = (df['shortwave_radiation (W/m2)'].values - diffuse) / dni_safe
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    df['theta'] = np.arccos(cos_theta)
    return df


def _build_solar_window(df, seq_len, scaler):
    """从 aligned CSV 构建光伏模型输入窗口 (7 特征, MinMaxScaled)"""
    if df is None or len(df) < seq_len:
        return None
    raw = df[SOLAR_FEATURE_COLS_7].tail(seq_len).values.astype(np.float32)
    if scaler is not None:
        raw = scaler.transform(raw)
    return raw


def _load_charging_df():
    """加载充电历史数据 — 直接使用内置测试集作为"后台实时同步数据" """
    test_path = os.path.join(ROOT_DIR, "Data", "dataset_selected_features_test.csv")
    if not os.path.exists(test_path):
        return None
    df = pd.read_csv(test_path, parse_dates=['timestamp'])
    df.sort_values('timestamp', inplace=True)
    return df


def _build_charging_window(df, seq_len, scaler_X):
    """从充电数据构建充电模型输入窗口 (6 特征, MinMaxScaled)"""
    if df is None or len(df) < seq_len:
        return None
    raw = df[CHARGING_FEATURE_COLS].tail(seq_len).values.astype(np.float32)
    if scaler_X is not None:
        raw = scaler_X.transform(raw)
    return raw


# ============================================================
# 模拟预测 (fallback)
# ============================================================
def _mock_prediction(n_steps, weather):
    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    times = [now + timedelta(minutes=15 * i) for i in range(n_steps)]
    solar_base = weather.get("radiation", 400) if weather else 400
    np.random.seed(42)
    hours = np.array([t.hour + t.minute / 60 for t in times])
    solar = np.zeros(n_steps)
    day_mask = (hours >= 6) & (hours <= 18)
    solar[day_mask] = solar_base * np.sin(np.pi * (hours[day_mask] - 6) / 12) * 1.2
    solar[solar < 0] = 0
    solar += np.random.randn(n_steps) * 5
    load = np.zeros(n_steps)
    load += 40 + 30 * np.exp(-0.5 * ((hours - 9) / 2) ** 2)
    load += 30 * np.exp(-0.5 * ((hours - 18) / 2) ** 2)
    load += 10 * np.exp(-0.5 * ((hours - 13) / 3) ** 2)
    load += np.random.randn(n_steps) * 3
    total_solar = np.sum(solar) * 0.25
    total_load = np.sum(load) * 0.25
    green_ratio = total_solar / total_load * 100 if total_load > 0 else 100
    solar_peak_idx = np.argmax(solar)
    load_peak_idx = np.argmax(load)
    return {
        "times": times, "solar": solar,
        "load_mean": load, "load_lower": load * 0.85, "load_upper": load * 1.15,
        "total_solar": total_solar, "total_load": total_load,
        "green_ratio": green_ratio,
        "solar_peak": solar[solar_peak_idx],
        "solar_peak_time": times[solar_peak_idx],
        "load_peak": load[load_peak_idx],
        "model_status": {"solar_ok": False, "charging_ok": False},
    }


# ============================================================
# 真实预测 (v3: 充电预测改为 input-driven)
# ============================================================
def _real_prediction(n_steps, weather, current_price=None, current_load=None):
    now = datetime.now()
    times = [now + timedelta(minutes=15 * i) for i in range(n_steps)]

    solar_model = load_solar_model()
    charging_model = load_charging_model()
    solar_ok = solar_model is not None
    charging_ok = charging_model is not None

    if solar_ok:
        solar = _predict_solar(solar_model, n_steps, weather)
    else:
        solar = _fallback_solar(n_steps, weather)

    if charging_ok:
        load_mean, load_lower, load_upper = _predict_charging(
            charging_model, n_steps, current_price=current_price, current_load=current_load
        )
    else:
        res = _fallback_charging(n_steps)
        load_mean, load_lower, load_upper = res["mean"], res["lower"], res["upper"]

    total_solar = np.sum(solar) * 0.25
    total_load = np.sum(load_mean) * 0.25
    green_ratio = total_solar / total_load * 100 if total_load > 0 else 100
    solar_peak_idx = np.argmax(solar)
    load_peak_idx = np.argmax(load_mean)

    return {
        "times": times, "solar": solar,
        "load_mean": load_mean, "load_lower": load_lower, "load_upper": load_upper,
        "total_solar": total_solar, "total_load": total_load,
        "green_ratio": green_ratio,
        "solar_peak": solar[solar_peak_idx],
        "solar_peak_time": times[solar_peak_idx],
        "load_peak": load_mean[load_peak_idx],
        "model_status": {"solar_ok": solar_ok, "charging_ok": charging_ok},
    }


def _predict_solar(model, n_steps, weather):
    """光伏迭代预测 (使用 scaler + 7 特征 + aligned CSV 历史数据)"""
    scaler = _load_solar_scaler()
    df = _load_aligned_df()

    seq_len = SOLAR_LOOKBACK  # 24

    if df is not None and scaler is not None:
        window = _build_solar_window(df, seq_len, scaler)
    else:
        window = np.random.randn(seq_len, 7).astype(np.float32) * 0.1

    if window is None:
        window = np.random.randn(seq_len, 7).astype(np.float32) * 0.1

    if len(window) < seq_len:
        pad = seq_len - len(window)
        window = np.vstack([np.zeros((pad, 7), dtype=np.float32), window])

    window = window[-seq_len:]

    now = datetime.now()
    current_hour_fractional = now.hour + now.minute / 60.0

    outputs = []
    for i in range(n_steps):
        inp = torch.FloatTensor(window[-seq_len:]).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred_scaled = model(inp).item()

        pred_row = window[-1].copy()
        pred_row[0] = pred_scaled
        pred_real = scaler.inverse_transform(pred_row.reshape(1, -1))[0, 0]
        pred_real = max(pred_real, 0.0)

        outputs.append(pred_real)

        hour_fractional = (current_hour_fractional + (i + 1) * 0.25) % 24
        next_feat = np.zeros(7, dtype=np.float32)

        power_scaled = (pred_real - scaler.data_min_[0]) / (scaler.data_max_[0] - scaler.data_min_[0] + 1e-8)
        next_feat[0] = power_scaled
        next_feat[1] = np.sin(2 * np.pi * hour_fractional / 24)
        next_feat[2] = np.cos(2 * np.pi * hour_fractional / 24)

        if weather and weather.get("radiation") is not None:
            rad = weather["radiation"]
            if 6 <= hour_fractional <= 18:
                rad_factor = np.sin(np.pi * (hour_fractional - 6) / 12)
            else:
                rad_factor = 0.0
            sw = rad * rad_factor
            direct = sw * 0.7
            diffuse = sw * 0.3
            dni = sw / max(np.cos(np.radians(30)), 0.1)
            next_feat[3] = (sw - scaler.data_min_[3]) / (scaler.data_max_[3] - scaler.data_min_[3] + 1e-8)
            next_feat[4] = (direct - scaler.data_min_[4]) / (scaler.data_max_[4] - scaler.data_min_[4] + 1e-8)
            next_feat[5] = (diffuse - scaler.data_min_[5]) / (scaler.data_max_[5] - scaler.data_min_[5] + 1e-8)
            next_feat[6] = (dni - scaler.data_min_[6]) / (scaler.data_max_[6] - scaler.data_min_[6] + 1e-8)
        else:
            next_feat[3:] = window[-1, 3:]

        window = np.vstack([window, next_feat.reshape(1, -1)])

    return np.array(outputs)


def _predict_charging(model, n_steps, current_price=None, current_load=None):
    """
    充电迭代预测 (v3: input-driven 模式)
    用户输入当前电价+负荷 → 追加到历史窗口末尾 → 迭代预测

    Parameters
    ----------
    model : nn.Module
        充电 TCN-Attention-LSTM 模型
    n_steps : int
        预测步数 (1=15min, 4=1h, 96=24h)
    current_price : float or None
        用户输入的当前电价 (元/kWh)
    current_load : float or None
        用户输入的当前充电负荷 (kW)
    """
    scaler_X, scaler_y = _load_charging_scalers()
    df = _load_charging_df()

    seq_len = CHARGING_LOOKBACK  # 96

    # --- 将用户输入作为最新一行追加到历史数据末尾 ---
    if current_price is not None and current_load is not None and df is not None:
        # 取最新一条时间戳 + 15min 作为新行的时间
        last_ts = df['timestamp'].max()
        new_ts = last_ts + pd.Timedelta(minutes=15)

        # 计算 lag 特征
        # lag_1 = current_load（上一个负荷值即当前值本身作为 lag_1）
        if len(df) >= 1:
            lag_1 = float(df['load_kw'].iloc[-1])
        else:
            lag_1 = current_load

        # lag_96 = 24小时前 (96步 * 15min)
        if len(df) >= 96:
            lag_96 = float(df['load_kw'].iloc[-96])
        elif len(df) > 0:
            lag_96 = float(df['load_kw'].iloc[0])
        else:
            lag_96 = current_load

        # lag_672 = 7天前 (672步 * 15min)
        if len(df) >= 672:
            lag_672 = float(df['load_kw'].iloc[-672])
        elif len(df) > 0:
            lag_672 = float(df['load_kw'].iloc[0])
        else:
            lag_672 = current_load

        # rolling_std_4 / rolling_mean_4
        if len(df) >= 5:
            recent_vals = df['load_kw'].iloc[-5:].values
            # 用 current_load 替换最新值重新计算
            recent_vals[-1] = current_load
            rolling_std_4 = float(np.std(recent_vals[-4:]))
            rolling_mean_4 = float(np.mean(recent_vals[-4:]))
        else:
            rolling_std_4 = 222.0  # 训练集均值的 fallback
            rolling_mean_4 = current_load

        # 构造新行
        new_row = {
            'timestamp': new_ts,
            'price': current_price,
            'load_kw': current_load,
            'lag_1': lag_1,
            'lag_96': lag_96,
            'lag_672': lag_672,
            'rolling_std_4': rolling_std_4,
            'rolling_mean_4': rolling_mean_4,
        }
        new_df = pd.DataFrame([new_row])
        df = pd.concat([df, new_df], ignore_index=True)

    # --- 构建输入窗口 ---
    if df is not None and scaler_X is not None:
        window = _build_charging_window(df, seq_len, scaler_X)
    else:
        window = np.random.randn(seq_len, 6).astype(np.float32) * 0.1

    if window is None:
        window = np.random.randn(seq_len, 6).astype(np.float32) * 0.1

    if len(window) < seq_len:
        pad = seq_len - len(window)
        window = np.vstack([np.zeros((pad, 6), dtype=np.float32), window])

    window = window[-seq_len:]

    # 获取最新负荷作为起始预测值
    if current_load is not None:
        latest_known_load = current_load
    elif df is not None:
        try:
            latest_known_load = float(df['load_kw'].iloc[-1])
        except:
            latest_known_load = 875.0
    else:
        latest_known_load = 875.0

    # 获取最新的电价，用于迭代中的电价推算
    if current_price is not None:
        latest_known_price = current_price
    else:
        now = datetime.now()
        latest_known_price = _get_price(now.hour)

    # 维护预测值队列 (用于 lag_1, lag_96, lag_672 的滚动更新)
    pred_history = []

    now = datetime.now()
    current_hour = now.hour

    outputs = []
    for i in range(n_steps):
        inp = torch.FloatTensor(window[-seq_len:]).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred_scaled = model(inp).item()

        # 逆归一化得到真实负荷
        pred_real = float(scaler_y.inverse_transform([[pred_scaled]])[0, 0])
        y_min = float(scaler_y.data_min_[0])
        y_max = float(scaler_y.data_max_[0])
        pred_real = max(pred_real, y_min * 0.8)
        pred_real = min(pred_real, y_max * 1.2)

        outputs.append(pred_real)
        pred_history.append(pred_real)

        # 构建下一步特征
        hour_fractional = (current_hour + (i + 1) * 0.25) % 24
        price_val = _get_price(int(hour_fractional))

        # lag_1: 上一时刻的负荷
        lag_1 = pred_real

        # lag_96: 24小时前
        steps_so_far = len(pred_history)
        if steps_so_far >= 96:
            lag_96 = pred_history[-96]
        elif df is not None and len(df) >= 96:
            offset = len(df) - 96 + steps_so_far
            if offset < len(df) and offset >= 0:
                lag_96 = float(df['load_kw'].iloc[offset])
            else:
                lag_96 = pred_real
        else:
            lag_96 = pred_real

        # lag_672: 7天前
        if df is not None and len(df) >= 672:
            offset = len(df) - 672 + steps_so_far
            if offset < len(df) and offset >= 0:
                lag_672 = float(df['load_kw'].iloc[offset])
            else:
                lag_672 = pred_real
        elif steps_so_far >= 672:
            lag_672 = pred_history[-672]
        else:
            lag_672 = pred_real

        # rolling_std_4 / rolling_mean_4
        if len(pred_history) >= 4:
            recent = np.array(pred_history[-4:])
            rolling_std_4 = float(np.std(recent))
            rolling_mean_4 = float(np.mean(recent))
        elif df is not None and len(df) >= 5:
            # 混合使用历史数据和预测数据
            needed = 4
            mixed = []
            history_available = min(4, len(df))
            for j in range(history_available):
                mixed.append(float(df['load_kw'].iloc[len(df) - history_available + j]))
            if len(pred_history) > 0:
                mixed = mixed[-(needed - len(pred_history)):] + pred_history
            elif len(mixed) == 0:
                mixed = [latest_known_load]
            recent = np.array(mixed)
            rolling_std_4 = float(np.std(recent))
            rolling_mean_4 = float(np.mean(recent))
        else:
            rolling_std_4 = 222.0
            rolling_mean_4 = 875.0

        next_feat_raw = np.array([[price_val, lag_1, lag_96, lag_672, rolling_std_4, rolling_mean_4]], dtype=np.float32)
        next_feat = scaler_X.transform(next_feat_raw)[0]

        window = np.vstack([window, next_feat.reshape(1, -1)])

    mean = np.array(outputs)
    lower = mean * 0.85
    upper = mean * 1.15
    return mean, lower, upper


def _get_price(hour):
    """根据小时返回电价 (与训练数据一致: peak=0.85, mid=0.63, valley=0.25)"""
    if 8 <= hour < 11 or 18 <= hour < 21:
        return 0.85
    elif 6 <= hour < 8 or 11 <= hour < 18 or 21 <= hour < 22:
        return 0.63
    else:
        return 0.25


def _fallback_solar(n_steps, weather):
    np.random.seed(42)
    hours = np.array([(datetime.now().hour + i * 0.25) % 24 for i in range(n_steps)])
    solar = np.zeros(n_steps)
    day_mask = (hours >= 6) & (hours <= 18)
    rad = weather.get("radiation", 400) if weather else 400
    solar[day_mask] = rad * np.sin(np.pi * (hours[day_mask] - 6) / 12) * 0.001
    solar += np.random.randn(n_steps) * 3
    solar[solar < 0] = 0
    return solar


def _fallback_charging(n_steps):
    np.random.seed(42)
    hours = np.array([(datetime.now().hour + i * 0.25) % 24 for i in range(n_steps)])
    load = np.zeros(n_steps)
    load += 40 + 30 * np.exp(-0.5 * ((hours - 9) / 2) ** 2)
    load += 30 * np.exp(-0.5 * ((hours - 18) / 2) ** 2)
    load += np.random.randn(n_steps) * 3
    return {"mean": load, "lower": load * 0.85, "upper": load * 1.15}


# ============================================================
# 预测入口 (v3: 新增 current_price / current_load 参数)
# ============================================================
def run_prediction(n_steps, weather, current_price=None, current_load=None):
    solar_model = load_solar_model()
    charging_model = load_charging_model()
    if solar_model is not None or charging_model is not None:
        return _real_prediction(n_steps, weather, current_price=current_price, current_load=current_load)
    return _mock_prediction(n_steps, weather)


# ============================================================
# app.py 兼容层 — 提供 app.py 直接调用的函数名
# ============================================================
_last_predictions = {}  # {"solar": dict, "charging": dict, "combined": dict}


def predict_solar(n_steps=4):
    """简化的光伏预测入口 (兼容 app.py)"""
    try:
        weather = {"radiation": 400}  # 默认辐照度
        model = load_solar_model()
        if model is None:
            return None
        return run_prediction(n_steps, weather)
    except Exception as e:
        print(f"[WARN] predict_solar 失败: {e}")
        return None


def predict_charging(price=0.8, load=20.0, n_steps=4):
    """简化的充电预测入口 (兼容 app.py)"""
    try:
        weather = {"radiation": 400}
        return run_prediction(n_steps, weather, current_price=price, current_load=load)
    except Exception as e:
        print(f"[WARN] predict_charging 失败: {e}")
        return None


def save_prediction_result(model_type: str, result: dict):
    """保存最近一次预测结果"""
    _last_predictions[model_type] = result


def get_last_prediction(model_type: str):
    """获取最近一次预测结果"""
    return _last_predictions.get(model_type)


def get_combined_prediction(solar_result, charging_result, n_steps=4):
    """合并光伏和充电预测结果，对齐到相同步数"""
    if solar_result is None or charging_result is None:
        return None
    try:
        s_solar = np.array(solar_result.get("solar", []))
        c_load = np.array(charging_result.get("load_mean", []))
        c_lower = np.array(charging_result.get("load_lower", []))
        c_upper = np.array(charging_result.get("load_upper", []))

        min_len = min(len(s_solar), len(c_load), n_steps)
        if min_len == 0:
            return None

        now = datetime.now()
        solar_vals = s_solar[:min_len]
        load_vals = c_load[:min_len]
        times = [now + timedelta(minutes=15 * i) for i in range(min_len)]

        total_solar = np.sum(solar_vals) * 0.25
        total_load = np.sum(load_vals) * 0.25
        green_ratio = total_solar / total_load * 100 if total_load > 0 else 100
        solar_peak_idx = np.argmax(solar_vals)
        load_peak_idx = np.argmax(load_vals)

        combined = {
            "times": times,
            "solar": solar_vals,
            "load_mean": load_vals,
            "load_lower": c_lower[:min_len] if len(c_lower) >= min_len else load_vals * 0.85,
            "load_upper": c_upper[:min_len] if len(c_upper) >= min_len else load_vals * 1.15,
            "total_solar": total_solar,
            "total_load": total_load,
            "green_ratio": green_ratio,
            "solar_peak": solar_vals[solar_peak_idx],
            "solar_peak_time": times[solar_peak_idx],
            "load_peak": load_vals[load_peak_idx],
            "model_status": {
                "solar_ok": True,
                "charging_ok": True,
            },
        }
        save_prediction_result("combined", combined)
        return combined
    except Exception as e:
        print(f"[WARN] get_combined_prediction 失败: {e}")
        return None


# ============================================================
# 策略建议生成
# ============================================================
def generate_strategy(result):
    solar = result["solar"]
    load = result["load_mean"]
    times = result["times"]
    green_ratio = result["green_ratio"]
    lines = ["## 💡 充放电策略建议\n"]
    lines.append(f"**绿电替代率**: {green_ratio:.1f}%\n")
    net = solar - load
    surplus_mask = net > 5
    deficit_mask = net < -5
    surplus_hours = []
    deficit_hours = []
    for i, (t, val) in enumerate(zip(times, net)):
        if surplus_mask[i]:
            surplus_hours.append((t.strftime("%H:%M"), val))
        if deficit_mask[i]:
            deficit_hours.append((t.strftime("%H:%M"), -val))
    if surplus_hours:
        lines.append("### 🟢 储能充电建议")
        lines.append("以下时段光伏出力大于充电需求，适合向储能系统充电：\n")
        for t, v in surplus_hours[:6]:
            lines.append(f"- **{t}** — 光伏盈余 {v:.1f} kW")
    if deficit_hours:
        lines.append("\n### 🔴 储能放电建议")
        lines.append("以下时段充电需求大于光伏出力，可调度储能放电补充：\n")
        for t, v in deficit_hours[:6]:
            lines.append(f"- **{t}** — 缺口 {v:.1f} kW")
    if not surplus_hours and not deficit_hours:
        lines.append("\n供需基本平衡，无需特殊调度。")
    lines.append(f"\n---\n*基于 {len(times)} 步预测生成 | 仅供参考*")
    return "\n".join(lines)