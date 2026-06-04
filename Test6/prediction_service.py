"""
预测服务：加载模型、执行预测、生成策略建议 (Test6 BentoML 版)
"""
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from config import (
    SOLAR_MODEL_PTH, CHARGING_MODEL_PTH,
    SOLAR_LOOKBACK, CHARGING_LOOKBACK,
    SOLAR_FEATURE_COLS, MC_DROPOUT_SAMPLES,
    SOLAR_FEATURE_DIM, CHARGING_FEATURE_DIM,
    BASE_DIR, ROOT_DIR,
    DATA_ALIGNED, DATA_SELECTED_TEST,
)

# ============================================================
# 充电模型类定义
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
# 光伏模型类定义
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
# 特征列
# ============================================================
CHARGING_FEATURE_COLS = ['price', 'lag_1', 'lag_96', 'lag_672', 'rolling_std_4', 'rolling_mean_4']

SOLAR_FEATURE_COLS_7 = [
    'power', 'hour_sin', 'hour_cos',
    'shortwave_radiation (W/m2)', 'direct_radiation (W/m2)',
    'diffuse_radiation (W/m2)', 'direct_normal_irradiance (W/m2)'
]

if TORCH_AVAILABLE:
    DEVICE = "cpu"

# 模型缓存
_solar_model = None
_solar_loaded = False
_charging_model = None
_charging_loaded = False
_CHARGING_INPUT_DIM = 6
_CHARGING_HIDDEN_SIZE = 121
_SOLAR_HIDDEN_SIZE = 128
_SOLAR_NUM_LAYERS = 2
_solar_scaler = None
_charging_scaler_X = None
_charging_scaler_y = None


# ============================================================
# Scaler 加载
# ============================================================
def _load_charging_scalers():
    global _charging_scaler_X, _charging_scaler_y
    if _charging_scaler_X is not None:
        return _charging_scaler_X, _charging_scaler_y
    import pickle
    scaler_X_path = os.path.join(ROOT_DIR, "Charging_Retraining", "scaler_X.pkl")
    scaler_y_path = os.path.join(ROOT_DIR, "Charging_Retraining", "scaler_y.pkl")
    if not os.path.exists(scaler_X_path) or not os.path.exists(scaler_y_path):
        return None, None
    with open(scaler_X_path, 'rb') as f:
        _charging_scaler_X = pickle.load(f)
    with open(scaler_y_path, 'rb') as f:
        _charging_scaler_y = pickle.load(f)
    return _charging_scaler_X, _charging_scaler_y


def _load_solar_scaler():
    global _solar_scaler
    if _solar_scaler is not None:
        return _solar_scaler
    import joblib
    scaler_path = os.path.join(ROOT_DIR, "Solar_Forecast", "simple_scaler_1.pkl")
    if not os.path.exists(scaler_path):
        return None
    _solar_scaler = joblib.load(scaler_path)
    return _solar_scaler


# ============================================================
# 模型加载
# ============================================================
def _infer_charging_hidden_size(pth_path):
    if not TORCH_AVAILABLE or not os.path.exists(pth_path):
        return _CHARGING_HIDDEN_SIZE
    state = torch.load(pth_path, map_location="cpu", weights_only=True)
    for k in state:
        if "lstm.weight_ih_l0" in k:
            return state[k].shape[0] // 4
    return _CHARGING_HIDDEN_SIZE


def load_solar_model():
    global _solar_model, _solar_loaded
    if _solar_loaded:
        return _solar_model
    if not TORCH_AVAILABLE:
        return None
    try:
        if not os.path.exists(SOLAR_MODEL_PTH):
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
            return None
        hs = _infer_charging_hidden_size(CHARGING_MODEL_PTH)
        model = HybridModel(input_dim=_CHARGING_INPUT_DIM, hidden_dim=hs, output_dim=1)
        state = torch.load(CHARGING_MODEL_PTH, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state)
        model.to(DEVICE)
        model.eval()
        _charging_model = model
        _charging_loaded = True
        return _charging_model
    except Exception as e:
        print(f"[WARN] 充电模型加载失败: {e}")
        return None


def load_all_models():
    """预加载所有模型和 scaler"""
    load_solar_model()
    load_charging_model()
    _load_solar_scaler()
    _load_charging_scalers()
    return {
        "solar_ok": _solar_loaded,
        "charging_ok": _charging_loaded,
        "solar_scaler_ok": _solar_scaler is not None,
        "charging_scaler_ok": _charging_scaler_X is not None,
    }


# ============================================================
# 数据窗口构建
# ============================================================
def _load_aligned_df():
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
    if df is None or len(df) < seq_len:
        return None
    raw = df[SOLAR_FEATURE_COLS_7].tail(seq_len).values.astype(np.float32)
    if scaler is not None:
        raw = scaler.transform(raw)
    return raw


def _load_charging_df():
    test_path = os.path.join(ROOT_DIR, "Data", "dataset_selected_features_test.csv")
    if not os.path.exists(test_path):
        return None
    df = pd.read_csv(test_path, parse_dates=['timestamp'])
    df.sort_values('timestamp', inplace=True)
    return df


def _build_charging_window(df, seq_len, scaler_X):
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
    return _build_result(times, solar, load, load * 0.95, load * 1.05,
                         total_solar, total_load, green_ratio,
                         solar_peak_idx, load_peak_idx,
                         {"solar_ok": False, "charging_ok": False})


def _build_result(times, solar, load_mean, load_lower, load_upper,
                  total_solar, total_load, green_ratio,
                  solar_peak_idx, load_peak_idx, model_status):
    return {
        "times": [t.strftime("%Y-%m-%d %H:%M") for t in times],
        "solar": solar.tolist() if hasattr(solar, 'tolist') else list(solar),
        "load_mean": load_mean.tolist() if hasattr(load_mean, 'tolist') else list(load_mean),
        "load_lower": load_lower.tolist() if hasattr(load_lower, 'tolist') else list(load_lower),
        "load_upper": load_upper.tolist() if hasattr(load_upper, 'tolist') else list(load_upper),
        "summary": {
            "total_solar": round(float(total_solar), 2),
            "total_load": round(float(total_load), 2),
            "green_ratio": round(float(green_ratio), 1),
            "solar_peak": round(float(solar[solar_peak_idx]), 2),
            "solar_peak_time": times[solar_peak_idx].strftime("%H:%M"),
            "load_peak": round(float(load_mean[load_peak_idx]), 2),
            "load_peak_time": times[load_peak_idx].strftime("%H:%M"),
        },
        "model_status": model_status,
        "n_steps": len(times),
    }


# ============================================================
# 真实预测
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

    return _build_result(times, solar, load_mean, load_lower, load_upper,
                         total_solar, total_load, green_ratio,
                         solar_peak_idx, load_peak_idx,
                         {"solar_ok": solar_ok, "charging_ok": charging_ok})


def _predict_solar(model, n_steps, weather):
    scaler = _load_solar_scaler()
    df = _load_aligned_df()
    seq_len = SOLAR_LOOKBACK

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
    scaler_X, scaler_y = _load_charging_scalers()
    df = _load_charging_df()
    seq_len = CHARGING_LOOKBACK

    if current_price is not None and current_load is not None and df is not None:
        last_ts = df['timestamp'].max()
        new_ts = last_ts + pd.Timedelta(minutes=15)

        if len(df) >= 1:
            lag_1 = float(df['load_kw'].iloc[-1])
        else:
            lag_1 = current_load

        if len(df) >= 96:
            lag_96 = float(df['load_kw'].iloc[-96])
        elif len(df) > 0:
            lag_96 = float(df['load_kw'].iloc[0])
        else:
            lag_96 = current_load

        if len(df) >= 672:
            lag_672 = float(df['load_kw'].iloc[-672])
        elif len(df) > 0:
            lag_672 = float(df['load_kw'].iloc[0])
        else:
            lag_672 = current_load

        if len(df) >= 5:
            recent_vals = df['load_kw'].iloc[-5:].values
            recent_vals[-1] = current_load
            rolling_std_4 = float(np.std(recent_vals[-4:]))
            rolling_mean_4 = float(np.mean(recent_vals[-4:]))
        else:
            rolling_std_4 = 222.0
            rolling_mean_4 = current_load

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

    if current_load is not None:
        latest_known_load = current_load
    elif df is not None:
        try:
            latest_known_load = float(df['load_kw'].iloc[-1])
        except:
            latest_known_load = 875.0
    else:
        latest_known_load = 875.0

    if current_price is not None:
        latest_known_price = current_price
    else:
        now = datetime.now()
        latest_known_price = _get_price(now.hour)

    pred_history = []
    now = datetime.now()
    current_hour = now.hour
    outputs = []

    for i in range(n_steps):
        inp = torch.FloatTensor(window[-seq_len:]).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred_scaled = model(inp).item()

        pred_real = float(scaler_y.inverse_transform([[pred_scaled]])[0, 0])
        y_min = float(scaler_y.data_min_[0])
        y_max = float(scaler_y.data_max_[0])
        pred_real = max(pred_real, y_min * 0.8)
        pred_real = min(pred_real, y_max * 1.2)

        outputs.append(pred_real)
        pred_history.append(pred_real)

        hour_fractional = (current_hour + (i + 1) * 0.25) % 24
        price_val = _get_price(int(hour_fractional))
        lag_1 = pred_real

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

        if len(pred_history) >= 4:
            recent = np.array(pred_history[-4:])
            rolling_std_4 = float(np.std(recent))
            rolling_mean_4 = float(np.mean(recent))
        elif df is not None and len(df) >= 5:
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
    lower = mean * 0.95
    upper = mean * 1.05
    return mean, lower, upper


def _get_price(hour):
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
    return {"mean": load, "lower": load * 0.95, "upper": load * 1.05}


# ============================================================
# 预测入口
# ============================================================
def run_prediction(n_steps, weather, current_price=None, current_load=None):
    solar_model = load_solar_model()
    charging_model = load_charging_model()
    if solar_model is not None or charging_model is not None:
        return _real_prediction(n_steps, weather, current_price=current_price, current_load=current_load)
    return _mock_prediction(n_steps, weather)


def run_solar_only(n_steps, weather=None):
    """仅光伏预测"""
    if weather is None:
        weather = {"radiation": 400}
    solar_model = load_solar_model()
    now = datetime.now()
    times = [now + timedelta(minutes=15 * i) for i in range(n_steps)]
    if solar_model:
        solar = _predict_solar(solar_model, n_steps, weather)
    else:
        solar = _fallback_solar(n_steps, weather)
    return {
        "times": [t.strftime("%Y-%m-%d %H:%M") for t in times],
        "solar": solar.tolist(),
        "total_solar": round(float(np.sum(solar) * 0.25), 2),
        "solar_peak": round(float(solar[np.argmax(solar)]), 2),
        "solar_peak_time": times[np.argmax(solar)].strftime("%H:%M"),
        "model_ok": solar_model is not None,
    }


def run_charging_only(n_steps, current_price=0.8, current_load=20.0):
    """仅充电预测"""
    charging_model = load_charging_model()
    now = datetime.now()
    times = [now + timedelta(minutes=15 * i) for i in range(n_steps)]
    if charging_model:
        load_mean, load_lower, load_upper = _predict_charging(
            charging_model, n_steps, current_price=current_price, current_load=current_load
        )
    else:
        res = _fallback_charging(n_steps)
        load_mean, load_lower, load_upper = res["mean"], res["lower"], res["upper"]
    return {
        "times": [t.strftime("%Y-%m-%d %H:%M") for t in times],
        "load_mean": load_mean.tolist(),
        "load_lower": load_lower.tolist(),
        "load_upper": load_upper.tolist(),
        "total_load": round(float(np.sum(load_mean) * 0.25), 2),
        "load_peak": round(float(load_mean[np.argmax(load_mean)]), 2),
        "load_peak_time": times[np.argmax(load_mean)].strftime("%H:%M"),
        "model_ok": charging_model is not None,
    }


# ============================================================
# 策略建议
# ============================================================
def generate_strategy(result):
    solar = np.array(result["solar"])
    load = np.array(result["load_mean"])
    times_raw = result["times"]
    green_ratio = result["summary"]["green_ratio"]

    net = solar - load
    surplus = []
    deficit = []
    for i, (t_str, val) in enumerate(zip(times_raw, net)):
        if val > 5:
            surplus.append({"time": t_str[-5:], "surplus": round(float(val), 1)})
        elif val < -5:
            deficit.append({"time": t_str[-5:], "deficit": round(float(-val), 1)})

    return {
        "green_ratio": green_ratio,
        "surplus_periods": surplus[:6],
        "deficit_periods": deficit[:6],
        "balanced": len(surplus) == 0 and len(deficit) == 0,
        "n_steps": len(times_raw),
    }