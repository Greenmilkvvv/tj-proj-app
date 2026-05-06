"""
预测服务：加载模型、执行预测、生成策略建议
"""
import os
import sys
import numpy as np
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
    DATA_ALIGNED,
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

if TORCH_AVAILABLE:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
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
# 辅助：从 aligned CSV 构建特征窗口
# ============================================================
def _build_solar_window(df, seq_len):
    """从 aligned CSV 数值列构建光伏模型输入窗口 (动态列匹配)"""
    num_cols = [c for c in df.columns if c != 'datetime' and np.issubdtype(df[c].dtype, np.number)]
    # 优先以 power 开头
    if 'power' in num_cols:
        idx = num_cols.index('power')
        num_cols = num_cols[idx:] + num_cols[:idx]
    raw = df[num_cols].tail(seq_len).values.astype(np.float32)
    n = raw.shape[0]
    dim = SOLAR_FEATURE_DIM
    out = np.zeros((n, dim), dtype=np.float32)
    out[:, 0] = raw[:, 0]  # power
    # 时间编码
    now = datetime.now()
    hours = np.arange(-n, 0) * 0.25 + now.hour + now.minute / 60
    out[:, 1] = np.sin(2 * np.pi * hours / 24)
    out[:, 2] = np.cos(2 * np.pi * hours / 24)
    # 其余列填充
    if raw.shape[1] > 1:
        copy_n = min(raw.shape[1] - 1, dim - 3)
        out[:, 3:3 + copy_n] = raw[:, 1:1 + copy_n]
    return out


def _build_charging_window(df, seq_len):
    """从 aligned CSV 数值列构建充电模型输入窗口"""
    num_cols = [c for c in df.columns if c != 'datetime' and np.issubdtype(df[c].dtype, np.number)]
    if 'power' in num_cols:
        idx = num_cols.index('power')
        num_cols = num_cols[idx:] + num_cols[:idx]
    raw = df[num_cols].tail(seq_len).values.astype(np.float32)
    n = raw.shape[0]
    dim = CHARGING_FEATURE_DIM
    out = np.zeros((n, dim), dtype=np.float32)
    out[:, 0] = raw[:, 0]
    now = datetime.now()
    hours = np.arange(-n, 0) * 0.25 + now.hour + now.minute / 60
    out[:, 1] = np.sin(2 * np.pi * hours / 24)
    out[:, 2] = np.cos(2 * np.pi * hours / 24)
    if raw.shape[1] > 1:
        copy_n = min(raw.shape[1] - 1, dim - 3)
        out[:, 3:3 + copy_n] = raw[:, 1:1 + copy_n]
    return out


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
# 真实预测
# ============================================================
def _real_prediction(n_steps, weather):
    now = datetime.now().replace(minute=0, second=0, microsecond=0)
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
        load_mean, load_lower, load_upper = _predict_charging(charging_model, n_steps)
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
    import pandas as pd
    seq_len = SOLAR_LOOKBACK
    aligned_path = DATA_ALIGNED
    if os.path.exists(aligned_path):
        df = pd.read_csv(aligned_path)
        window = _build_solar_window(df, seq_len)
    else:
        window = np.random.randn(seq_len, SOLAR_FEATURE_DIM).astype(np.float32) * 0.1
    outputs = []
    hour_now = datetime.now().hour
    for i in range(n_steps):
        if len(window) < seq_len:
            window = np.pad(window, ((seq_len - len(window), 0), (0, 0)), mode="edge")
        inp = torch.FloatTensor(window[-seq_len:]).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred = model(inp).item()
        outputs.append(max(pred, 0))
        next_feat = window[-1].copy()
        hour = (hour_now + (i + 1) * 0.25) % 24
        next_feat[1] = np.sin(2 * np.pi * hour / 24)
        next_feat[2] = np.cos(2 * np.pi * hour / 24)
        next_feat[0] = pred
        if weather and weather.get("radiation") is not None and len(next_feat) > 3:
            rad_factor = max(0, np.sin(np.pi * (hour - 6) / 12)) if 6 <= hour <= 18 else 0
            next_feat[3] = weather["radiation"] * rad_factor
        window = np.vstack([window, next_feat.reshape(1, -1)])
    return np.array(outputs)


def _predict_charging(model, n_steps):
    import pandas as pd
    seq_len = CHARGING_LOOKBACK
    aligned_path = DATA_ALIGNED
    if os.path.exists(aligned_path):
        df = pd.read_csv(aligned_path)
        window = _build_charging_window(df, seq_len)
    else:
        window = np.random.randn(seq_len, CHARGING_FEATURE_DIM).astype(np.float32) * 0.1
    outputs = []
    hour_now = datetime.now().hour
    for i in range(n_steps):
        if len(window) < seq_len:
            window = np.pad(window, ((seq_len - len(window), 0), (0, 0)), mode="edge")
        inp = torch.FloatTensor(window[-seq_len:]).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred = model(inp).item()
        outputs.append(max(pred, 0))
        next_feat = window[-1].copy()
        hour = (hour_now + (i + 1) * 0.25) % 24
        next_feat[1] = np.sin(2 * np.pi * hour / 24)
        next_feat[2] = np.cos(2 * np.pi * hour / 24)
        next_feat[0] = pred
        window = np.vstack([window, next_feat.reshape(1, -1)])
    mean = np.array(outputs)
    return mean, mean * 0.85, mean * 1.15


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
# 预测入口
# ============================================================
def run_prediction(n_steps, weather):
    solar_model = load_solar_model()
    charging_model = load_charging_model()
    if solar_model is not None or charging_model is not None:
        return _real_prediction(n_steps, weather)
    return _mock_prediction(n_steps, weather)


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