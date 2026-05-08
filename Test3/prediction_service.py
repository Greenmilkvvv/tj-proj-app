"""
预测服务：加载模型、执行预测、生成策略建议 (Test3: LightGBM 版本)
=============================
核心变更：充电负荷预测从 TCN-Attention-LSTM 替换为 LightGBM。
光伏预测保留原有 LSTM 模型不变。
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# PyTorch (仅光伏模型需要)
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
    print("[INFO] PyTorch 可用")
except ImportError:
    TORCH_AVAILABLE = False
    print("[WARN] PyTorch 不可用")

from config import (
    SOLAR_MODEL_PTH, CHARGING_MODEL_PTH,
    SOLAR_LOOKBACK, CHARGING_LOOKBACK,
    SOLAR_FEATURE_COLS, MC_DROPOUT_SAMPLES,
    SOLAR_FEATURE_DIM, CHARGING_FEATURE_DIM,
    BASE_DIR, ROOT_DIR,
    DATA_ALIGNED, CHARGING_FEATURE_COLS,
    PEAK_PRICE, MID_PRICE, VALLEY_PRICE,
)

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
def _load_solar_scaler():
    """加载光伏模型 scaler (MinMaxScaler)"""
    try:
        import joblib
    except ImportError:
        import pickle as _pkl
        joblib = None
    scaler_path = os.path.join(ROOT_DIR, "Solar_Forecast", "simple_scaler_1.pkl")
    if not os.path.exists(scaler_path):
        print("[WARN] 光伏 scaler 文件不存在")
        return None
    try:
        if joblib is not None:
            scaler = joblib.load(scaler_path)
        else:
            with open(scaler_path, 'rb') as f:
                scaler = _pkl.load(f)
        print("[OK] 光伏 scaler 加载成功")
        return scaler
    except Exception as e:
        print(f"[WARN] 光伏 scaler 加载失败: {e}")
        return None


def _load_charging_scalers():
    """加载 LightGBM 模型使用的 StandardScaler"""
    scaler_X_path = os.path.join(BASE_DIR, "scaler_X_lightgbm.pkl")
    scaler_y_path = os.path.join(BASE_DIR, "scaler_y_lightgbm.pkl")
    if not os.path.exists(scaler_X_path) or not os.path.exists(scaler_y_path):
        print("[WARN] 充电 scaler 文件不存在, 将使用模拟预测")
        return None, None
    with open(scaler_X_path, 'rb') as f:
        scaler_X = pickle.load(f)
    with open(scaler_y_path, 'rb') as f:
        scaler_y = pickle.load(f)
    print("[OK] 充电 scaler 加载成功 (StandardScaler)")
    return scaler_X, scaler_y


# ============================================================
# 模型缓存
# ============================================================
_solar_model = None
_solar_loaded = False
_lgb_model = None
_lgb_loaded = False
DEVICE = "cpu"

_SOLAR_HIDDEN_SIZE = 128
_SOLAR_NUM_LAYERS = 2

if TORCH_AVAILABLE:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] 预测设备: {DEVICE}")


# ============================================================
# 模型加载
# ============================================================
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
        lstm = LSTMPredictor(
            input_size=SOLAR_FEATURE_DIM, hidden_size=_SOLAR_HIDDEN_SIZE,
            num_layers=_SOLAR_NUM_LAYERS, output_size=1,
            dropout=0.2, bidirectional=False,
        )
        model = GeneratorWithFeatures(lstm)
        state = torch.load(SOLAR_MODEL_PTH, map_location=DEVICE, weights_only=True)
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
    """加载 LightGBM 充电负荷预测模型"""
    global _lgb_model, _lgb_loaded
    if _lgb_loaded:
        return _lgb_model
    try:
        if not os.path.exists(CHARGING_MODEL_PTH):
            print(f"[WARN] LightGBM 模型文件不存在: {CHARGING_MODEL_PTH}")
            print(f"[HINT] 请先运行 train_lightgbm.py 训练模型")
            return None
        with open(CHARGING_MODEL_PTH, 'rb') as f:
            model = pickle.load(f)
        _lgb_model = model
        _lgb_loaded = True
        print(f"[OK] LightGBM 充电模型加载成功 (best_iter={model.best_iteration_})")
        return _lgb_model
    except Exception as e:
        print(f"[WARN] LightGBM 模型加载失败: {e}")
        return None


# ============================================================
# 历史窗口构建
# ============================================================
def _load_aligned_df():
    """加载 aligned CSV 并做特征工程"""
    aligned_path = DATA_ALIGNED
    if not os.path.exists(aligned_path):
        return None
    df = pd.read_csv(aligned_path, parse_dates=['datetime'])
    df.sort_values('datetime', inplace=True)
    df['hour'] = df['datetime'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    # theta
    diffuse = df['diffuse_radiation (W/m2)'].values
    dni = df['direct_normal_irradiance (W/m2)'].values
    dni_safe = np.where(np.abs(dni) < 1e-6, 1.0, dni)
    cos_theta = (df['shortwave_radiation (W/m2)'].values - diffuse) / dni_safe
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    df['theta'] = np.arccos(cos_theta)
    return df


def _load_charging_df():
    """加载充电测试数据"""
    test_path = os.path.join(ROOT_DIR, "Data", "dataset_selected_features_test.csv")
    if not os.path.exists(test_path):
        return None
    df = pd.read_csv(test_path, parse_dates=['timestamp'])
    df.sort_values('timestamp', inplace=True)
    return df


def _build_solar_window(df, seq_len, scaler):
    """从 aligned CSV 构建光伏模型输入窗口"""
    if df is None or len(df) < seq_len:
        return None
    feature_cols_7 = [
        'power', 'hour_sin', 'hour_cos',
        'shortwave_radiation (W/m2)', 'direct_radiation (W/m2)',
        'diffuse_radiation (W/m2)', 'direct_normal_irradiance (W/m2)'
    ]
    raw = df[feature_cols_7].tail(seq_len).values.astype(np.float32)
    if scaler is not None:
        raw = scaler.transform(raw)
    return raw


# ============================================================
# LightGBM 充电预测 (核心变更)
# ============================================================
def _predict_charging_lgb(lgb_model, n_steps):
    """
    使用 LightGBM 模型进行充电负荷迭代预测。
    LightGBM 输入为展平的滑动窗口特征向量 (lookback * n_features)。
    由于 LightGBM 直接输出真实值 (训练时未使用标准化后的 y)，不需要 inverse_transform。
    """
    scaler_X, scaler_y = _load_charging_scalers()
    df = _load_charging_df()

    seq_len = CHARGING_LOOKBACK  # 96
    feature_cols = CHARGING_FEATURE_COLS  # 6 个特征

    # 从测试集获取历史窗口数据 (原始值)
    if df is not None:
        available_cols = [c for c in feature_cols + ['load_kw'] if c in df.columns]
        if len(available_cols) < len(feature_cols) + 1:
            print("[WARN] 充电数据列不足，使用 fallback")
            df = None

    # 构建历史数据窗口 (原始值)
    if df is not None:
        raw_data = df[available_cols].tail(seq_len).values.astype(np.float64)
    else:
        # Fallback: 生成随机窗口
        raw_data = np.random.randn(seq_len, len(feature_cols) + 1).astype(np.float64) * 100 + 500

    # 确保窗口足够长
    if len(raw_data) < seq_len:
        pad = seq_len - len(raw_data)
        pad_data = np.vstack([np.zeros((pad, raw_data.shape[1])), raw_data])
        raw_data = pad_data

    raw_data = raw_data[-seq_len:]

    # 提取特征和标签的历史序列
    # 列顺序: [price, lag_1, lag_96, lag_672, rolling_std_4, rolling_mean_4, load_kw]
    hist_features = raw_data[:, :-1]  # (seq_len, 6)
    hist_load = raw_data[:, -1]       # (seq_len,)

    # 获取最近的真实负荷值
    last_real_load = float(hist_load[-1]) if hist_load[-1] > 0 else 500.0

    # 维护预测历史 (用于 lag_96, lag_672, rolling 特征)
    pred_history = list(hist_load[-672:])  # 保留足够的历史

    # 电价函数
    now = datetime.now()
    current_hour = now.hour

    def get_price(hour):
        h = int(hour) % 24
        if 8 <= h < 11 or 18 <= h < 21:
            return PEAK_PRICE
        elif 6 <= h < 8 or 11 <= h < 18 or 21 <= h < 22:
            return MID_PRICE
        else:
            return VALLEY_PRICE

    outputs = []
    for i in range(n_steps):
        # 构建当前特征窗口 (原始值)
        if len(pred_history) >= seq_len:
            window_features = []
            for j in range(seq_len):
                idx = len(pred_history) - seq_len + j
                load_j = pred_history[idx]
                # 计算当前特征: price, lag_1, lag_96, lag_672, rolling_std_4, rolling_mean_4
                hour_j = (now.hour + now.minute / 60 + (i - seq_len + j + 1) * 0.25) % 24
                price_j = get_price(hour_j)

                lag_1_j = pred_history[idx - 1] if idx >= 1 else load_j
                lag_96_j = pred_history[idx - 96] if idx >= 96 else load_j
                lag_672_j = pred_history[idx - 672] if idx >= 672 else load_j

                recent_4 = pred_history[max(0, idx - 3): idx + 1]
                rolling_std_4_j = float(np.std(recent_4)) if len(recent_4) >= 2 else 200.0
                rolling_mean_4_j = float(np.mean(recent_4))

                window_features.append([price_j, lag_1_j, lag_96_j, lag_672_j, rolling_std_4_j, rolling_mean_4_j])
            window_features = np.array(window_features, dtype=np.float64)
        else:
            # 初始几步用历史特征
            window_features = hist_features[-seq_len:]

        # 展平特征: (6 * 96,) = 576 维
        flat_features = window_features.flatten().reshape(1, -1)

        # LightGBM 预测 (直接输出真实负荷值 kW)
        try:
            pred_real = float(lgb_model.predict(flat_features)[0])
        except Exception as e:
            print(f"[WARN] LightGBM 预测失败 at step {i}: {e}")
            pred_real = float(np.mean(outputs)) if outputs else 500.0

        # 确保预测值为正
        if pred_real <= 0 and len(outputs) > 0:
            prev_positive = [v for v in outputs if v > 0]
            pred_real = prev_positive[-1] if prev_positive else 100.0
        elif pred_real <= 0:
            pred_real = 100.0

        outputs.append(pred_real)
        pred_history.append(pred_real)

    mean = np.array(outputs)
    # 不确定性: 使用 15% 波动 (LightGBM 不提供置信区间)
    lower = mean * 0.85
    upper = mean * 1.15
    return mean, lower, upper


# ============================================================
# 光伏预测 (与 Test2 一致)
# ============================================================
def _predict_solar(model, n_steps, weather):
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


# ============================================================
# Fallback 预测
# ============================================================
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
# 模拟预测 (all fallback)
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
# 真实预测入口
# ============================================================
def _real_prediction(n_steps, weather):
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
        load_mean, load_lower, load_upper = _predict_charging_lgb(charging_model, n_steps)
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