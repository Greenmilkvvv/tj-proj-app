"""
预测服务模块
封装光伏预测（CNN-LSTM）和充电负荷预测（TCN-Attention-LSTM Hybrid）
支持滚动多步预测、MC Dropout 不确定性估计、容错回退
"""
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# PyTorch 可选
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[INFO] PyTorch 不可用，预测将使用模拟数据")

# 添加项目根目录到路径以导入 NN
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Solar_Forecast"))

from config import (
    SOLAR_MODEL_PTH, CHARGING_MODEL_PTH,
    SOLAR_LOOKBACK, CHARGING_LOOKBACK,
    MC_DROPOUT_SAMPLES,
    SOLAR_FEATURE_COLS,
    DATA_ALL_TRAIN, DATA_ALIGNED,
)


# ============================================================
# 模型加载
# ============================================================
_solar_model = None
_solar_loaded = False
_charging_model = None
_charging_loaded = False
DEVICE = "cpu"

if TORCH_AVAILABLE:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] 预测设备: {DEVICE}")


def load_solar_model():
    """加载光伏预测模型"""
    global _solar_model, _solar_loaded
    if _solar_loaded:
        return _solar_model

    if not TORCH_AVAILABLE:
        print("[INFO] PyTorch 不可用，光伏模型回退到模拟")
        return None

    if not os.path.exists(SOLAR_MODEL_PTH):
        print(f"[WARN] 光伏模型文件不存在: {SOLAR_MODEL_PTH}")
        return None

    try:
        from NN import GeneratorWithFeatures
        model = GeneratorWithFeatures(
            input_size=SOLAR_FEATURE_DIM,
            hidden_size=128,
            num_layers=2,
            output_size=1,
            dropout=0.2,
            bidirectional=False,
            cnn_channels=[64, 64],
            kernel_size=3,
        )
        state_dict = torch.load(SOLAR_MODEL_PTH, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        _solar_model = model
        _solar_loaded = True
        print(f"[INFO] 光伏模型已加载: {SOLAR_MODEL_PTH}")
        return model
    except Exception as e:
        print(f"[WARN] 光伏模型加载失败: {e}")
        return None


def load_charging_model():
    """加载充电负荷预测模型"""
    global _charging_model, _charging_loaded
    if _charging_loaded:
        return _charging_model

    if not TORCH_AVAILABLE:
        print("[INFO] PyTorch 不可用，充电模型回退到模拟")
        return None

    if not os.path.exists(CHARGING_MODEL_PTH):
        print(f"[WARN] 充电模型文件不存在: {CHARGING_MODEL_PTH}")
        return None

    # 从 notebook 中定义的 HybridModel 结构
    # 由于模型定义在 .ipynb 中，这里直接内联定义
    try:
        import torch.nn as nn

        class HybridModel(nn.Module):
            """TCN-Attention-LSTM Hybrid 模型（与训练保持一致）"""
            def __init__(self, input_size=6, hidden_size=64, output_size=1,
                         num_layers=2, dropout=0.2, bidirectional=False,
                         cnn_channels=None, use_attention=True):
                super().__init__()
                if cnn_channels is None:
                    cnn_channels = [64, 128]
                self.input_size = input_size
                self.hidden_size = hidden_size
                self.use_attention = use_attention
                direction = 2 if bidirectional else 1

                # TCN 部分
                tcn_layers = []
                in_ch = input_size
                for out_ch in cnn_channels:
                    tcn_layers.extend([
                        nn.Conv1d(in_ch, out_ch, 3, padding=1),
                        nn.BatchNorm1d(out_ch),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                    ])
                    in_ch = out_ch
                self.tcn = nn.Sequential(*tcn_layers)
                tcn_out = cnn_channels[-1]

                # Attention
                if use_attention:
                    self.attention = nn.MultiheadAttention(tcn_out, 4, dropout=dropout, batch_first=True)

                # LSTM
                self.lstm = nn.LSTM(tcn_out, hidden_size, num_layers,
                                    batch_first=True, dropout=dropout if num_layers > 1 else 0,
                                    bidirectional=bidirectional)

                # FC
                fc_in = hidden_size * direction + input_size  # 跳跃连接
                self.fc = nn.Sequential(
                    nn.Linear(fc_in, 64),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(64, output_size),
                )

            def forward(self, x):
                raw_last = x[:, -1, :]  # 保留最后一步原始特征
                # TCN
                x_tcn = x.transpose(1, 2)
                x_tcn = self.tcn(x_tcn)
                x_tcn = x_tcn.transpose(1, 2)
                # Attention
                if self.use_attention:
                    x_attn, _ = self.attention(x_tcn, x_tcn, x_tcn)
                else:
                    x_attn = x_tcn
                # LSTM
                _, (hn, _) = self.lstm(x_attn)
                lstm_out = hn[-1]
                # 跳跃连接
                combined = torch.cat([lstm_out, raw_last], dim=1)
                return torch.relu(self.fc(combined))

        model = HybridModel(input_size=6, hidden_size=64, num_layers=2)
        state_dict = torch.load(CHARGING_MODEL_PTH, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        _charging_model = model
        _charging_loaded = True
        print(f"[INFO] 充电负荷模型已加载: {CHARGING_MODEL_PTH}")
        return model
    except Exception as e:
        print(f"[WARN] 充电模型加载失败: {e}")
        return None


# ============================================================
# 输入特征构建
# ============================================================
def _load_history_csv():
    """加载历史数据 CSV"""
    try:
        df = pd.read_csv(DATA_ALL_TRAIN, parse_dates=["timestamp"])
        return df
    except Exception:
        return None


def build_solar_input(weather_data, history_df=None, lookback=SOLAR_LOOKBACK):
    """
    构建光伏模型输入 (lookback, 5)
    特征: power, hour_sin, hour_cos, shortwave_radiation, direct_radiation
    """
    # 尝试从 aligned CSV 获取历史光伏数据
    aligned_path = DATA_ALIGNED
    if os.path.exists(aligned_path):
        try:
            adf = pd.read_csv(aligned_path, parse_dates=["datetime"])
            adf = adf.sort_values("datetime")
            # 取最后 lookback 行
            recent = adf.tail(lookback)
            features = np.zeros((lookback, 5), dtype=np.float32)
            for i, col in enumerate(SOLAR_FEATURE_COLS):
                if col in recent.columns:
                    features[:, i] = recent[col].values.astype(np.float32)
            return features
        except Exception as e:
            print(f"[WARN] 构建光伏输入失败(aligned CSV): {e}")

    # 回退：基于天气数据构造模拟历史
    radiation = weather_data.get("current_radiation", 500) if weather_data else 500
    cloudcover = weather_data.get("current_cloudcover", 30) if weather_data else 30
    cf = 1 - cloudcover / 100 * 0.7

    features = np.zeros((lookback, 5), dtype=np.float32)
    now = datetime.now()
    for i in range(lookback):
        t = now - timedelta(minutes=15 * (lookback - i))
        hour = t.hour + t.minute / 60
        features[i, 0] = max(0, radiation * cf * np.sin(np.pi * (hour - 6) / 12)) if 6 <= hour <= 18 else 0
        features[i, 1] = np.sin(2 * np.pi * hour / 24)
        features[i, 2] = np.cos(2 * np.pi * hour / 24)
        features[i, 3] = radiation
        features[i, 4] = radiation * 0.7  # 近似直接辐射
    return features


def build_charging_input(history_df=None, lookback=CHARGING_LOOKBACK):
    """
    构建充电模型输入 (lookback, 6)
    特征: price, lag_1, lag_96, lag_672, rolling_mean_4, rolling_std_4
    从 training CSV 提取
    """
    if history_df is None:
        history_df = _load_history_csv()

    if history_df is not None and len(history_df) >= lookback:
        feature_cols = ["price", "lag_1", "lag_96", "lag_672", "rolling_mean_4", "rolling_std_4"]
        available = [c for c in feature_cols if c in history_df.columns]
        if len(available) >= 4:  # 至少需要4个特征
            recent = history_df.tail(lookback)
            features = np.zeros((lookback, 6), dtype=np.float32)
            for i, col in enumerate(feature_cols):
                if col in recent.columns:
                    features[:, i] = recent[col].values.astype(np.float32)
                else:
                    features[:, i] = 0.0
            return features

    # 回退：模拟数据
    print("[WARN] 无法构建充电模型输入，使用模拟数据")
    features = np.zeros((lookback, 6), dtype=np.float32)
    t = np.linspace(0, 4 * np.pi, lookback)
    features[:, 0] = 0.6  # price
    features[:, 1] = 1000 + 200 * np.sin(t)  # lag_1
    features[:, 2] = 1000 + 200 * np.sin(t + 1)
    features[:, 3] = 1000 + 200 * np.sin(t - 2)
    features[:, 4] = 1000
    features[:, 5] = 50 + 30 * np.sin(t)
    return features


# ============================================================
# 预测主函数
# ============================================================
def run_prediction(n_steps, weather_data):
    """
    执行完整预测流水线
    Returns: dict {
        solar: array[n_steps],
        load_mean: array[n_steps],
        load_upper: array[n_steps],
        load_lower: array[n_steps],
        times: list[datetime],
        solar_peak, load_peak, green_ratio,
        model_status: dict
    }
    """
    status = {"solar_ok": False, "charging_ok": False}

    # 生成时间轴
    now = datetime.now().replace(second=0, microsecond=0)
    now = now - timedelta(minutes=now.minute % 15)  # 对齐到15min
    future_times = [now + timedelta(minutes=15 * i) for i in range(n_steps)]

    # ---- 光伏预测 ----
    solar_model = load_solar_model()
    if solar_model is not None and TORCH_AVAILABLE:
        try:
            solar_features = build_solar_input(weather_data, lookback=SOLAR_LOOKBACK)
            solar_pred = _rolling_solar_predict(solar_model, solar_features, n_steps, weather_data)
            status["solar_ok"] = True
            print(f"[INFO] 光伏预测完成 ({n_steps} 步)")
        except Exception as e:
            print(f"[ERROR] 光伏预测失败: {e}")
            solar_pred = _mock_solar(future_times, weather_data)
    else:
        solar_pred = _mock_solar(future_times, weather_data)

    # ---- 充电负荷预测 ----
    charging_model = load_charging_model()
    if charging_model is not None and TORCH_AVAILABLE:
        try:
            history_df = _load_history_csv()
            charging_features = build_charging_input(history_df, lookback=CHARGING_LOOKBACK)
            load_mean, load_std = _rolling_charging_predict(
                charging_model, charging_features, n_steps, samples=MC_DROPOUT_SAMPLES
            )
            status["charging_ok"] = True
            print(f"[INFO] 充电负荷预测完成 ({n_steps} 步)")
        except Exception as e:
            print(f"[ERROR] 充电预测失败: {e}")
            load_mean = _mock_load(future_times)
            load_std = load_mean * 0.1
    else:
        load_mean = _mock_load(future_times)
        load_std = load_mean * 0.1

    load_upper = load_mean + 1.96 * load_std
    load_lower = np.clip(load_mean - 1.96 * load_std, 0, None)

    # ---- 汇总指标 ----
    solar_peak = float(np.max(solar_pred))
    solar_peak_time = future_times[int(np.argmax(solar_pred))]
    load_peak = float(np.max(load_mean))
    total_solar = float(np.sum(solar_pred) * 0.25)
    total_load = float(np.sum(load_mean) * 0.25)
    green_ratio = min(total_solar / max(total_load, 0.01) * 100, 100)

    return {
        "solar": solar_pred,
        "load_mean": load_mean,
        "load_upper": load_upper,
        "load_lower": load_lower,
        "times": future_times,
        "solar_peak": solar_peak,
        "solar_peak_time": solar_peak_time,
        "load_peak": load_peak,
        "total_solar": total_solar,
        "total_load": total_load,
        "green_ratio": green_ratio,
        "model_status": status,
    }


# ============================================================
# 滚动预测
# ============================================================
def _rolling_solar_predict(model, initial_features, n_steps, weather_data):
    """光伏模型滚动预测"""
    window = initial_features.copy()
    predictions = []

    radiation = weather_data.get("current_radiation", 500) if weather_data else 500
    cloudcover = weather_data.get("current_cloudcover", 30) if weather_data else 30

    with torch.no_grad():
        for step in range(n_steps):
            x = torch.FloatTensor(window).unsqueeze(0).to(DEVICE)  # (1, lookback, 5)
            pred = model(x).item()
            pred = max(0, pred)  # 光伏出力非负
            predictions.append(pred)

            # 滚动窗口：用预测值追加
            new_row = window[-1].copy()
            new_row[0] = pred
            # 更新时间特征
            t = datetime.now() + timedelta(minutes=15 * (step + 1))
            hour = t.hour + t.minute / 60
            new_row[1] = np.sin(2 * np.pi * hour / 24)
            new_row[2] = np.cos(2 * np.pi * hour / 24)
            new_row[3] = radiation
            new_row[4] = radiation * 0.7
            window = np.vstack([window[1:], new_row])

    return np.array(predictions)


def _rolling_charging_predict(model, initial_features, n_steps, samples=30):
    """充电模型 MC Dropout 滚动预测"""
    window = initial_features.copy()
    all_predictions = np.zeros((n_steps, samples))

    for step in range(n_steps):
        x = torch.FloatTensor(window).unsqueeze(0).to(DEVICE)  # (1, lookback, 6)
        # MC Dropout: 多次采样
        model.train()  # 保持 dropout 开启
        step_preds = []
        for _ in range(samples):
            with torch.no_grad():
                p = model(x).item()
            step_preds.append(max(0, p))
        all_predictions[step, :] = step_preds

        # 滚动窗口
        new_row = window[-1].copy()
        new_row[1] = np.mean(step_preds)  # 用均值更新 lag_1
        window = np.vstack([window[1:], new_row])

    mean_pred = all_predictions.mean(axis=1)
    std_pred = all_predictions.std(axis=1)
    return mean_pred, std_pred


# ============================================================
# 模拟回退数据
# ============================================================
def _mock_solar(future_times, weather_data):
    """模拟光伏出力曲线"""
    radiation = weather_data.get("current_radiation", 500) if weather_data else 500
    cloudcover = weather_data.get("current_cloudcover", 30) if weather_data else 30
    cf = 1 - cloudcover / 100 * 0.7

    preds = []
    for t in future_times:
        hour = t.hour + t.minute / 60
        if 6 <= hour <= 18:
            base = radiation * cf * np.sin(np.pi * (hour - 6) / 12)
            noise = np.random.normal(0, base * 0.05)
            preds.append(max(0, base + noise))
        else:
            preds.append(max(0, np.random.normal(0, 5)))
    return np.array(preds)


def _mock_load(future_times):
    """模拟充电负荷曲线（基于日模式）"""
    preds = []
    for t in future_times:
        hour = t.hour + t.minute / 60
        # 模拟双峰：早8点、晚17点
        base = 800 + 300 * np.exp(-((hour - 8) / 3) ** 2) + 400 * np.exp(-((hour - 17) / 4) ** 2)
        noise = np.random.normal(0, 30)
        preds.append(max(50, base + noise))
    return np.array(preds)


# ============================================================
# 策略建议
# ============================================================
def generate_strategy(prediction_result):
    """基于预测结果生成策略建议"""
    solar = prediction_result["solar"]
    load = prediction_result["load_mean"]
    times = prediction_result["times"]
    green_ratio = prediction_result["green_ratio"]
    status = prediction_result["model_status"]

    lines = []

    # 模型状态
    model_lines = []
    if status["solar_ok"]:
        model_lines.append("✅ 光伏模型: CNN-LSTM 已加载")
    else:
        model_lines.append("⚠️ 光伏模型: 未加载，使用日模式模拟")
    if status["charging_ok"]:
        model_lines.append("✅ 充电模型: TCN-Attention-LSTM 已加载")
    else:
        model_lines.append("⚠️ 充电模型: 未加载，使用日模式模拟")

    lines.append("### 📊 预测总览")
    lines.append(f"- ☀️ 光伏总出力: **{prediction_result['total_solar']:.1f} kWh**")
    lines.append(f"- ⚡ 充电总需求: **{prediction_result['total_load']:.1f} kWh**")
    lines.append(f"- 🌿 绿电替代率: **{green_ratio:.1f}%**")

    lines.append("")
    lines.append("### 🔌 模型状态")
    lines.extend(model_lines)

    lines.append("")
    lines.append("### ⚖️ 供需平衡分析")
    net = solar - load
    surplus_steps = np.sum(net > 0)
    deficit_steps = np.sum(net < 0)
    total_steps = len(net)
    lines.append(f"- 光伏盈余时段: {surplus_steps}/{total_steps} 步 ({surplus_steps/total_steps*100:.0f}%)")
    lines.append(f"- 光伏不足时段: {deficit_steps}/{total_steps} 步 ({deficit_steps/total_steps*100:.0f}%)")
    if surplus_steps > 0:
        max_surplus_idx = np.argmax(net)
        lines.append(f"- 最大盈余: {net[max_surplus_idx]:.1f} kW @ {times[max_surplus_idx].strftime('%H:%M')}")
    if deficit_steps > 0:
        max_deficit_idx = np.argmin(net)
        lines.append(f"- 最大缺口: {-net[max_deficit_idx]:.1f} kW @ {times[max_deficit_idx].strftime('%H:%M')}")

    lines.append("")
    lines.append("### 💰 电价策略建议")

    from config import PRICE_PERIODS

    def get_price_period(hour):
        for period, ranges in PRICE_PERIODS.items():
            for s, e in ranges:
                if s <= hour < e:
                    return period
        return "mid"

    # 统计各时段情况
    valley_surplus = 0
    peak_deficit = 0
    for i, t in enumerate(times):
        period = get_price_period(t.hour)
        if period == "valley" and net[i] > 0:
            valley_surplus += net[i]
        elif period == "peak" and net[i] < 0:
            peak_deficit += abs(net[i])

    if valley_surplus > 0:
        lines.append(f"- 🔋 谷时光伏盈余约 {valley_surplus*0.25:.1f} kWh，建议在谷时集中充电，利用低价+绿电双重优势")
    if peak_deficit > 0:
        lines.append(f"- ⚡ 峰时光伏缺口约 {peak_deficit*0.25:.1f} kWh，建议减少峰时充电，优先使用谷时或光伏盈余时段")

    if green_ratio > 80:
        lines.append("- 🌞 光伏出力充裕，可优先消纳绿电")
    elif green_ratio < 30:
        lines.append("- ⚠️ 光伏出力不足，建议更多依赖谷时电网充电")

    lines.append("")
    lines.append("> ⚠️ 以上建议基于规则引擎生成，仅供策略参考")

    return "\n".join(lines)