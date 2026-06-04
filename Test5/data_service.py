"""
数据处理与可视化引擎 (v3)
===========================
移除上传依赖，直接使用内置数据集 (Data/*.csv) 作为"后台实时同步"数据。
保留所有图表展示逻辑不变。
"""
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from config import (
    ROOT_DIR, DATA_ALIGNED, DATA_SELECTED_TEST,
    PREDICTION_OPTIONS, DEFAULT_PREDICTION_STEPS,
)


# ============================================================
# 内置数据加载器
# ============================================================
def load_builtin_data():
    """
    加载内置数据集作为"后台实时同步"数据
    返回:
      - aligned_df: 光伏历史数据 (aligned_2026_01_02.csv)
      - charging_df: 充电历史数据 (dataset_selected_features_test.csv)
    """
    aligned_df = None
    if os.path.exists(DATA_ALIGNED):
        aligned_df = pd.read_csv(DATA_ALIGNED, parse_dates=["datetime"])

    charging_df = None
    charging_path = os.path.join(ROOT_DIR, "Data", "dataset_selected_features_test.csv")
    if os.path.exists(charging_path):
        charging_df = pd.read_csv(charging_path, parse_dates=["timestamp"])

    return aligned_df, charging_df


# ============================================================
# 历史数据展示
# ============================================================
def get_recent_charging_data(hours=24):
    """获取最近 N 小时的充电历史记录 (用于面板展示)"""
    _, df = load_builtin_data()
    if df is None:
        return []
    df = df.sort_values("timestamp")
    # 截取最近 hours*4 行 (15分钟一个)
    n_rows = min(hours * 4, len(df))
    recent = df.tail(n_rows)
    records = []
    for _, row in recent.iterrows():
        records.append({
            "time": row["timestamp"].strftime("%m-%d %H:%M") if hasattr(row["timestamp"], "strftime") else str(row["timestamp"]),
            "load": round(float(row.get("load_kw", 0)), 1),
            "price": round(float(row.get("price", 0)), 3),
        })
    return records


def get_recent_solar_data(hours=24):
    """获取最近 N 小时的光伏历史记录"""
    df, _ = load_builtin_data()
    if df is None:
        return []
    df = df.sort_values("datetime")
    n_rows = min(hours * 4, len(df))
    recent = df.tail(n_rows)
    records = []
    for _, row in recent.iterrows():
        val = float(row.get("power", 0)) if "power" in row else 0.0
        records.append({
            "time": row["datetime"].strftime("%m-%d %H:%M") if hasattr(row["datetime"], "strftime") else str(row["datetime"]),
            "power": round(val, 1),
        })
    return records


# ============================================================
# 图表引擎 (保持与 Test4 相同)
# ============================================================
def build_time_features(result):
    """从预测结果构建时间特征 DataFrame"""
    if not PLOTLY_AVAILABLE:
        return pd.DataFrame()
    times = result["times"]
    records = []
    for i, t in enumerate(times):
        ts_str = t.strftime("%m-%d %H:%M")
        records.append({
            "time": ts_str,
            "hour": round(t.hour + t.minute / 60, 2),
            "solar": round(float(result["solar"][i]), 2),
            "load": round(float(result["load_mean"][i]), 2),
            "load_lower": round(float(result["load_lower"][i]), 2),
            "load_upper": round(float(result["load_upper"][i]), 2),
            "net": round(float(result["solar"][i] - result["load_mean"][i]), 2),
        })
    return pd.DataFrame(records)


def create_prediction_chart(result):
    """主预测图表：光伏 + 充电负荷 + 不确定性区间"""
    if not PLOTLY_AVAILABLE:
        return None
    df = build_time_features(result)
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.55, 0.45],
        subplot_titles=("光伏出力预测 + 充电负荷预测", "净负荷 (光伏 - 充电)"),
    )

    # Row 1: 光伏出力
    fig.add_trace(
        go.Scatter(
            x=df["time"], y=df["solar"],
            # mode="lines+markers",
            mode="lines",
            name="光伏出力 (kW)",
            line=dict(color="#ff9800", width=2),
            marker=dict(size=5),
            fill="tozeroy",
            fillcolor="rgba(255,152,0,0.15)",
        ),
        row=1, col=1,
    )

    # Row 1: 充电负荷 + 置信区间
    fig.add_trace(
        go.Scatter(
            x=df["time"], y=df["load"],
            # mode="lines+markers",
            mode="lines",
            name="充电负荷 (kW)",
            line=dict(color="#2196f3", width=2),
            marker=dict(size=5),
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=pd.concat([df["time"], df["time"][::-1]]),
            y=pd.concat([df["load_upper"], df["load_lower"][::-1]]),
            fill="toself",
            fillcolor="rgba(33,150,243,0.15)",
            line=dict(color="rgba(33,150,243,0.1)", width=0),
            name="负荷 95%-105% 区间",
        ),
        row=1, col=1,
    )

    # Row 2: 净负荷
    colors = ["#4caf50" if v >= 0 else "#f44336" for v in df["net"]]
    fig.add_trace(
        go.Bar(
            x=df["time"], y=df["net"],
            name="净负荷",
            marker_color=colors,
            opacity=0.8,
        ),
        row=2, col=1,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=2, col=1)

    fig.update_layout(
        height=550,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        margin=dict(l=40, r=40, t=50, b=40),
    )
    fig.update_xaxes(title_text="时间", row=2, col=1)
    fig.update_yaxes(title_text="kW", row=1, col=1)
    fig.update_yaxes(title_text="kW", row=2, col=1)

    return fig


def create_summary_chart(result):
    """摘要饼图 + 柱状图"""
    if not PLOTLY_AVAILABLE:
        return None
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "pie"}, {"type": "bar"}]],
        subplot_titles=("绿电替代率", "关键指标"),
    )

    green_ratio = result.get("green_ratio", 0)
    # Pie
    fig.add_trace(
        go.Pie(
            labels=["绿电覆盖", "电网供电"],
            values=[green_ratio, max(100 - green_ratio, 0)],
            marker_colors=["#4caf50", "#e0e0e0"],
            hole=0.4,
            textinfo="label+percent",
        ),
        row=1, col=1,
    )

    # Bar
    categories = ["总光伏 (kWh)", "总负荷 (kWh)", "峰值光伏 (kW)", "峰值负荷 (kW)"]
    values = [
        round(result.get("total_solar", 0), 1),
        round(result.get("total_load", 0), 1),
        round(result.get("solar_peak", 0), 1),
        round(result.get("load_peak", 0), 1),
    ]
    colors_bar = ["#ff9800", "#2196f3", "#ff9800", "#2196f3"]
    fig.add_trace(
        go.Bar(x=categories, y=values, marker_color=colors_bar, text=values, textposition="outside"),
        row=1, col=2,
    )

    fig.update_layout(
        height=350,
        template="plotly_white",
        margin=dict(l=40, r=40, t=40, b=40),
        showlegend=False,
    )
    return fig


# ============================================================
# 误差分析图表 (基于内置数据)
# ============================================================
# ============================================================
# 误差分析 — 充电模型回测 (真实 TCN-Attention-LSTM)
# ============================================================
def _charging_sliding_backtest():
    """充电滑动窗口回测内部函数，返回 (y_true, y_pred, hours) 或 (None, None, None)"""
    try:
        import pickle
        import torch
    except ImportError:
        return None, None, None

    from config import CHARGING_MODEL_PTH
    from prediction_service import HybridModel, _load_charging_scalers

    if not os.path.exists(CHARGING_MODEL_PTH):
        return None, None, None

    test_path = os.path.join(ROOT_DIR, "Data", "dataset_selected_features_test.csv")
    if not os.path.exists(test_path):
        return None, None, None

    df_test = pd.read_csv(test_path, parse_dates=["timestamp"])
    df_test.sort_values("timestamp", inplace=True)

    scaler_X, scaler_y = _load_charging_scalers()
    if scaler_X is None or scaler_y is None:
        return None, None, None

    feature_cols = ["price", "lag_1", "lag_96", "lag_672", "rolling_std_4", "rolling_mean_4"]
    target_col = "load_kw"

    state = torch.load(CHARGING_MODEL_PTH, map_location="cpu", weights_only=True)
    for k in state:
        if "lstm.weight_ih_l0" in k:
            hidden_dim = state[k].shape[0] // 4
            break
    else:
        hidden_dim = 121

    model = HybridModel(input_dim=6, hidden_dim=hidden_dim, output_dim=1)
    model.load_state_dict(state)
    model.eval()

    seq_len = 96
    X_test = df_test[feature_cols].values.astype(np.float32)
    y_test = df_test[target_col].values.astype(np.float32)
    X_scaled = scaler_X.transform(X_test)

    preds = []
    actuals = []
    hours = []

    for i in range(seq_len, len(X_scaled)):
        inp = torch.FloatTensor(X_scaled[i - seq_len : i]).unsqueeze(0)
        with torch.no_grad():
            pred_scaled = model(inp).item()
        pred_real = scaler_y.inverse_transform([[pred_scaled]])[0, 0]
        preds.append(float(pred_real))
        actuals.append(float(y_test[i]))
        try:
            ts = df_test["timestamp"].iloc[i]
            hours.append(ts.hour)
        except Exception:
            hours.append(0)

    if len(preds) < 10:
        return None, None, None

    return np.array(actuals), np.array(preds), np.array(hours)


def run_backtest_charging():
    """充电模型回测图：真实值 vs 预测值 + 残差分布"""
    if not PLOTLY_AVAILABLE:
        return go.Figure(), "⚠️ Plotly 不可用"

    y_true, y_pred, _ = _charging_sliding_backtest()
    if y_true is None:
        fig = go.Figure()
        fig.add_annotation(text="充电模型或数据不可用", showarrow=False, font=dict(size=14))
        fig.update_layout(height=350, template="plotly_white")
        return fig, "⚠️ 充电模型回测不可用"

    residuals = y_true - y_pred
    rmse = np.sqrt(np.mean(residuals ** 2))
    mae = np.mean(np.abs(residuals))
    wape = np.sum(np.abs(residuals)) / np.sum(np.abs(y_true)) * 100 if np.sum(np.abs(y_true)) > 0 else float("nan")
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    n_show = min(300, len(y_true))
    idx = np.linspace(0, len(y_true) - 1, n_show).astype(int)
    t_show = y_true[idx]
    p_show = y_pred[idx]
    r_show = residuals[idx]

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.55, 0.45],
        subplot_titles=("真实值 vs 预测值 (TCN-Attention-LSTM 模型)", "残差分布 (真实 - 预测)"),
        vertical_spacing=0.12,
    )
    fig.add_trace(go.Scatter(
        x=list(range(len(t_show))), y=t_show,
        mode="lines", name="真实值",
        line=dict(color="#2196f3", width=2),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=list(range(len(p_show))), y=p_show,
        mode="lines", name="预测值",
        line=dict(color="#ff9800", width=1.5, dash="dot"),
    ), row=1, col=1)

    colors_res = ["#4caf50" if v >= 0 else "#f44336" for v in r_show]
    fig.add_trace(go.Bar(
        x=list(range(len(r_show))), y=r_show,
        marker_color=colors_res, name="残差", opacity=0.6,
    ), row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="#666", row=2, col=1)

    fig.update_xaxes(title_text="样本序号", row=2, col=1)
    fig.update_yaxes(title_text="负荷 (kW)", row=1, col=1)
    fig.update_yaxes(title_text="残差 (kW)", row=2, col=1)
    fig.update_layout(
        height=550,
        template="plotly_white",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=50, b=40),
    )

    summary = (
        f"### 📊 充电回测指标 (n={len(y_true)})\n\n"
        f"| 指标 | 值 |\n"
        f"|------|----|\n"
        f"| RMSE | **{rmse:.2f}** kW |\n"
        f"| MAE  | **{mae:.2f}** kW |\n"
        f"| WAPE | **{wape:.1f}%** |\n"
        f"| R²   | **{r2:.4f}** |\n"
        f"\n> 使用真实 **TCN-Attention-LSTM** 模型滑动窗口滚动预测"
    )

    return fig, summary


def build_charging_error_distribution_chart():
    """充电残差分布直方图（带正态拟合）"""
    if not PLOTLY_AVAILABLE:
        return go.Figure()

    y_true, y_pred, _ = _charging_sliding_backtest()
    if y_true is None:
        return _error_fallback("充电模型回测不可用")

    residuals = y_true - y_pred

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=residuals,
        nbinsx=50,
        marker_color="#2196f3",
        opacity=0.75,
        name="充电残差",
    ))
    fig.add_vline(
        x=0, line_dash="dash", line_color="#f44336", line_width=2,
        annotation_text="0",
    )

    mu, sigma = np.mean(residuals), np.std(residuals)
    x_span = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 200)
    y_pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_span - mu) / sigma) ** 2)
    bin_width = (residuals.max() - residuals.min()) / 50 if residuals.max() > residuals.min() else 1
    y_scaled = y_pdf * len(residuals) * bin_width
    fig.add_trace(go.Scatter(
        x=x_span, y=y_scaled,
        mode="lines",
        name=f"正态拟合 (μ={mu:.1f}, σ={sigma:.1f})",
        line=dict(color="#f44336", width=2, dash="dot"),
    ))

    fig.update_layout(
        title="充电残差分布直方图 (TCN-Attention-LSTM)",
        xaxis=dict(title="残差 (kW)"),
        yaxis=dict(title="频次"),
        margin=dict(l=40, r=40, t=40, b=40),
        template="plotly_white",
        height=400,
        bargap=0.05,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def build_charging_error_by_hour_chart():
    """充电按小时分组的误差分析 (MAE & RMSE)"""
    if not PLOTLY_AVAILABLE:
        return go.Figure()

    y_true, y_pred, hours = _charging_sliding_backtest()
    if y_true is None:
        return _error_fallback("充电模型回测不可用")

    error_df = pd.DataFrame({
        "hour": hours,
        "abs_error": np.abs(y_true - y_pred),
        "sq_error": (y_true - y_pred) ** 2,
    })
    hourly = error_df.groupby("hour").agg(
        MAE=("abs_error", "mean"),
        RMSE=("sq_error", lambda x: np.sqrt(x.mean())),
        count=("sq_error", "count"),
    ).reset_index()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=hourly["hour"], y=hourly["MAE"],
        name="MAE (kW)",
        marker_color="#2196f3",
        opacity=0.7,
    ))
    fig.add_trace(go.Scatter(
        x=hourly["hour"], y=hourly["RMSE"],
        mode="lines+markers",
        name="RMSE (kW)",
        line=dict(color="#f44336", width=2),
        marker=dict(size=6),
    ))

    fig.update_layout(
        title="充电按小时误差分析 (TCN-Attention-LSTM, MAE & RMSE)",
        xaxis=dict(title="小时", dtick=2),
        yaxis=dict(title="误差 (kW)"),
        margin=dict(l=40, r=40, t=40, b=40),
        template="plotly_white",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ============================================================
# 误差分析 — 光伏模型回测 (LSTM + GAN)
# ============================================================
def _solar_sliding_backtest():
    """光伏滑动窗口回测内部函数，返回 (y_true, y_pred, hours) 或 (None, None, None)"""
    try:
        import torch
    except ImportError:
        return None, None, None

    from config import SOLAR_MODEL_PTH
    from prediction_service import LSTMPredictor, GeneratorWithFeatures, _load_solar_scaler, _load_aligned_df

    if not os.path.exists(SOLAR_MODEL_PTH):
        return None, None, None

    df = _load_aligned_df()
    if df is None:
        return None, None, None

    scaler = _load_solar_scaler()
    if scaler is None:
        return None, None, None

    feature_cols = [
        "power", "hour_sin", "hour_cos",
        "shortwave_radiation (W/m2)", "direct_radiation (W/m2)",
        "diffuse_radiation (W/m2)", "direct_normal_irradiance (W/m2)"
    ]

    state = torch.load(SOLAR_MODEL_PTH, map_location="cpu", weights_only=True)
    lstm = LSTMPredictor(input_size=7, hidden_size=128, num_layers=2, output_size=1, dropout=0.2)
    model = GeneratorWithFeatures(lstm)
    model.load_state_dict(state)
    model.eval()

    seq_len = 24
    X = df[feature_cols].values.astype(np.float32)
    y = df["power"].values.astype(np.float32)
    X_scaled = scaler.transform(X)

    preds = []
    actuals = []
    hours = []

    for i in range(seq_len, len(X_scaled)):
        inp = torch.FloatTensor(X_scaled[i - seq_len : i]).unsqueeze(0)
        with torch.no_grad():
            pred_scaled = model(inp).item()
        pred_row = X_scaled[i - 1].copy()
        pred_row[0] = pred_scaled
        pred_real = scaler.inverse_transform(pred_row.reshape(1, -1))[0, 0]
        preds.append(float(max(pred_real, 0)))
        actuals.append(float(y[i]))
        try:
            ts = df["datetime"].iloc[i]
            hours.append(ts.hour)
        except Exception:
            hours.append(0)

    if len(preds) < 10:
        return None, None, None

    return np.array(actuals), np.array(preds), np.array(hours)


def run_backtest_solar():
    """光伏模型回测图：真实值 vs 预测值 + 残差分布"""
    if not PLOTLY_AVAILABLE:
        return go.Figure(), "⚠️ Plotly 不可用"

    y_true, y_pred, _ = _solar_sliding_backtest()
    if y_true is None:
        fig = go.Figure()
        fig.add_annotation(text="光伏模型或数据不可用", showarrow=False, font=dict(size=14))
        fig.update_layout(height=350, template="plotly_white")
        return fig, "⚠️ 光伏模型回测不可用"

    residuals = y_true - y_pred
    rmse = np.sqrt(np.mean(residuals ** 2))
    mae = np.mean(np.abs(residuals))
    wape = np.sum(np.abs(residuals)) / np.sum(np.abs(y_true)) * 100 if np.sum(np.abs(y_true)) > 0 else float("nan")
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    n_show = min(300, len(y_true))
    idx = np.linspace(0, len(y_true) - 1, n_show).astype(int)
    t_show = y_true[idx]
    p_show = y_pred[idx]
    r_show = residuals[idx]

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.55, 0.45],
        subplot_titles=("真实值 vs 预测值 (LSTM + GAN 模型)", "残差分布 (真实 - 预测)"),
        vertical_spacing=0.12,
    )
    fig.add_trace(go.Scatter(
        x=list(range(len(t_show))), y=t_show,
        mode="lines", name="真实值",
        line=dict(color="#ff9800", width=2),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=list(range(len(p_show))), y=p_show,
        mode="lines", name="预测值",
        line=dict(color="#2196f3", width=1.5, dash="dot"),
    ), row=1, col=1)

    colors_res = ["#4caf50" if v >= 0 else "#f44336" for v in r_show]
    fig.add_trace(go.Bar(
        x=list(range(len(r_show))), y=r_show,
        marker_color=colors_res, name="残差", opacity=0.6,
    ), row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="#666", row=2, col=1)

    fig.update_xaxes(title_text="样本序号", row=2, col=1)
    fig.update_yaxes(title_text="光伏功率 (kW)", row=1, col=1)
    fig.update_yaxes(title_text="残差 (kW)", row=2, col=1)
    fig.update_layout(
        height=550,
        template="plotly_white",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=50, b=40),
    )

    summary = (
        f"### 📊 光伏回测指标 (n={len(y_true)})\n\n"
        f"| 指标 | 值 |\n"
        f"|------|----|\n"
        f"| RMSE | **{rmse:.2f}** kW |\n"
        f"| MAE  | **{mae:.2f}** kW |\n"
        f"| WAPE | **{wape:.1f}%** |\n"
        f"| R²   | **{r2:.4f}** |\n"
        f"\n> 使用真实 **LSTM + GAN** 模型滑动窗口滚动预测"
    )

    return fig, summary


def build_solar_error_distribution_chart():
    """光伏残差分布直方图（带正态拟合）"""
    if not PLOTLY_AVAILABLE:
        return go.Figure()

    y_true, y_pred, _ = _solar_sliding_backtest()
    if y_true is None:
        return _error_fallback("光伏模型回测不可用")

    residuals = y_true - y_pred

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=residuals,
        nbinsx=50,
        marker_color="#ff9800",
        opacity=0.75,
        name="光伏残差",
    ))
    fig.add_vline(
        x=0, line_dash="dash", line_color="#f44336", line_width=2,
        annotation_text="0",
    )

    mu, sigma = np.mean(residuals), np.std(residuals)
    x_span = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 200)
    y_pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_span - mu) / sigma) ** 2)
    bin_width = (residuals.max() - residuals.min()) / 50 if residuals.max() > residuals.min() else 1
    y_scaled = y_pdf * len(residuals) * bin_width
    fig.add_trace(go.Scatter(
        x=x_span, y=y_scaled,
        mode="lines",
        name=f"正态拟合 (μ={mu:.1f}, σ={sigma:.1f})",
        line=dict(color="#f44336", width=2, dash="dot"),
    ))

    fig.update_layout(
        title="光伏残差分布直方图 (LSTM + GAN)",
        xaxis=dict(title="残差 (kW)"),
        yaxis=dict(title="频次"),
        margin=dict(l=40, r=40, t=40, b=40),
        template="plotly_white",
        height=400,
        bargap=0.05,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def build_solar_error_by_hour_chart():
    """光伏按小时分组的误差分析 (MAE & RMSE)"""
    if not PLOTLY_AVAILABLE:
        return go.Figure()

    y_true, y_pred, hours = _solar_sliding_backtest()
    if y_true is None:
        return _error_fallback("光伏模型回测不可用")

    error_df = pd.DataFrame({
        "hour": hours,
        "abs_error": np.abs(y_true - y_pred),
        "sq_error": (y_true - y_pred) ** 2,
    })
    hourly = error_df.groupby("hour").agg(
        MAE=("abs_error", "mean"),
        RMSE=("sq_error", lambda x: np.sqrt(x.mean())),
        count=("sq_error", "count"),
    ).reset_index()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=hourly["hour"], y=hourly["MAE"],
        name="MAE (kW)",
        marker_color="#ff9800",
        opacity=0.7,
    ))
    fig.add_trace(go.Scatter(
        x=hourly["hour"], y=hourly["RMSE"],
        mode="lines+markers",
        name="RMSE (kW)",
        line=dict(color="#f44336", width=2),
        marker=dict(size=6),
    ))

    fig.update_layout(
        title="光伏按小时误差分析 (LSTM + GAN, MAE & RMSE)",
        xaxis=dict(title="小时", dtick=2),
        yaxis=dict(title="误差 (kW)"),
        margin=dict(l=40, r=40, t=40, b=40),
        template="plotly_white",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _error_fallback(msg):
    """当误差分析失败时，返回一个带有消息的简单图表"""
    if not PLOTLY_AVAILABLE:
        return None
    fig = go.Figure()
    fig.add_annotation(
        x=0.5, y=0.5,
        text=f"⚠️ {msg}",
        showarrow=False,
        font=dict(size=16, color="#888"),
    )
    fig.update_layout(
        height=400,
        template="plotly_white",
    )
    return fig


# ============================================================
# 模型信息展示
# ============================================================
def get_charging_model_info():
    """充电模型 (TCN-Attention-LSTM) 的基本信息"""
    from config import CHARGING_MODEL_PTH, CHARGING_FEATURE_DIM

    lines = [
        "### 🔋 充电负荷预测模型 (TCN-Attention-LSTM)",
        "",
        "#### 模型架构",
        "| 组件 | 参数 |",
        "|------|------|",
        "| 类型 | TCN + Attention + LSTM |",
        "| TCN 层 | 5层, dilations=[1,2,4,8,16] |",
        "| Attention | Simple Attention |",
        "| LSTM | 1层, hidden=动态推断 |",
        "| 输入窗口 | 96 步 (24 小时) |",
        "| 输出 | 1 步 (15分钟) |",
        "",
    ]

    if os.path.exists(CHARGING_MODEL_PTH):
        try:
            import torch
            state = torch.load(CHARGING_MODEL_PTH, map_location="cpu", weights_only=True)
            param_count = sum(v.numel() for v in state.values())
            lines.append(f"| 参数量 | {param_count:,} |")
            lines.append(f"| 权重文件 | `Charging_Retraining/best_pth/final_best_hybrid_model.pth` |")
        except Exception:
            lines.append("| 权重文件 | `Charging_Retraining/best_pth/final_best_hybrid_model.pth` |")

    lines += [
        "",
        "#### 预测模式 (v3 新增)",
        "充电预测改为 **Input-Driven** 模式:",
        "1. 用户输入当前电价 (元/kWh) 和当前充电负荷 (kW)",
        "2. 系统将输入追加到内置历史数据末尾",
        "3. 模型基于更新后的窗口进行迭代预测",
        "",
        "三种预测时长可选:",
        "- **15min 单步**: 1步预测",
        "- **1h 预测**: 4步预测",
        "- **24h 预测**: 96步预测",
    ]

    return "\n".join(lines)


# ============================================================
# 数据探索函数 (基于内置数据)
# ============================================================
def _get_data_for_exploration():
    """获取数据探索用的 DataFrame（优先使用充电测试数据）"""
    _, charging_df = load_builtin_data()
    return charging_df


def get_dataset_overview():
    """返回数据集基本信息 HTML"""
    from config import DATA_ALIGNED
    charging_df = _get_data_for_exploration()
    _, aligned_df = load_builtin_data()

    rows = []

    # 充电数据
    if charging_df is not None:
        time_col = "timestamp" if "timestamp" in charging_df.columns else "datetime"
        if time_col in charging_df.columns and len(charging_df) > 0:
            dmin = charging_df[time_col].min()
            dmax = charging_df[time_col].max()
            date_range = f"{str(dmin)[:16]} ~ {str(dmax)[:16]}"
        else:
            date_range = "—"
        missing = charging_df.isnull().sum().sum()
        rows.append(
            f"<tr>"
            f"<td><strong>充电负荷数据</strong></td>"
            f"<td>{len(charging_df):,}</td>"
            f"<td>{len(charging_df.columns)}</td>"
            f"<td>{date_range}</td>"
            f"<td>{missing}</td>"
            f"</tr>"
        )

    # 光伏数据
    if aligned_df is not None:
        time_col = "datetime"
        if time_col in aligned_df.columns and len(aligned_df) > 0:
            dmin = aligned_df[time_col].min()
            dmax = aligned_df[time_col].max()
            date_range = f"{str(dmin)[:16]} ~ {str(dmax)[:16]}"
        else:
            date_range = "—"
        missing = aligned_df.isnull().sum().sum()
        rows.append(
            f"<tr>"
            f"<td><strong>光伏历史数据</strong></td>"
            f"<td>{len(aligned_df):,}</td>"
            f"<td>{len(aligned_df.columns)}</td>"
            f"<td>{date_range}</td>"
            f"<td>{missing}</td>"
            f"</tr>"
        )

    if not rows:
        return "<p style='color:#888;'>未找到内置数据集，请检查 Data/ 目录下的 CSV 文件。</p>"

    return f"""
    <table style='width:100%; border-collapse:collapse; font-size:14px;'>
    <thead>
        <tr style='background:#f5f5f5;'>
            <th style='padding:10px; text-align:left;'>数据集</th>
            <th style='padding:10px; text-align:right;'>行数</th>
            <th style='padding:10px; text-align:right;'>列数</th>
            <th style='padding:10px; text-align:left;'>日期范围</th>
            <th style='padding:10px; text-align:right;'>缺失值</th>
        </tr>
    </thead>
    <tbody>
        {''.join(rows)}
    </tbody>
    </table>
    """


def get_available_dates():
    """获取可选日期列表"""
    df = _get_data_for_exploration()
    if df is None:
        return []
    time_col = "timestamp" if "timestamp" in df.columns else "datetime"
    if time_col not in df.columns:
        return []
    dates = sorted(df[time_col].dt.date.unique())
    return [str(d) for d in dates[:30]]


def plot_daily_load_curves(date_strs):
    """绘制指定日期的负荷曲线（支持多日叠加）"""
    df = _get_data_for_exploration()
    if df is None or not date_strs:
        return go.Figure()

    time_col = "timestamp" if "timestamp" in df.columns else "datetime"
    if time_col not in df.columns:
        return go.Figure()

    load_col = "load_kw"
    if load_col not in df.columns:
        return go.Figure()

    df[time_col] = pd.to_datetime(df[time_col])

    fig = go.Figure()
    colors = ["#2196f3", "#f44336", "#4caf50", "#ff9800", "#9c27b0", "#00bcd4"]

    for i, ds in enumerate(date_strs):
        try:
            d = pd.to_datetime(ds).date()
            # 严格按日期和小时范围过滤：只取当天 00:00 ~ 23:59 的数据
            day_mask = df[time_col].dt.date == d
            day_df = df[day_mask].sort_values(time_col).copy()
            if len(day_df) == 0:
                continue

            # 确保数据严格在当天范围内（排除跨天数据混入）
            day_start = pd.Timestamp(d)
            day_end = day_start + pd.Timedelta(days=1)
            day_df = day_df[(day_df[time_col] >= day_start) & (day_df[time_col] < day_end)]

            if len(day_df) == 0:
                continue

            # 使用连续的分钟索引作为 x 轴，避免首尾因"分类"x轴而产生的连线问题
            day_df["minutes"] = day_df[time_col].dt.hour * 60 + day_df[time_col].dt.minute
            times = day_df["minutes"].values / 60.0  # 转为小时 (0.0 ~ 23.75)

            color = colors[i % len(colors)]

            # 在 points > 1 且间隔过大时不连线，避免跨夜连接
            connect_gaps = True
            if len(day_df) > 1:
                diff = np.diff(day_df["minutes"].values)
                if np.any(diff > 60):  # 存在超过 1 小时的间隔
                    connect_gaps = False

            fig.add_trace(go.Scatter(
                x=times,
                y=day_df[load_col].values,
                mode="lines+markers",
                name=str(d),
                line=dict(color=color, width=2),
                marker=dict(size=3),
                connectgaps=connect_gaps,
            ))
        except Exception as e:
            print(f"[WARNING] 日期 {ds} 绘图失败: {e}")

    fig.update_layout(
        title="历史日负荷曲线对比",
        xaxis=dict(
            title="当日时间 (小时)",
            tickangle=0,
            dtick=2,
            range=[0, 24],
            tickmode="linear",
        ),
        yaxis=dict(title="负荷 (kW)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=40, b=60),
        template="plotly_white",
        height=400,
    )
    return fig


def get_correlation_chart():
    """Pearson 相关系数条形图"""
    df = _get_data_for_exploration()
    if df is None or "load_kw" not in df.columns:
        return go.Figure()

    exclude = ["timestamp", "datetime", "load_kw"]
    numeric_cols = [c for c in df.columns if c not in exclude and df[c].dtype in ["float64", "int64"]]
    if not numeric_cols:
        return go.Figure()

    corrs = {}
    for col in numeric_cols:
        valid = df[[col, "load_kw"]].dropna()
        if len(valid) > 10:
            corr = valid[col].corr(valid["load_kw"])
            corrs[col] = corr

    if not corrs:
        return go.Figure()

    sorted_items = sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True)
    labels = [k for k, v in sorted_items]
    values = [v for k, v in sorted_items]
    colors = ["#4caf50" if v > 0 else "#f44336" for v in values]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=labels, x=values,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.3f}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        title="特征与负荷 (load_kw) 的 Pearson 相关系数",
        xaxis=dict(title="相关系数", range=[-1, 1]),
        yaxis=dict(title="特征"),
        margin=dict(l=150, r=40, t=40, b=40),
        template="plotly_white",
        height=max(300, len(labels) * 25),
    )
    return fig


def get_hourly_profile_chart():
    """24小时负荷均值 ± 标准差"""
    df = _get_data_for_exploration()
    if df is None or "load_kw" not in df.columns:
        return go.Figure()

    time_col = "timestamp" if "timestamp" in df.columns else "datetime"
    if time_col not in df.columns:
        return go.Figure()

    df[time_col] = pd.to_datetime(df[time_col])
    df["hour"] = df[time_col].dt.hour

    stats = df.groupby("hour")["load_kw"].agg(["mean", "std"]).reset_index()
    stats["upper"] = stats["mean"] + stats["std"]
    stats["lower"] = (stats["mean"] - stats["std"]).clip(0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(stats["hour"]) + list(stats["hour"])[::-1],
        y=list(stats["upper"]) + list(stats["lower"])[::-1],
        fill="toself",
        fillcolor="rgba(33,150,243,0.2)",
        line=dict(color="rgba(255,255,255,0)"),
        name="±1σ",
        showlegend=True,
    ))
    fig.add_trace(go.Scatter(
        x=stats["hour"], y=stats["mean"],
        mode="lines+markers",
        name="均值",
        line=dict(color="#1976d2", width=2.5),
        marker=dict(size=6),
    ))
    fig.update_layout(
        title="小时级负荷画像 (均值 ± 1σ)",
        xaxis=dict(title="小时", dtick=2),
        yaxis=dict(title="负荷 (kW)"),
        margin=dict(l=40, r=40, t=40, b=40),
        template="plotly_white",
        height=400,
    )
    return fig


def get_solar_model_info():
    """光伏模型 (LSTM + GAN) 的基本信息"""
    from config import SOLAR_MODEL_PTH

    lines = [
        "### ☀️ 光伏预测模型 (LSTM + GAN)",
        "",
        "#### 模型架构",
        "| 组件 | 参数 |",
        "|------|------|",
        "| 类型 | LSTM + GAN 对抗训练 |",
        "| 隐藏层 | 2层 LSTM, hidden=128 |",
        "| Dropout | 0.2 |",
        "| 输入窗口 | 24 步 (6 小时) |",
        "| 输出 | 1 步 (15分钟) |",
        "",
    ]

    if os.path.exists(SOLAR_MODEL_PTH):
        try:
            import torch
            state = torch.load(SOLAR_MODEL_PTH, map_location="cpu", weights_only=True)
            param_count = sum(v.numel() for v in state.values())
            lines.append(f"| 参数量 | {param_count:,} |")
            lines.append(f"| 权重文件 | `Solar_Forecast/best_pth/best_generator.pth` |")
        except Exception:
            lines.append("| 权重文件 | `Solar_Forecast/best_pth/best_generator.pth` |")

    lines += [
        "",
        "#### 说明",
        "光伏预测采用 LSTM 加 GAN 对抗训练框架：",
        "1. **Generator**: LSTM 生成预测功率",
        "2. **Discriminator**: 判别器鉴别生成 vs 真实功率",
        "3. **对抗训练**: 提升预测分布的真实性",
    ]

    return "\n".join(lines)