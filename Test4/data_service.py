"""
数据探索服务模块
提供历史数据加载、统计、相关性分析、日负荷曲线、误差分析等功能
"""
import os
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

from config import DATA_ALL_TRAIN, DATA_ALL_TEST, DATA_ALIGNED, DATA_SELECTED_TEST, ROOT_DIR

# 导入上传服务
try:
    from upload_service import get_charging_data, get_solar_data, has_uploaded_data
except ImportError:
    # 容错：如果 upload_service 不可用
    def get_charging_data():
        return _load_csv(DATA_SELECTED_TEST)
    def get_solar_data():
        if os.path.exists(DATA_ALIGNED):
            df = pd.read_csv(DATA_ALIGNED, parse_dates=["datetime"])
            df.sort_values("datetime", inplace=True)
            df["hour"] = pd.to_datetime(df["datetime"]).dt.hour
            df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
            df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
            return df
        return None
    def has_uploaded_data(x=None):
        return False


# ============================================================
# 数据加载（懒加载 + 缓存）
# ============================================================
_cache = {}

def _load_csv(path):
    """带缓存的 CSV 加载"""
    if path not in _cache:
        try:
            df = pd.read_csv(path, parse_dates=["timestamp"] if "timestamp" in open(path).readline() else None)
            if "datetime" in df.columns:
                df["datetime"] = pd.to_datetime(df["datetime"])
            _cache[path] = df
        except Exception as e:
            print(f"[ERROR] 加载 {path} 失败: {e}")
            return None
    return _cache[path]


def get_train_data():
    return _load_csv(DATA_ALL_TRAIN)

def get_test_data():
    return _load_csv(DATA_ALL_TEST)

def get_merged_charging():
    """获取合并后的充电数据 (历史 + 上传), 优先使用上传数据"""
    return get_charging_data()

def get_merged_solar():
    """获取合并后的光伏数据 (历史 + 上传), 优先使用上传数据"""
    return get_solar_data()


# ============================================================
# 数据集概览
# ============================================================
def get_dataset_overview():
    """返回数据集基本信息 HTML"""
    train_df = get_train_data()
    test_df = get_test_data()

    rows = []
    for name, df in [("训练集", train_df), ("测试集", test_df)]:
        if df is None:
            rows.append(f"<tr><td>{name}</td><td colspan='4' style='color:#dc3545;'>加载失败</td></tr>")
            continue
        date_range = "—"
        time_col = None
        for col in ["timestamp", "datetime"]:
            if col in df.columns:
                time_col = col
                break
        if time_col is not None and len(df) > 0:
            dmin = df[time_col].min()
            dmax = df[time_col].max()
            date_range = f"{dmin} ~ {dmax}"

        missing = df.isnull().sum().sum()
        rows.append(
            f"<tr>"
            f"<td><strong>{name}</strong></td>"
            f"<td>{len(df):,}</td>"
            f"<td>{len(df.columns)}</td>"
            f"<td>{date_range}</td>"
            f"<td>{missing}</td>"
            f"</tr>"
        )

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


# ============================================================
# 历史负荷曲线
# ============================================================
def get_available_dates():
    """获取可选日期列表"""
    df = get_train_data()
    if df is None:
        return []
    time_col = "timestamp" if "timestamp" in df.columns else "datetime"
    if time_col not in df.columns:
        return []
    dates = sorted(df[time_col].dt.date.unique())
    return [str(d) for d in dates[:30]]


def plot_daily_load_curves(date_strs):
    """绘制指定日期的负荷曲线（支持多日叠加）"""
    df = get_train_data()
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
            day_df = df[df[time_col].dt.date == d].sort_values(time_col)
            if len(day_df) == 0:
                continue
            times = [f"{t.hour:02d}:{t.minute:02d}" for t in day_df[time_col]]
            color = colors[i % len(colors)]
            fig.add_trace(go.Scatter(
                x=times, y=day_df[load_col].values,
                mode="lines+markers",
                name=str(d),
                line=dict(color=color, width=2),
                marker=dict(size=3),
            ))
        except Exception as e:
            print(f"[WARNING] 日期 {ds} 绘图失败: {e}")

    fig.update_layout(
        title="历史日负荷曲线对比",
        xaxis=dict(title="时间", tickangle=45, dtick=16),
        yaxis=dict(title="负荷 (kW)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=40, b=60),
        template="plotly_white",
        height=400,
    )
    return fig


# ============================================================
# 特征相关性分析
# ============================================================
def get_correlation_chart():
    """Pearson 相关系数条形图"""
    df = get_train_data()
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


# ============================================================
# 小时级负荷画像
# ============================================================
def get_hourly_profile_chart():
    """24小时负荷均值 ± 标准差"""
    df = get_train_data()
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


# ============================================================
# 误差分析 — 训练曲线
# ============================================================
TRAIN_HISTORY_PKL = os.path.join(ROOT_DIR, "Charging_Forecast", "final_train_history.pkl")


def get_training_loss_chart():
    """充电模型训练/验证损失曲线"""
    if not os.path.exists(TRAIN_HISTORY_PKL):
        fig = go.Figure()
        fig.add_annotation(text="训练历史文件不可用", showarrow=False, font=dict(size=14))
        fig.update_layout(height=350, template="plotly_white")
        return fig, "⚠️ 训练历史文件不存在"

    try:
        with open(TRAIN_HISTORY_PKL, "rb") as f:
            history = pickle.load(f)
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"加载失败: {e}", showarrow=False, font=dict(size=14))
        fig.update_layout(height=350, template="plotly_white")
        return fig, f"⚠️ 加载失败: {e}"

    train_losses = history.get("train_losses", [])
    val_logs = history.get("val_logs", [])

    fig = go.Figure()
    epochs = list(range(1, len(train_losses) + 1))

    if train_losses:
        fig.add_trace(go.Scatter(
            x=epochs, y=train_losses,
            mode="lines",
            name="训练 Loss",
            line=dict(color="#2196f3", width=1.5),
        ))
    if val_logs:
        if len(val_logs) == 1 and len(train_losses) > 1:
            fig.add_hline(
                y=val_logs[0], line_dash="dash", line_color="#f44336",
                annotation_text=f"验证 Loss = {val_logs[0]:.4f}",
            )
        else:
            fig.add_trace(go.Scatter(
                x=epochs, y=val_logs,
                mode="lines",
                name="验证 Loss",
                line=dict(color="#f44336", width=1.5, dash="dot"),
            ))

    fig.update_layout(
        title="充电模型 (TCN-Attention-LSTM) 训练过程",
        xaxis=dict(title="Epoch"),
        yaxis=dict(title="Loss (MSE)"),
        margin=dict(l=40, r=40, t=40, b=40),
        template="plotly_white",
        height=350,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    final_loss = train_losses[-1] if train_losses else float("nan")
    return fig, f"✅ 最终训练 Loss: {final_loss:.4f}"


# ============================================================
# 误差分析 — 充电模型回测 (真实 TCN-Attention-LSTM 模型)
# ============================================================
def _real_charging_sliding_backtest(df, feature_cols, target_col="load_kw", lookback=96):
    """充电滑动窗口回测：使用真实 TCN-Attention-LSTM 模型逐步预测"""
    import torch
    try:
        from prediction_service import HybridModel
    except ImportError:
        # 回退：直接导入
        import torch.nn as nn
        class HybridModel(nn.Module):
            pass  # 占位，实际上会从 prediction_service 加载

    from config import CHARGING_MODEL_PTH, CHARGING_FEATURE_DIM
    import pickle as pkl

    scaler_X_path = os.path.join(ROOT_DIR, "Charging_Forecast", "scaler_X.pkl")
    scaler_y_path = os.path.join(ROOT_DIR, "Charging_Forecast", "scaler_y.pkl")

    if not os.path.exists(scaler_X_path) or not os.path.exists(scaler_y_path):
        return None, None, "充电 scaler 文件不存在"
    if not os.path.exists(CHARGING_MODEL_PTH):
        return None, None, "充电模型权重不存在"

    # 加载 scaler
    with open(scaler_X_path, 'rb') as f:
        scaler_X = pkl.load(f)
    with open(scaler_y_path, 'rb') as f:
        scaler_y = pkl.load(f)

    # 确保数据列存在
    for c in feature_cols + [target_col]:
        if c not in df.columns:
            return None, None, f"数据缺失列: {c}"

    data = df[feature_cols].values.astype(np.float64)
    target = df[target_col].values.astype(np.float64)
    n = len(data)

    if n < lookback + 10:
        return None, None, f"数据不足 (n={n}, need >={lookback+10})"

    # 标准化特征
    from sklearn.preprocessing import MinMaxScaler
    data_scaled = scaler_X.transform(data)

    # 加载模型
    # 推断 hidden_size
    import torch.nn as nn
    # 从 prediction_service 导入模型定义
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from prediction_service import HybridModel, TCNBlock, SimpleAttention
    del sys.path[0]

    state = torch.load(CHARGING_MODEL_PTH, map_location="cpu", weights_only=True)
    # 推断 hidden_size
    hidden_size = CHARGING_FEATURE_DIM  # fallback
    for k in state:
        if "lstm.weight_ih_l0" in k:
            hidden_size = state[k].shape[0] // 4
            break

    device = "cpu"
    model = HybridModel(input_dim=CHARGING_FEATURE_DIM, hidden_dim=hidden_size, output_dim=1)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for i in range(lookback, n):
            # 取窗口 (lookback 步 x 6 特征)
            window = torch.FloatTensor(data_scaled[i - lookback:i]).unsqueeze(0).to(device)
            pred_scaled = model(window).item()

            # 反标准化
            pred_real = float(scaler_y.inverse_transform([[pred_scaled]])[0, 0])
            pred_real = max(pred_real, 0.0)

            y_pred.append(pred_real)
            y_true.append(target[i])

    return np.array(y_true), np.array(y_pred), "ok"


def run_backtest_charging():
    """对充电负荷数据进行回测（真实模型），优先使用上传数据，返回指标和图表"""
    # 优先使用合并数据 (含上传)
    df = get_merged_charging()
    using_upload = has_uploaded_data("charging")

    if df is None:
        fig = go.Figure()
        fig.add_annotation(text="数据不可用", showarrow=False, font=dict(size=14))
        fig.update_layout(height=350, template="plotly_white")
        return fig, "⚠️ 无可用数据"

    feature_cols = ["price", "lag_1", "lag_96", "lag_672", "rolling_std_4", "rolling_mean_4"]
    target_col = "load_kw"

    # 确保时间列为 datetime
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

    # 仅在上传数据区域运行回测（如果存在上传数据）
    if using_upload:
        try:
            from upload_service import _upload_cache
            upload_df = _upload_cache.get("charging")
            if upload_df is not None and len(upload_df) >= 100:
                upload_df = upload_df.copy()
                upload_df["timestamp"] = pd.to_datetime(upload_df["timestamp"])
                upload_df = upload_df.sort_values("timestamp").reset_index(drop=True)
                # 确保所有特征列存在
                for c in feature_cols:
                    if c not in upload_df.columns:
                        return _error_fig_and_msg(f"上传数据缺少特征列: {c}")
                df = upload_df  # 仅在上传数据上回测
        except Exception as e:
            print(f"[WARN] 获取上传数据失败: {e}")

    if len(df) < 100:
        fig = go.Figure()
        fig.add_annotation(text=f"数据不足 (n={len(df)}, 需要 ≥100)", showarrow=False, font=dict(size=14))
        fig.update_layout(height=350, template="plotly_white")
        return fig, f"⚠️ 数据不足 (仅 {len(df)} 行)"

    for c in feature_cols + [target_col]:
        if c not in df.columns:
            return _error_fig_and_msg(f"数据缺失列: {c}")

    result = _real_charging_sliding_backtest(df, feature_cols, target_col)
    if result[0] is None:
        return _error_fig_and_msg(result[2] if len(result) > 2 else "回测失败")

    y_true, y_pred, _ = result
    residuals = y_true - y_pred

    rmse = np.sqrt(np.mean(residuals ** 2))
    mae = np.mean(np.abs(residuals))
    nonzero = np.abs(y_true) > 10
    mape = np.mean(np.abs(residuals[nonzero] / y_true[nonzero])) * 100 if nonzero.sum() > 0 else float("nan")
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # 下采样展示
    n_show = min(300, len(y_true))
    idx = np.linspace(0, len(y_true) - 1, n_show).astype(int)
    t_show = y_true[idx]
    p_show = y_pred[idx]
    r_show = residuals[idx]

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.55, 0.45],
        subplot_titles=("真实值 vs 预测值 (真实 TCN-Attention-LSTM 模型)", "残差分布 (真实 - 预测)"),
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

    data_source = "**上传数据**" if using_upload else "历史测试集"
    summary = (
        f"### 📊 充电回测指标 ({data_source}, n={len(y_true)})\n\n"
        f"| 指标 | 值 |\n"
        f"|------|----|\n"
        f"| RMSE | **{rmse:.2f}** kW |\n"
        f"| MAE  | **{mae:.2f}** kW |\n"
        f"| MAPE | **{mape:.1f}%** (|load|>10kW) |\n"
        f"| R²   | **{r2:.4f}** |\n"
        f"\n> 使用真实 **TCN-Attention-LSTM** 模型滑动窗口滚动预测"
        + ("\n> 基于上传数据评估，反映真实模型性能" if using_upload else "")
    )

    return fig, summary


def _error_fig_and_msg(msg):
    """辅助函数：返回错误图和消息"""
    fig = go.Figure()
    fig.add_annotation(text=msg, showarrow=False, font=dict(size=14))
    fig.update_layout(height=350, template="plotly_white")
    return fig, f"⚠️ {msg}"


def build_error_distribution_chart():
    """残差分布直方图（带正态拟合）— 使用真实模型回测结果"""
    df = get_merged_charging()
    if df is None:
        return _error_fig_and_msg("数据不可用")[0]

    feature_cols = ["price", "lag_1", "lag_96", "lag_672", "rolling_std_4", "rolling_mean_4"]
    target_col = "load_kw"

    # 确保时间列
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

    result = _real_charging_sliding_backtest(df, feature_cols, target_col)
    if result[0] is None:
        return _error_fig_and_msg(result[2] if len(result) > 2 else "回测失败")[0]

    y_true, y_pred, _ = result
    residuals = y_true - y_pred

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=residuals,
        nbinsx=50,
        marker_color="#2196f3",
        opacity=0.75,
        name="残差",
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
        title="残差分布直方图 (TCN-Attention-LSTM)",
        xaxis=dict(title="残差 (kW)"),
        yaxis=dict(title="频次"),
        margin=dict(l=40, r=40, t=40, b=40),
        template="plotly_white",
        height=400,
        bargap=0.05,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def build_error_by_hour_chart():
    """按小时分组的误差分析 (MAE & RMSE) — 使用真实模型回测结果"""
    df = get_merged_charging()
    if df is None:
        return _error_fig_and_msg("数据不可用")[0]

    feature_cols = ["price", "lag_1", "lag_96", "lag_672", "rolling_std_4", "rolling_mean_4"]
    target_col = "load_kw"

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

    result = _real_charging_sliding_backtest(df, feature_cols, target_col)
    if result[0] is None:
        return _error_fig_and_msg(result[2] if len(result) > 2 else "回测失败")[0]

    y_true, y_pred, _ = result

    # 提取对应的小时
    ts_col = "timestamp" if "timestamp" in df.columns else None
    if ts_col is None:
        hours = np.arange(len(y_true)) % 24
    else:
        lookback = 96
        n_test = len(y_true)
        ts_all = df[ts_col].values
        if len(ts_all) < lookback + n_test:
            ts_test = ts_all[-n_test:]
        else:
            ts_test = ts_all[lookback: lookback + n_test]
        hours = np.array([pd.Timestamp(t).hour for t in ts_test[:n_test]])

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
        title="按小时误差分析 (TCN-Attention-LSTM, MAE & RMSE)",
        xaxis=dict(title="小时", dtick=2),
        yaxis=dict(title="误差 (kW)"),
        margin=dict(l=40, r=40, t=40, b=40),
        template="plotly_white",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ============================================================
# 误差分析 — 光伏模型回测
# ============================================================
def _solar_sliding_backtest(df, feature_cols, target_col="power", lookback=24):
    """光伏滑动窗口回测：使用真实 LSTM 模型逐步预测"""
    import sys
    import torch
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Solar_Forecast"))
    try:
        from NN import LSTMPredictor, GeneratorWithFeatures
    except ImportError:
        # 回退到 prediction_service 中的定义
        from prediction_service import LSTMPredictor, GeneratorWithFeatures

    from config import SOLAR_MODEL_PTH, SOLAR_FEATURE_DIM

    scaler_path = os.path.join(ROOT_DIR, "Solar_Forecast", "simple_scaler_1.pkl")
    if not os.path.exists(scaler_path) or not os.path.exists(SOLAR_MODEL_PTH):
        return None, None

    try:
        import joblib
        scaler = joblib.load(scaler_path)
    except Exception:
        import pickle
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

    # 确保 df 包含所有特征列
    available = [c for c in feature_cols if c in df.columns]
    if len(available) < 4:  # 至少需要 power + 3 个辐射列
        return None, None

    data = df[feature_cols].values.astype(np.float64)
    n = len(data)
    if n < lookback + 10:
        return None, None

    # 标准化
    data_scaled = scaler.transform(data)

    # 加载模型
    DEVICE = "cpu"
    lstm = LSTMPredictor(
        input_size=SOLAR_FEATURE_DIM, hidden_size=128,
        num_layers=2, output_size=1, dropout=0.2, bidirectional=False,
    )
    model = GeneratorWithFeatures(lstm)
    state = torch.load(SOLAR_MODEL_PTH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for i in range(lookback, n):
            inp = torch.FloatTensor(data_scaled[i - lookback:i]).unsqueeze(0).to(DEVICE)
            pred_scaled = model(inp).item()

            # 反标准化 (MinMaxScaler, power 在索引 0)
            p_min = scaler.data_min_[0]
            p_max = scaler.data_max_[0]
            pred_real = pred_scaled * (p_max - p_min) + p_min
            pred_real = max(pred_real, 0.0)

            y_pred.append(pred_real)
            y_true.append(data[i, 0])

    return np.array(y_true), np.array(y_pred)


def run_backtest_solar():
    """对光伏预测模型进行回测，返回指标和图表"""
    aligned_path = DATA_ALIGNED
    if not os.path.exists(aligned_path):
        fig = go.Figure()
        fig.add_annotation(text="对齐数据文件不存在", showarrow=False, font=dict(size=14))
        fig.update_layout(height=350, template="plotly_white")
        return fig, "⚠️ 数据文件不存在"

    df = pd.read_csv(aligned_path, parse_dates=["datetime"])
    df.sort_values("datetime", inplace=True)

    # 特征工程：与训练时保持一致
    df["hour"] = df["datetime"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    feature_cols = [
        "power", "hour_sin", "hour_cos",
        "shortwave_radiation (W/m2)", "direct_radiation (W/m2)",
        "diffuse_radiation (W/m2)", "direct_normal_irradiance (W/m2)",
    ]

    for c in feature_cols:
        if c not in df.columns:
            fig = go.Figure()
            fig.add_annotation(text=f"缺失列: {c}", showarrow=False, font=dict(size=14))
            fig.update_layout(height=350, template="plotly_white")
            return fig, f"⚠️ 数据列缺失: {c}"

    y_true, y_pred = _solar_sliding_backtest(df, feature_cols)
    if y_true is None:
        fig = go.Figure()
        fig.add_annotation(text="光伏模型加载失败或样本不足", showarrow=False, font=dict(size=14))
        fig.update_layout(height=350, template="plotly_white")
        return fig, "⚠️ 光伏模型不可用"

    residuals = y_true - y_pred
    rmse = np.sqrt(np.mean(residuals ** 2))
    mae = np.mean(np.abs(residuals))
    nonzero = np.abs(y_true) > 1  # 光伏功率阈值 1kW（低于充电的10kW）
    mape = np.mean(np.abs(residuals[nonzero] / y_true[nonzero])) * 100 if nonzero.sum() > 0 else float("nan")

    # R² 计算
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # 下采样展示（只取白天有功率的时段，晚间0功率无意义）
    daytime_mask = y_true > 0.5
    daytime_indices = np.where(daytime_mask)[0]
    n_show = min(300, len(daytime_indices))
    if n_show > 0:
        idx = np.linspace(0, len(daytime_indices) - 1, n_show).astype(int)
        show_indices = daytime_indices[idx]
        t_show = y_true[show_indices]
        p_show = y_pred[show_indices]
        r_show = residuals[show_indices]
    else:
        n_all = min(200, len(y_true))
        idx = np.linspace(0, len(y_true) - 1, n_all).astype(int)
        t_show = y_true[idx]
        p_show = y_pred[idx]
        r_show = residuals[idx]

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.55, 0.45],
        subplot_titles=("光伏真实值 vs 预测值 (回测·日间时段)", "残差分布 (真实 - 预测)"),
        vertical_spacing=0.12,
    )

    fig.add_trace(go.Scatter(
        x=list(range(len(t_show))), y=t_show,
        mode="lines", name="真实值",
        line=dict(color="#4caf50", width=2),
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

    fig.update_xaxes(title_text="样本序号 (仅日间)", row=2, col=1)
    fig.update_yaxes(title_text="功率 (kW)", row=1, col=1)
    fig.update_yaxes(title_text="残差 (kW)", row=2, col=1)
    fig.update_layout(
        height=550,
        template="plotly_white",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=50, b=40),
    )

    summary = (
        f"### ☀️ 光伏回测指标 (n={len(y_true)})\n\n"
        f"| 指标 | 值 |\n"
        f"|------|----|\n"
        f"| RMSE | **{rmse:.2f}** kW |\n"
        f"| MAE  | **{mae:.2f}** kW |\n"
        f"| MAPE | **{mape:.1f}%** (功率>1kW) |\n"
        f"| R²   | **{r2:.4f}** |\n"
    )

    return fig, summary


def build_solar_error_distribution_chart():
    """光伏残差分布直方图（带正态拟合）"""
    aligned_path = DATA_ALIGNED
    if not os.path.exists(aligned_path):
        fig = go.Figure()
        fig.add_annotation(text="数据不可用", showarrow=False, font=dict(size=14))
        fig.update_layout(height=350, template="plotly_white")
        return fig

    df = pd.read_csv(aligned_path, parse_dates=["datetime"])
    df.sort_values("datetime", inplace=True)
    df["hour"] = df["datetime"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    feature_cols = [
        "power", "hour_sin", "hour_cos",
        "shortwave_radiation (W/m2)", "direct_radiation (W/m2)",
        "diffuse_radiation (W/m2)", "direct_normal_irradiance (W/m2)",
    ]
    y_true, y_pred = _solar_sliding_backtest(df, feature_cols)
    if y_true is None:
        fig = go.Figure()
        fig.add_annotation(text="回测失败", showarrow=False, font=dict(size=14))
        fig.update_layout(height=350, template="plotly_white")
        return fig

    residuals = y_true - y_pred

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=residuals,
        nbinsx=50,
        marker_color="#4caf50",
        opacity=0.75,
        name="光伏残差",
    ))
    fig.add_vline(
        x=0, line_dash="dash", line_color="#f44336", line_width=2,
        annotation_text="0",
    )

    # 正态拟合参考线
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
        title="光伏残差分布直方图",
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
    aligned_path = DATA_ALIGNED
    if not os.path.exists(aligned_path):
        fig = go.Figure()
        fig.add_annotation(text="数据不可用", showarrow=False, font=dict(size=14))
        fig.update_layout(height=350, template="plotly_white")
        return fig

    df = pd.read_csv(aligned_path, parse_dates=["datetime"])
    df.sort_values("datetime", inplace=True)
    df["hour"] = df["datetime"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    feature_cols = [
        "power", "hour_sin", "hour_cos",
        "shortwave_radiation (W/m2)", "direct_radiation (W/m2)",
        "diffuse_radiation (W/m2)", "direct_normal_irradiance (W/m2)",
    ]
    y_true, y_pred = _solar_sliding_backtest(df, feature_cols)
    if y_true is None:
        fig = go.Figure()
        fig.add_annotation(text="回测失败", showarrow=False, font=dict(size=14))
        fig.update_layout(height=350, template="plotly_white")
        return fig

    # 提取对应的小时
    lookback = 24
    n_test = len(y_true)
    ts_all = df["datetime"].values
    if len(ts_all) < lookback + n_test:
        ts_test = ts_all[-n_test:]
    else:
        ts_test = ts_all[lookback: lookback + n_test]
    hours = np.array([pd.Timestamp(t).hour for t in ts_test[:n_test]])

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
        marker_color="#4caf50",
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
        title="光伏按小时误差分析 (MAE & RMSE)",
        xaxis=dict(title="小时", dtick=2),
        yaxis=dict(title="误差 (kW)"),
        margin=dict(l=40, r=40, t=40, b=40),
        template="plotly_white",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ============================================================
# 误差分析 — 光伏模型性能参考
# ============================================================
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
        "",
        # "> ⚠️ 光伏模型训练历史文件未单独保存，无法展示训练 Loss 曲线。",
    ]

    return "\n".join(lines)