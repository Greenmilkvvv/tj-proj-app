"""
数据探索服务模块 (Test3: LightGBM 版本)
提供历史数据加载、统计、相关性分析、日负荷曲线等功能，
并将回测改为 LightGBM 模型评估。
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

from config import (
    DATA_ALL_TRAIN, DATA_ALL_TEST, DATA_ALIGNED,
    DATA_SELECTED_TEST, ROOT_DIR,
    CHARGING_FEATURE_COLS, CHARGING_LOOKBACK, CHARGING_MODEL_PTH,
    SOLAR_MODEL_PTH, SOLAR_FEATURE_DIM,
)


# ============================================================
# 数据加载（懒加载 + 缓存）
# ============================================================
_cache = {}

def _load_csv(path):
    if path not in _cache:
        try:
            first_line = open(path, encoding='utf-8').readline()
            parse_dates_flag = ["timestamp"] if "timestamp" in first_line else None
            df = pd.read_csv(path, parse_dates=parse_dates_flag)
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


# ============================================================
# 数据集概览
# ============================================================
def get_dataset_overview():
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
    df = get_train_data()
    if df is None:
        return []
    time_col = "timestamp" if "timestamp" in df.columns else "datetime"
    if time_col not in df.columns:
        return []
    dates = sorted(df[time_col].dt.date.unique())
    return [str(d) for d in dates[:30]]


def plot_daily_load_curves(date_strs):
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
# LightGBM 训练历史
# ============================================================
LIGHTGBM_HISTORY_PKL = os.path.join(os.path.dirname(__file__), "lightgbm_train_history.pkl")


def get_training_loss_chart():
    """LightGBM 训练验证损失曲线"""
    if not os.path.exists(LIGHTGBM_HISTORY_PKL):
        fig = go.Figure()
        fig.add_annotation(text="训练历史文件不可用 (请先训练 LightGBM)", showarrow=False, font=dict(size=14))
        fig.update_layout(height=350, template="plotly_white")
        return fig, "⚠️ 训练历史文件不存在"
    try:
        with open(LIGHTGBM_HISTORY_PKL, "rb") as f:
            history = pickle.load(f)
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"加载失败: {e}", showarrow=False, font=dict(size=14))
        fig.update_layout(height=350, template="plotly_white")
        return fig, f"⚠️ 加载失败: {e}"

    train_losses = history.get("train_losses", [])
    val_losses = history.get("val_losses", [])
    best_iter = history.get("best_iteration", None)

    fig = go.Figure()
    iterations = list(range(1, len(train_losses) + 1))

    if train_losses:
        fig.add_trace(go.Scatter(
            x=iterations, y=train_losses,
            mode="lines",
            name="训练 Loss",
            line=dict(color="#2196f3", width=1.5),
        ))
    if val_losses:
        fig.add_trace(go.Scatter(
            x=iterations, y=val_losses,
            mode="lines",
            name="验证 Loss",
            line=dict(color="#f44336", width=1.5, dash="dot"),
        ))
    if best_iter is not None:
        fig.add_vline(x=best_iter, line_dash="dash", line_color="#4caf50",
                      annotation_text=f"Best Iter {best_iter}")

    fig.update_layout(
        title="充电模型 (LightGBM) 训练过程",
        xaxis=dict(title="Iteration"),
        yaxis=dict(title="Loss (MSE)"),
        margin=dict(l=40, r=40, t=40, b=40),
        template="plotly_white",
        height=350,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    final_loss = train_losses[-1] if train_losses else float("nan")
    best_info = f", best_iter={best_iter}" if best_iter else ""
    return fig, f"✅ 最终训练 Loss: {final_loss:.4f}{best_info}"


# ============================================================
# 充电模型回测 (LightGBM)
# ============================================================
def _lgb_backtest(df, feature_cols, target_col, lookback=96):
    """使用 LightGBM 模型进行滑动窗口回测"""
    if not os.path.exists(CHARGING_MODEL_PTH):
        print("[WARN] LightGBM 模型不存在，无法回测")
        return None, None, 0

    try:
        from importlib import reload
        import prediction_service
        reload(prediction_service)
        scaler_X, scaler_y = prediction_service._load_charging_scalers()
    except Exception:
        scaler_X, scaler_y = None, None

    with open(CHARGING_MODEL_PTH, 'rb') as f:
        model = pickle.load(f)

    data = df[feature_cols + [target_col]].dropna().values.astype(np.float64)
    n = len(data)
    if n < lookback + 10:
        return None, None, 0

    X_all, y_all = [], []
    for i in range(n - lookback):
        X_all.append(data[i:i + lookback, :-1].flatten())
        y_all.append(data[i + lookback, -1])
    X_all = np.array(X_all, dtype=np.float64)
    y_all = np.array(y_all, dtype=np.float64)

    split = int(len(X_all) * 0.6)
    if split < 2:
        return None, None, 0

    # 标准化特征
    if scaler_X is not None:
        X_test = scaler_X.transform(X_all[split:])
    else:
        from sklearn.preprocessing import StandardScaler
        s = StandardScaler()
        X_train_s = s.fit_transform(X_all[:split])
        X_test = s.transform(X_all[split:])

    y_true = y_all[split:]
    y_pred = model.predict(X_test)

    return y_true, y_pred, len(y_true)


def run_backtest_charging():
    """对充电负荷测试集进行 LightGBM 回测"""
    df = _load_csv(DATA_SELECTED_TEST)
    if df is None or len(df) < 100:
        fig = go.Figure()
        fig.add_annotation(text="测试数据不足", showarrow=False, font=dict(size=14))
        fig.update_layout(height=350, template="plotly_white")
        return fig, "⚠️ 数据不足"

    feature_cols = CHARGING_FEATURE_COLS
    target_col = "load_kw"

    for c in feature_cols + [target_col]:
        if c not in df.columns:
            fig = go.Figure()
            fig.add_annotation(text=f"缺失列: {c}", showarrow=False, font=dict(size=14))
            fig.update_layout(height=350, template="plotly_white")
            return fig, f"⚠️ 数据列缺失: {c}"

    y_true, y_pred, test_size = _lgb_backtest(df, feature_cols, target_col)
    if y_true is None:
        fig = go.Figure()
        fig.add_annotation(text="LightGBM 模型不可用 (请先训练)", showarrow=False, font=dict(size=14))
        fig.update_layout(height=350, template="plotly_white")
        return fig, "⚠️ LightGBM 模型未找到"

    residuals = y_true - y_pred
    rmse = np.sqrt(np.mean(residuals ** 2))
    mae = np.mean(np.abs(residuals))
    nonzero = np.abs(y_true) > 10
    mape = np.mean(np.abs(residuals[nonzero] / y_true[nonzero])) * 100 if nonzero.sum() > 0 else float("nan")

    n_show = min(200, len(y_true))
    idx = np.linspace(0, len(y_true) - 1, n_show).astype(int)
    t_show = y_true[idx]
    p_show = y_pred[idx]
    r_show = residuals[idx]

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.55, 0.45],
        subplot_titles=("真实值 vs 预测值 (LightGBM 回测)", "残差分布 (真实 - 预测)"),
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
        f"### 📊 LightGBM 回测指标 (n={len(y_true)})\n\n"
        f"| 指标 | 值 |\n"
        f"|------|----|\n"
        f"| RMSE | **{rmse:.2f}** kW |\n"
        f"| MAE  | **{mae:.2f}** kW |\n"
        f"| MAPE | **{mape:.1f}%** |\n"
        f"| 测试集大小 | {test_size} 个样本 |\n"
    )

    return fig, summary


def build_error_distribution_chart():
    """残差分布直方图 (LightGBM)"""
    df = _load_csv(DATA_SELECTED_TEST)
    if df is None:
        fig = go.Figure()
        fig.add_annotation(text="数据不可用", showarrow=False, font=dict(size=14))
        fig.update_layout(height=350, template="plotly_white")
        return fig

    y_true, y_pred, _ = _lgb_backtest(df, CHARGING_FEATURE_COLS, "load_kw")
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
        marker_color="#2196f3",
        opacity=0.75,
        name="残差",
    ))
    fig.add_vline(x=0, line_dash="dash", line_color="#f44336", line_width=2, annotation_text="0")

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
        title="残差分布直方图 (LightGBM)",
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
    """按小时分组的误差分析 (LightGBM)"""
    df = _load_csv(DATA_SELECTED_TEST)
    if df is None:
        fig = go.Figure()
        fig.add_annotation(text="数据不可用", showarrow=False, font=dict(size=14))
        fig.update_layout(height=350, template="plotly_white")
        return fig

    y_true, y_pred, _ = _lgb_backtest(df, CHARGING_FEATURE_COLS, "load_kw")
    if y_true is None:
        fig = go.Figure()
        fig.add_annotation(text="回测失败", showarrow=False, font=dict(size=14))
        fig.update_layout(height=350, template="plotly_white")
        return fig

    ts_col = "timestamp" if "timestamp" in df.columns else None
    if ts_col is None:
        hours = np.arange(len(y_true)) % 24
    else:
        df[ts_col] = pd.to_datetime(df[ts_col])
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
        title="按小时误差分析 — LightGBM (MAE & RMSE)",
        xaxis=dict(title="小时", dtick=2),
        yaxis=dict(title="误差 (kW)"),
        margin=dict(l=40, r=40, t=40, b=40),
        template="plotly_white",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ============================================================
# 光伏模型回测 (与 Test2 相同)
# ============================================================
def _solar_sliding_backtest(df, feature_cols, target_col="power", lookback=24):
    import sys
    import torch

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Solar_Forecast"))
    try:
        from NN import LSTMPredictor, GeneratorWithFeatures
    except ImportError:
        from prediction_service import LSTMPredictor, GeneratorWithFeatures

    scaler_path = os.path.join(ROOT_DIR, "Solar_Forecast", "simple_scaler_1.pkl")
    if not os.path.exists(scaler_path) or not os.path.exists(SOLAR_MODEL_PTH):
        return None, None

    try:
        import joblib
        scaler = joblib.load(scaler_path)
    except Exception:
        import pickle as _pkl
        with open(scaler_path, "rb") as f:
            scaler = _pkl.load(f)

    available = [c for c in feature_cols if c in df.columns]
    if len(available) < 4:
        return None, None

    data = df[feature_cols].values.astype(np.float64)
    n = len(data)
    if n < lookback + 10:
        return None, None

    data_scaled = scaler.transform(data)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
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
            p_min = scaler.data_min_[0]
            p_max = scaler.data_max_[0]
            pred_real = pred_scaled * (p_max - p_min) + p_min
            pred_real = max(pred_real, 0.0)
            y_pred.append(pred_real)
            y_true.append(data[i, 0])

    return np.array(y_true), np.array(y_pred)


def run_backtest_solar():
    aligned_path = DATA_ALIGNED
    if not os.path.exists(aligned_path):
        fig = go.Figure()
        fig.add_annotation(text="对齐数据文件不存在", showarrow=False, font=dict(size=14))
        fig.update_layout(height=350, template="plotly_white")
        return fig, "⚠️ 数据文件不存在"

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
    nonzero = np.abs(y_true) > 1
    mape = np.mean(np.abs(residuals[nonzero] / y_true[nonzero])) * 100 if nonzero.sum() > 0 else float("nan")
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

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
        x=residuals, nbinsx=50,
        marker_color="#4caf50", opacity=0.75,
        name="光伏残差",
    ))
    fig.add_vline(x=0, line_dash="dash", line_color="#f44336", line_width=2, annotation_text="0")

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


def get_solar_model_info():
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
        "#### 充电预测模型 (LightGBM)",
        "",
        "| 组件 | 参数 |",
        "|------|------|",
        "| 类型 | LightGBM Regressor |",
        "| 输入维度 | 576 (96步 × 6特征) |",
        "| 输出 | 1 步负荷 (kW) |",
        "| 训练策略 | 早停 (early_stopping) |",
        "",
    ]

    if os.path.exists(SOLAR_MODEL_PTH):
        try:
            import torch
            state = torch.load(SOLAR_MODEL_PTH, map_location="cpu", weights_only=True)
            param_count = sum(v.numel() for v in state.values())
            lines.append(f"| 光伏参数量 | {param_count:,} |")
        except Exception:
            pass

    return "\n".join(lines)