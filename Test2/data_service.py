"""
数据探索服务模块
提供历史数据加载、统计、相关性分析、日负荷曲线等功能
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

from config import DATA_ALL_TRAIN, DATA_ALL_TEST


# ============================================================
# 数据加载（懒加载 + 缓存）
# ============================================================
_cache = {}

def _load_csv(path):
    """带缓存的 CSV 加载"""
    if path not in _cache:
        try:
            df = pd.read_csv(path, parse_dates=["timestamp"] if "timestamp" in open(path).readline() else None)
            # 兼容 aligned CSV 的 datetime 列
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
    return [str(d) for d in dates[:30]]  # 限制30个日期


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
            # 生成时间轴 (15min粒度)
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

    # 排除非数值列
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

    # 按绝对值降序排列
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
    # 误差带
    fig.add_trace(go.Scatter(
        x=list(stats["hour"]) + list(stats["hour"])[::-1],
        y=list(stats["upper"]) + list(stats["lower"])[::-1],
        fill="toself",
        fillcolor="rgba(33,150,243,0.2)",
        line=dict(color="rgba(255,255,255,0)"),
        name="±1σ",
        showlegend=True,
    ))
    # 均值线
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