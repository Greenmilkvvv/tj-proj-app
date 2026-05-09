"""
光储充智能预测 Demo v2 — Gradio 主应用
变化: 数据上传独立为第一个 Tab, 数据探索精简, UI 优化
"""
import os
import sys
import gradio as gr
import plotly.graph_objects as go
from datetime import datetime
import threading
import time
import warnings
warnings.filterwarnings("ignore")

# 导入服务模块
from config import APP_TITLE, APP_PORT, APP_THEME, PREDICTION_OPTIONS, DEFAULT_PREDICTION_STEPS
from weather_service import fetch_weather_data, build_radiation_chart, get_current_weather_summary
from data_service import (
    get_dataset_overview, get_available_dates, plot_daily_load_curves,
    get_correlation_chart, get_hourly_profile_chart,
    get_training_loss_chart, run_backtest_charging,
    build_error_distribution_chart, build_error_by_hour_chart,
    get_solar_model_info,
    run_backtest_solar,
    build_solar_error_distribution_chart, build_solar_error_by_hour_chart,
    get_merged_charging, get_merged_solar,
)
from prediction_service import run_prediction, generate_strategy
from upload_service import save_uploaded_charging, save_uploaded_solar, has_uploaded_data, clear_uploaded_data
from generate_sample_data import inject_sample_data

# ============================================================
# 全局缓存
# ============================================================
_weather_cache = None
_weather_cache_time = 0
WEATHER_CACHE_TTL = 300  # 5 分钟
_upload_data_available = False  # 上传状态标志

# ============================================================
# CSS 样式 (v2 优化)
# ============================================================
CUSTOM_CSS = """
.gradio-container { max-width: 1400px !important; margin: auto; }
.warning-box {
    background: #fff3cd; border: 1px solid #ffc107; border-radius: 8px;
    padding: 12px; margin: 10px 0; color: #856404;
}
.info-box {
    background: #d1ecf1; border: 1px solid #17a2b8; border-radius: 8px;
    padding: 12px; margin: 10px 0; color: #0c5460;
}
.success-box {
    background: #d4edda; border: 1px solid #28a745; border-radius: 8px;
    padding: 12px; margin: 10px 0; color: #155724;
}
.title-section { text-align: center; margin-bottom: 20px; }
.title-section h1 { font-size: 28px; color: #2c3e50; margin-bottom: 6px; }
.title-section p { font-size: 14px; color: #7f8c8d; }
.card-grid {
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 12px 0;
}
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 10px; padding: 16px; color: white; text-align: center;
    box-shadow: 0 2px 10px rgba(0,0,0,0.15);
}
.metric-card.green { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }
.metric-card.orange { background: linear-gradient(135deg, #f2994a 0%, #f2c94c 100%); }
.metric-value { font-size: 28px; font-weight: bold; }
.metric-label { font-size: 13px; opacity: 0.9; margin-top: 4px; }
footer { visibility: hidden; }

/* ---- Dark mode ---- */
.dark .upload-card h3,
.dark .upload-card p,
.dark .upload-card .prose {
    color: #e0e0e0 !important;
}
.dark .upload-card {
    background: #1e1e1e !important;
    border-color: #444 !important;
}

.dark table thead th,
.dark .table-wrap thead th,
.dark table th {
    color: #e0e0e0 !important;
    background-color: #2a2a2a !important;
}
.dark table tbody td,
.dark .table-wrap tbody td {
    color: #d0d0d0 !important;
}

/* ---- v2 新增: 数据状态指示器 ---- */
.status-indicator {
    display: inline-flex; align-items: center; gap: 8px;
    padding: 8px 16px; border-radius: 20px; font-size: 14px; font-weight: 500;
}
.status-indicator.ready {
    background: #d4edda; color: #155724; border: 1px solid #c3e6cb;
}
.status-indicator.pending {
    background: #fff3cd; color: #856404; border: 1px solid #ffeeba;
}
.status-indicator.error {
    background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb;
}

/* ---- 上传区域卡片 ---- */
.upload-card {
    background: var(--card-bg, #f8f9fa); border-radius: 12px;
    padding: 24px; border: 2px dashed var(--border-color, #dee2e6);
    transition: all 0.3s ease;
}
.upload-card:hover {
    border-color: #667eea;
}
"""

# ============================================================
# 气象数据获取（带缓存）
# ============================================================
def get_weather():
    global _weather_cache, _weather_cache_time
    now = time.time()
    if _weather_cache is not None and now - _weather_cache_time < WEATHER_CACHE_TTL:
        return _weather_cache
    data = fetch_weather_data()
    _weather_cache = data
    _weather_cache_time = now
    return data

# ============================================================
# 预测结果缓存
# ============================================================
_last_prediction = None

def get_last_prediction():
    return _last_prediction

# ============================================================
# Tab 1: 数据上传（新增，原数据探索中的上传部分）
# ============================================================
def build_upload_tab():
    """构建数据上传 Tab（独立为第一个 Tab）"""
    gr.Markdown("""
    ## 📤 上传自定义数据
    
    上传您的 CSV 文件以扩展数据集，上传后可立即在「预测核心」和「数据探索」中使用。
    文件需包含时间列及负荷/功率列，系统会自动计算缺失的衍生特征。
    """)

    # ---- 示例数据开关 ----
    with gr.Row():
        use_sample = gr.Checkbox(
            label="☑️ 使用示例数据（模拟 2026-03-01 ~ 当前日期）",
            value=False,
            info="勾选后将自动填充充电负荷和光伏出力的模拟数据，无需手动上传 CSV",
        )

    with gr.Row():
        # 充电数据上传
        with gr.Column(scale=1, elem_classes="upload-card"):
            gr.Markdown("### ⚡ 充电负荷数据")
            gr.Markdown("*要求: `timestamp` + `load_kw` 列*")
            upload_charging = gr.File(
                label="选择充电 CSV 文件",
                file_types=[".csv"],
                type="filepath",
            )
            charging_status = gr.Markdown("📋 等待上传")

        # 光伏数据上传
        with gr.Column(scale=1, elem_classes="upload-card"):
            gr.Markdown("### ☀️ 光伏功率数据")
            gr.Markdown("*要求: `datetime` + `power` 列*")
            upload_solar = gr.File(
                label="选择光伏 CSV 文件",
                file_types=[".csv"],
                type="filepath",
            )
            solar_status = gr.Markdown("📋 等待上传")

    # 状态总览
    gr.Markdown("---")
    with gr.Row():
        global_status = gr.Markdown(
            value=_build_upload_status_html(),
            elem_classes="status-indicator pending",
        )
        btn_clear = gr.Button("🗑️ 清除所有上传数据", variant="secondary", size="sm")

    # 数据预览
    gr.Markdown("### 📊 上传数据预览")
    with gr.Row():
        preview_charging = gr.Dataframe(
            label="充电数据预览（前5行）",
            value=None,
            interactive=False,
        )
        preview_solar = gr.Dataframe(
            label="光伏数据预览（前5行）",
            value=None,
            interactive=False,
        )

    # ---- 事件绑定 ----
    upload_charging.upload(
        fn=_handle_charging_upload_v2,
        inputs=[upload_charging],
        outputs=[charging_status, global_status, preview_charging],
    )

    upload_solar.upload(
        fn=_handle_solar_upload_v2,
        inputs=[upload_solar],
        outputs=[solar_status, global_status, preview_solar],
    )

    btn_clear.click(
        fn=_handle_clear_upload_v2,
        inputs=[],
        outputs=[charging_status, solar_status, global_status, preview_charging, preview_solar],
    )

    use_sample.change(
        fn=_handle_use_sample,
        inputs=[use_sample],
        outputs=[charging_status, solar_status, global_status, preview_charging, preview_solar],
    )

    return upload_charging, upload_solar, global_status

def _build_upload_status_html():
    """生成状态指示器 HTML"""
    has_charging = has_uploaded_data("charging")
    has_solar = has_uploaded_data("solar")
    
    if has_charging and has_solar:
        cls = "ready"
        icon = "✅"
        text = "数据就绪 — 充电 & 光伏数据均已上传"
    elif has_charging:
        cls = "pending"
        icon = "⚠️"
        text = "部分就绪 — 仅充电数据可用，光伏将使用内置数据"
    elif has_solar:
        cls = "pending"
        icon = "⚠️"
        text = "部分就绪 — 仅光伏数据可用，充电将使用内置数据"
    else:
        cls = "pending"
        icon = "📋"
        text = "使用内置数据集 — 上传自定义数据以获得更精准预测"
    
    return f'<div class="status-indicator {cls}">{icon} {text}</div>'

def _handle_charging_upload_v2(filepath):
    """处理充电数据上传 (v2 增强版)"""
    global _upload_data_available
    import pandas as pd
    if filepath is None:
        return "📋 等待上传", _build_upload_status_html(), None
    try:
        msg, ok = save_uploaded_charging(filepath)
        if ok:
            _upload_data_available = True
            
            # 生成预览
            from upload_service import get_charging_data
            df = get_charging_data()
            preview = df.head(5) if df is not None else None
            return f"✅ {msg}", _build_upload_status_html(), preview
        else:
            return f"⚠️ {msg}", _build_upload_status_html(), None
    except Exception as e:
        return f"❌ 上传失败: {str(e)}", _build_upload_status_html(), None

def _handle_solar_upload_v2(filepath):
    """处理光伏数据上传 (v2 增强版)"""
    global _upload_data_available
    import pandas as pd
    if filepath is None:
        return "📋 等待上传", _build_upload_status_html(), None
    try:
        msg, ok = save_uploaded_solar(filepath)
        if ok:
            _upload_data_available = True
            
            # 生成预览
            from upload_service import get_solar_data
            df = get_solar_data()
            preview = df.head(5) if df is not None else None
            return f"✅ {msg}", _build_upload_status_html(), preview
        else:
            return f"⚠️ {msg}", _build_upload_status_html(), None
    except Exception as e:
        return f"❌ 上传失败: {str(e)}", _build_upload_status_html(), None

def _handle_clear_upload_v2():
    """清除上传数据 (v2)"""
    global _upload_data_available
    _upload_data_available = False
    msg = clear_uploaded_data()
    return "📋 等待上传", "📋 等待上传", _build_upload_status_html(), None, None


def _handle_use_sample(checked: bool):
    """处理示例数据复选框切换"""
    global _upload_data_available
    import pandas as pd
    from upload_service import _upload_cache

    if checked:
        # 生成并注入示例数据
        try:
            charging_df, solar_df = inject_sample_data()
            # 自动补充 lag 特征（与正常上传流程一致）
            from upload_service import _compute_lag_features
            charging_df = _compute_lag_features(charging_df)
            # 直接注入缓存（绕过 CSV 解析）
            _upload_cache["charging"] = charging_df
            _upload_cache["solar"] = solar_df
            _upload_data_available = True

            charging_status = (
                f"✅ 示例充电数据就绪\n"
                f"📅 时间范围: {charging_df['timestamp'].min()} ~ {charging_df['timestamp'].max()}\n"
                f"📊 共 {len(charging_df)} 行"
            )
            solar_status = (
                f"✅ 示例光伏数据就绪\n"
                f"📅 时间范围: {solar_df['datetime'].min()} ~ {solar_df['datetime'].max()}\n"
                f"📊 共 {len(solar_df)} 行"
            )
            preview_c = charging_df.head(5)
            preview_s = solar_df.head(5)
            return charging_status, solar_status, _build_upload_status_html(), preview_c, preview_s
        except Exception as e:
            return f"❌ 生成示例数据失败: {e}", "📋 等待上传", _build_upload_status_html(), None, None
    else:
        # 清除示例数据
        _upload_cache.clear()
        _upload_data_available = False
        return "📋 等待上传", "📋 等待上传", _build_upload_status_html(), None, None

# ============================================================
# Tab 2: 预测核心
# ============================================================
def build_prediction_tab():
    """构建预测 Tab 的 UI 组件"""
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ 预测参数")
            
            # 数据状态指示器
            data_status = gr.HTML(value=_build_upload_status_html())
            
            forecast_option = gr.Dropdown(
                choices=list(PREDICTION_OPTIONS.keys()),
                value="6 小时",
                label="预测时长",
            )
            btn_run = gr.Button("🚀 执行预测", variant="primary", size="lg")
            
            # 预测指标（紧凑布局）
            with gr.Accordion("📈 预测指标 & 峰谷信息", open=True):
                with gr.Row():
                    solar_total = gr.Textbox(label="☀️ 光伏总出力 (kWh)", value="—")
                    load_total = gr.Textbox(label="⚡ 充电总需求 (kWh)", value="—")
                with gr.Row():
                    green_ratio = gr.Textbox(label="🌿 绿电替代率", value="—")
                    solar_peak_info = gr.Textbox(label="🔆 光伏峰值", value="—")
                with gr.Row():
                    load_peak_info = gr.Textbox(label="📊 负荷峰值", value="—")
                    model_status_md = gr.Markdown("")

        with gr.Column(scale=2):
            forecast_chart = gr.Plot(label="预测曲线")
            forecast_balance_chart = gr.Plot(label="供需平衡")

    # 事件绑定
    btn_run.click(
        fn=_run_prediction_ui,
        inputs=[forecast_option],
        outputs=[forecast_chart, forecast_balance_chart,
                 solar_total, load_total, green_ratio,
                 solar_peak_info, load_peak_info, model_status_md],
    )

    return forecast_option, btn_run

def _run_prediction_ui(option_label):
    """执行预测并更新 UI"""
    global _last_prediction
    n_steps = PREDICTION_OPTIONS.get(option_label, DEFAULT_PREDICTION_STEPS)
    weather = get_weather()
    result = run_prediction(n_steps, weather)
    _last_prediction = result

    # 图表1: 光伏 + 负荷预测
    fig1 = go.Figure()
    times_str = [t.strftime("%H:%M") for t in result["times"]]

    # 光伏预测
    fig1.add_trace(go.Scatter(
        x=times_str, y=result["solar"],
        mode="lines",
        name="☀️ 光伏预测",
        line=dict(color="#ff9800", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(255,152,0,0.15)",
        yaxis="y1",
    ))

    # 负荷预测（含置信区间）
    fig1.add_trace(go.Scatter(
        x=times_str, y=result["load_mean"],
        mode="lines",
        name="⚡ 充电负荷预测",
        line=dict(color="#2196f3", width=2.5),
        yaxis="y1",
    ))
    fig1.add_trace(go.Scatter(
        x=times_str + times_str[::-1],
        y=list(result["load_upper"]) + list(result["load_lower"])[::-1],
        fill="toself",
        fillcolor="rgba(33,150,243,0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        name="95% 置信区间",
    ))

    fig1.update_layout(
        title=f"光伏出力 & 充电负荷预测 ({option_label})",
        xaxis=dict(title="时间", tickangle=45),
        yaxis=dict(title="功率 (kW)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=40, b=40),
        template="plotly_white",
        height=400,
    )

    # 图表2: 供需平衡 (净功率)
    net = result["solar"] - result["load_mean"]
    colors = ["#4caf50" if v >= 0 else "#f44336" for v in net]

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=times_str, y=net,
        marker_color=colors,
        name="光伏-负荷",
    ))
    fig2.add_hline(y=0, line_dash="dash", line_color="#666")
    fig2.update_layout(
        title="供需平衡 (光伏盈余 > 0)",
        xaxis=dict(title="时间", tickangle=45),
        yaxis=dict(title="净功率 (kW)"),
        margin=dict(l=40, r=40, t=40, b=40),
        template="plotly_white",
        height=350,
    )

    # 指标
    solar_total_txt = f"{result['total_solar']:.1f}"
    load_total_txt = f"{result['total_load']:.1f}"
    green_ratio_txt = f"{result['green_ratio']:.1f}%"
    solar_peak_txt = f"{result['solar_peak']:.1f} kW @ {result['solar_peak_time'].strftime('%H:%M')}"
    load_peak_txt = f"{result['load_peak']:.1f} kW"

    # 模型状态
    status = result["model_status"]
    if status["solar_ok"] and status["charging_ok"]:
        status_md = "✅ 光伏 LSTM | ✅ 充电 TCN-Attention-LSTM"
    elif status["solar_ok"]:
        status_md = "✅ 光伏 LSTM | ⚠️ 充电模拟"
    elif status["charging_ok"]:
        status_md = "⚠️ 光伏模拟 | ✅ 充电 TCN-Attention-LSTM"
    else:
        status_md = "⚠️ 模型未加载，使用日模式模拟"

    return fig1, fig2, solar_total_txt, load_total_txt, green_ratio_txt, solar_peak_txt, load_peak_txt, status_md

# ============================================================
# Tab 3: 气象监测
# ============================================================
def build_weather_tab():
    """构建气象 Tab 的 UI 组件"""
    btn_refresh = gr.Button("🔄 刷新气象数据", variant="secondary")
    weather_rad_chart = gr.Plot(label="辐照度趋势")
    weather_forecast_table = gr.Dataframe(label="未来3小时气象预报", headers=["时间", "辐照度", "云量", "降雨量", "天气"])
    weather_warning = gr.HTML("")
    weather_summary = gr.HTML("")

    btn_refresh.click(
        fn=_refresh_weather,
        inputs=[],
        outputs=[weather_summary, weather_rad_chart, weather_forecast_table, weather_warning],
    )
    return btn_refresh

def _refresh_weather():
    """刷新气象数据"""
    data = get_weather()
    summary = get_current_weather_summary(data)
    chart = build_radiation_chart(data)
    
    if data and data.get("forecast_df") is not None:
        forecast_df = data["forecast_df"].copy()
        table = forecast_df[["时间", "辐照度 (W/m²)", "云量 (%)", "降雨量 (mm)", "天气"]]
    else:
        table = _empty_table()

    warning = data.get("warning_msg", "") if data else "⚠️ 气象数据获取失败"
    return summary, chart, table, warning

def _empty_table():
    import pandas as pd
    return pd.DataFrame(columns=["时间", "辐照度 (W/m²)", "云量 (%)", "降雨量 (mm)", "天气"])

# ============================================================
# Tab 4: 数据探索（精简版，移除上传组件）
# ============================================================
def build_data_tab():
    """构建数据探索 Tab 的 UI 组件（v2: 移除上传功能）"""
    # ---- 数据集概览 ----
    gr.Markdown("### 📋 数据集概览")
    gr.Markdown("*提示: 如需上传自定义数据，请切换到「📤 数据上传」标签页*")
    overview_html = gr.HTML(value=get_dataset_overview())

    # ---- 历史日负荷曲线 ----
    gr.Markdown("### 📉 历史日负荷曲线")
    available_dates = get_available_dates()
    default_dates = available_dates[:2] if len(available_dates) >= 2 else available_dates
    date_selector = gr.Dropdown(
        choices=available_dates,
        value=default_dates,
        multiselect=True,
        label="选择日期（最多3个）",
    )
    daily_chart = gr.Plot(label="日负荷曲线")

    # ---- 特征相关性分析 ----
    gr.Markdown("### 📊 特征相关性分析")
    corr_chart = gr.Plot(label="Pearson 相关系数", value=get_correlation_chart())

    # ---- 小时级负荷画像 ----
    gr.Markdown("### ⏰ 小时级负荷画像")
    gr.Markdown("""
    > **说明**: 小时级负荷画像展示 24 小时中每小时的平均负荷水平 ± 1 个标准差。  
    > 横轴为一天中的 24 个小时（0点 ~ 23点），纵轴为该时段内的平均负荷值。  
    > 帮助识别一天内负荷的高峰与低谷时段。
    """)
    hourly_chart = gr.Plot(label="均值 ± 1σ", value=get_hourly_profile_chart())

    # 日期选择事件
    date_selector.change(
        fn=_update_daily_chart,
        inputs=[date_selector],
        outputs=[daily_chart],
    )

    return date_selector, daily_chart, corr_chart, hourly_chart

def _update_daily_chart(date_strs):
    if len(date_strs) > 3:
        date_strs = date_strs[:3]
    return plot_daily_load_curves(date_strs)

# ============================================================
# Tab 5: 策略建议
# ============================================================
def build_strategy_tab():
    """构建策略建议 Tab 的 UI 组件"""
    btn_gen = gr.Button("📋 生成策略建议", variant="primary")
    strategy_output = gr.Markdown(
        value="> 请先在「预测核心」Tab 执行预测，然后点击上方按钮生成策略建议"
    )

    btn_gen.click(
        fn=_generate_strategy_ui,
        inputs=[],
        outputs=[strategy_output],
    )
    return btn_gen

def _generate_strategy_ui():
    result = get_last_prediction()
    if result is None:
        return "### ⚠️ 尚未执行预测\n\n请先在 **预测核心** Tab 中点击「执行预测」，获取预测结果后再生成策略建议。"
    return generate_strategy(result)

# ============================================================
# Tab 6: 误差分析
# ============================================================
def build_error_tab():
    """构建误差分析 Tab 的 UI 组件"""
    btn_run_eval = gr.Button("🔍 运行评估", variant="primary")

    # ---- 充电模型 ----
    gr.Markdown("## ⚡ 充电模型 (TCN-Attention-LSTM)")

    gr.Markdown("### 📊 充电模型回测")
    charging_backtest_chart = gr.Plot(label="充电 — 真实值 vs 预测值")
    charging_backtest_summary = gr.Markdown("")

    gr.Markdown("### 📈 充电残差分布分析")
    with gr.Row():
        charging_error_dist_chart = gr.Plot(label="充电残差分布直方图")
        charging_error_hour_chart = gr.Plot(label="充电按小时误差")

    # ---- 光伏模型 ----
    gr.Markdown("## ☀️ 光伏模型 (LSTM + GAN)")

    gr.Markdown("### 📊 光伏模型回测")
    solar_backtest_chart = gr.Plot(label="光伏 — 真实值 vs 预测值")
    solar_backtest_summary = gr.Markdown("")

    gr.Markdown("### 📈 光伏残差分布分析")
    with gr.Row():
        solar_error_dist_chart = gr.Plot(label="光伏残差分布直方图")
        solar_error_hour_chart = gr.Plot(label="光伏按小时误差")

    gr.Markdown("### 📋 光伏模型信息")
    solar_model_info = gr.Markdown("")

    btn_run_eval.click(
        fn=_run_evaluation,
        inputs=[],
        outputs=[
            charging_backtest_chart, charging_backtest_summary,
            charging_error_dist_chart, charging_error_hour_chart,
            solar_backtest_chart, solar_backtest_summary,
            solar_error_dist_chart, solar_error_hour_chart,
            solar_model_info,
        ],
    )
    return btn_run_eval

def _run_evaluation():
    """执行误差分析（充电 + 光伏）"""
    # --- 充电模型 ---
    charging_backtest_fig, charging_backtest_msg = run_backtest_charging()
    charging_error_dist_fig = build_error_distribution_chart()
    charging_error_hour_fig = build_error_by_hour_chart()

    # --- 光伏模型 ---
    solar_backtest_fig, solar_backtest_msg = run_backtest_solar()
    solar_error_dist_fig = build_solar_error_distribution_chart()
    solar_error_hour_fig = build_solar_error_by_hour_chart()
    solar_info = get_solar_model_info()

    return (
        charging_backtest_fig, charging_backtest_msg,
        charging_error_dist_fig, charging_error_hour_fig,
        solar_backtest_fig, solar_backtest_msg,
        solar_error_dist_fig, solar_error_hour_fig,
        solar_info,
    )

# ============================================================
# 主应用
# ============================================================
def create_app():
    """构建 Gradio Blocks 应用 (v2: 新 Tab 顺序)"""
    with gr.Blocks(
        title=APP_TITLE,
        theme=APP_THEME,
        css=CUSTOM_CSS,
    ) as app:
        # 标题 + 主题切换按钮
        gr.HTML(
            f"""
            <div style="text-align:center; margin-bottom:20px;">
                <h1 style="font-size:28px; margin-bottom:4px;">
                    ☀️🔋⚡ {APP_TITLE}
                </h1>
                <p style="font-size:14px; opacity:0.7;">
                    上传数据 → 光伏预测 (LSTM + GAN) | 充电负荷预测 (TCN-Attention-LSTM) | 策略建议
                </p>
            </div>
            """
        )
        
        # Dark/Light 主题切换按钮
        with gr.Row(elem_classes="theme-row"):
            theme_btn = gr.Button("🌙 深色 / ☀️ 浅色", variant="secondary", size="sm")
        
        theme_btn.click(
            fn=None,
            js="""
            function() {
                document.body.classList.toggle('dark');
                return [];
            }
            """,
        )

        # Tab 页 (v2 新顺序)
        with gr.Tabs():
            # Tab 1: 数据上传（新增）
            with gr.TabItem("📤 数据上传", id="tab_upload"):
                build_upload_tab()

            # Tab 2: 预测核心
            with gr.TabItem("🎯 预测核心", id="tab_predict"):
                build_prediction_tab()

            # Tab 3: 气象监测
            with gr.TabItem("🌤️ 气象监测", id="tab_weather"):
                build_weather_tab()

            # Tab 4: 数据探索（精简版）
            with gr.TabItem("📊 数据探索", id="tab_data"):
                date_selector, daily_chart, corr_chart, hourly_chart = build_data_tab()

            # Tab 5: 策略建议
            with gr.TabItem("💡 策略建议", id="tab_strategy"):
                build_strategy_tab()

            # Tab 6: 误差分析
            with gr.TabItem("🔬 误差分析", id="tab_error"):
                build_error_tab()

    return app

# ============================================================
# 启动
# ============================================================
if __name__ == "__main__":
    print("=" * 50)
    print(f"  {APP_TITLE}")
    print("=" * 50)
    print("[启动] 加载气象数据...")
    get_weather()

    # 启动 Gradio
    app = create_app()
    app.queue(default_concurrency_limit=5)
    app.launch(
        server_port=APP_PORT,
        share=False,
        show_error=True,
    )