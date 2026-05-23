"""
主应用入口 (v3)
===========================
变更:
- 移除「数据上传」Tab，使用内置数据集
- 充电负荷预测改为 Input-Driven 三模式选择
- 修复页面宽度漂移问题
- 保留所有展示逻辑
"""
import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False

# 自动添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    APP_TITLE, APP_PORT, APP_THEME, CUSTOM_CSS,
    PREDICTION_OPTIONS, DEFAULT_PREDICTION_STEPS,
    CHARGING_MODEL_PTH, SOLAR_MODEL_PTH,
)

from data_service import (
    get_recent_charging_data, get_recent_solar_data,
    create_prediction_chart, create_summary_chart,
    run_backtest_charging, run_backtest_solar,
    build_charging_error_distribution_chart,
    build_charging_error_by_hour_chart,
    build_solar_error_distribution_chart,
    build_solar_error_by_hour_chart,
    get_charging_model_info, get_solar_model_info,
    get_dataset_overview, get_available_dates,
    plot_daily_load_curves, get_correlation_chart,
    get_hourly_profile_chart, _error_fallback,
)

from prediction_service import (
    run_prediction, generate_strategy,
    predict_solar, predict_charging,
    save_prediction_result, get_last_prediction,
    get_combined_prediction,
)

from weather_service import (
    get_weather, get_current_weather_summary,
    build_radiation_chart,
)

# ============================================================
# 固定宽度 CSS (解决页面宽度漂移)
# ============================================================
FIXED_WIDTH_CSS = """
/* 固定整体宽度，防止 Tab 切换时宽度漂移 */
.gradio-container {
    max-width: 1200px !important;
    min-width: 1100px !important;
    margin-left: auto !important;
    margin-right: auto !important;
}
/* 确保所有 Tab 内容区域保持相同宽度 */
.tabs > .tabitem {
    width: 100% !important;
}
/* 防止表格撑开 */
table {
    table-layout: fixed !important;
    word-break: break-all !important;
}
.dark {
    /* 深色模式保持 */
}
"""

# ============================================================
# Tab 1: 预测核心 (重构版 - 充电负荷改为三模式输入驱动)
# ============================================================
# 全局变量: 存储最后一次充电预测的输入参数
_last_charging_input: dict = {}
_last_charging_mode: str = ""
_last_charging_result: dict = {}

def build_prediction_tab():
    """构建预测核心 Tab (v4: 统一交互 - 一步选择 + 一键执行联合预测)"""
    gr.Markdown(
        """
        ## 🎯 联合预测引擎
        
        统一的预测步数选择和充电参数输入，一键同时执行光伏和充电负荷预测。
        """
    )

    # ---- 充电参数输入 ----
    gr.Markdown("### 🔋 充电参数")
    with gr.Row():
        price_input = gr.Number(
            label="当前充电电价 (元/kWh)",
            value=0.8,
            precision=3,
            minimum=0,
            step=0.1,
        )
        load_input = gr.Number(
            label="当前充电负荷 (kW)",
            value=20.0,
            precision=1,
            minimum=0,
            step=1.0,
        )

    # ---- 统一预测控制 ----
    gr.Markdown("### 🎯 预测控制")
    with gr.Row():
        steps_dropdown = gr.Dropdown(
            choices=["15分钟 (1步)", "1小时 (4步)", "24小时 (96步)"],
            value="1小时 (4步)",
            label="预测步数",
            scale=3,
        )
        btn_predict = gr.Button("🚀 执行联合预测", variant="primary", scale=2)

    # ---- 状态指示 ----
    status_html_base = gr.HTML("<span style='color:#888;'>⏳ 等待执行预测</span>")
    status_html = gr.HTML("")

    # ---- 联合预测可视化 ----
    gr.Markdown("---")
    gr.Markdown("### 📊 光伏 + 充电 联合预测")

    with gr.Row():
        joint_plot = gr.Plot(label="预测对比曲线", value=None)
        summary_plot = gr.Plot(label="关键指标摘要", value=None)

    metrics_html = gr.HTML("")

    # ---- 事件绑定 (一键同时执行光伏 + 充电预测) ----
    btn_predict.click(
        fn=_do_joint_prediction,
        inputs=[steps_dropdown, price_input, load_input],
        outputs=[status_html],
    ).then(
        fn=_refresh_joint_view,
        inputs=[steps_dropdown],
        outputs=[joint_plot, summary_plot, metrics_html],
    )

    return (steps_dropdown, btn_predict,
            price_input, load_input,
            joint_plot, summary_plot, status_html, metrics_html)


def _do_joint_prediction(steps_label: str, price: float, load_kw: float):
    """统一执行光伏 + 充电预测"""
    global _last_charging_input, _last_charging_result

    step_map = {"15分钟 (1步)": 1, "1小时 (4步)": 4, "24小时 (96步)": 96}
    n_steps = step_map.get(steps_label, 4)

    _last_charging_input = {"price": price, "load_kw": load_kw}

    status_parts = []

    # 执行光伏预测
    try:
        solar_result = predict_solar(n_steps=n_steps)
        if solar_result is None:
            status_parts.append("☀️ 光伏: ❌ 模型未加载")
        else:
            save_prediction_result("solar", solar_result)
            status_parts.append("☀️ 光伏: ✅ 完成")
    except Exception as e:
        status_parts.append(f"☀️ 光伏: ❌ {e}")

    # 执行充电预测
    try:
        charging_result = predict_charging(price=price, load=load_kw, n_steps=n_steps)
        if charging_result is None:
            status_parts.append("🔋 充电: ❌ 模型未加载")
        else:
            _last_charging_result = charging_result
            save_prediction_result("charging", charging_result)
            status_parts.append("🔋 充电: ✅ 完成")
    except Exception as e:
        status_parts.append(f"🔋 充电: ❌ {e}")

    return f"<span>{'&nbsp;' * 8} </span>".join(status_parts)


def _refresh_joint_view(steps_label: str):
    """刷新联合预测视图"""
    solar_result = get_last_prediction("solar")
    charging_result = get_last_prediction("charging")

    joint_fig = None
    summary_fig = None
    metrics_html = "<p style='color:#888;'>请先执行光伏和充电预测</p>"

    if solar_result is not None and charging_result is not None:
        # 合并预测
        step_map = {"15分钟 (1步)": 1, "1小时 (4步)": 4, "24小时 (96步)": 96}
        n_steps = step_map.get(steps_label, 4)
        combined = get_combined_prediction(solar_result, charging_result, n_steps=n_steps)
        if combined:
            joint_fig = create_prediction_chart(combined)
            summary_fig = create_summary_chart(combined)
            metrics_html = _build_metrics_html(combined)
    elif solar_result is not None:
        combined = solar_result.copy()
        combined["load_mean"] = [0] * len(combined.get("solar", []))
        combined["load_lower"] = [0] * len(combined.get("solar", []))
        combined["load_upper"] = [0] * len(combined.get("solar", []))
        # 计算净负荷
        combined["net"] = combined["solar"]
        n = len(combined["solar"])
        combined["total_solar"] = sum(combined["solar"]) * 0.25
        combined["total_load"] = 0
        combined["solar_peak"] = max(combined["solar"]) if len(combined["solar"]) > 0 else 0
        combined["solar_peak_time"] = None
        combined["load_peak"] = 0
        combined["green_ratio"] = 100.0
        joint_fig = create_prediction_chart(combined)
        summary_fig = create_summary_chart(combined)
        metrics_html = "<p style='color:#888;'>⚠️ 尚未执行充电预测，仅展示光伏数据</p>"
    elif charging_result is not None:
        combined = charging_result.copy()
        combined["solar"] = [0] * len(combined.get("load_mean", []))
        combined["net"] = [-v for v in combined.get("load_mean", [])]
        n = len(combined["load_mean"])
        combined["total_solar"] = 0
        combined["total_load"] = sum(combined["load_mean"]) * 0.25
        combined["solar_peak"] = 0
        combined["solar_peak_time"] = None
        combined["load_peak"] = max(combined["load_mean"]) if len(combined["load_mean"]) > 0 else 0
        combined["green_ratio"] = 0
        joint_fig = create_prediction_chart(combined)
        summary_fig = create_summary_chart(combined)
        metrics_html = "<p style='color:#888;'>⚠️ 尚未执行光伏预测，仅展示充电数据</p>"

    return joint_fig, summary_fig, metrics_html


def _build_solar_only_chart(result):
    """构建仅光伏的预测图"""
    if not PLOTLY_AVAILABLE:
        return None
    times = result.get("times", [])
    solar = result.get("solar", [])
    times_str = [t.strftime("%m-%d %H:%M") for t in times]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times_str, y=solar,
        mode="lines+markers",
        name="光伏出力 (kW)",
        line=dict(color="#ff9800", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(255,152,0,0.15)",
    ))
    fig.update_layout(
        title="光伏出力预测",
        xaxis=dict(title="时间", tickangle=45),
        yaxis=dict(title="kW"),
        height=350,
        template="plotly_white",
        margin=dict(l=40, r=40, t=40, b=40),
    )
    return fig


def _build_charging_only_chart(result):
    """构建仅充电的预测图"""
    if not PLOTLY_AVAILABLE:
        return None
    times = result.get("times", [])
    load_mean = result.get("load_mean", [])
    load_lower = result.get("load_lower", [])
    load_upper = result.get("load_upper", [])
    times_str = [t.strftime("%m-%d %H:%M") for t in times]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times_str, y=load_mean,
        mode="lines+markers",
        name="充电负荷 (kW)",
        line=dict(color="#2196f3", width=2.5),
    ))
    # 置信区间
    x_fill = times_str + times_str[::-1]
    y_fill = load_upper + load_lower[::-1]
    fig.add_trace(go.Scatter(
        x=x_fill, y=y_fill,
        fill="toself",
        fillcolor="rgba(33,150,243,0.15)",
        line=dict(width=0),
        name="负荷 85%-115% 区间",
    ))
    fig.update_layout(
        title="充电负荷预测",
        xaxis=dict(title="时间", tickangle=45),
        yaxis=dict(title="kW"),
        height=350,
        template="plotly_white",
        margin=dict(l=40, r=40, t=40, b=40),
    )
    return fig


def _build_metrics_html(result):
    """构建关键指标 HTML"""
    total_solar = result.get("total_solar", 0)
    total_load = result.get("total_load", 0)
    green_ratio = result.get("green_ratio", 0)
    solar_peak = result.get("solar_peak", 0)
    load_peak = result.get("load_peak", 0)
    return f"""
    <div style="display:flex; flex-wrap:wrap; gap:16px;">
        <div style="flex:1; min-width:140px; background:#fff3e0; border-radius:8px; padding:12px; text-align:center;">
            <div style="font-size:12px; color:#888;">总光伏发电</div>
            <div style="font-size:22px; font-weight:bold; color:#ff9800;">{total_solar:.1f} kWh</div>
        </div>
        <div style="flex:1; min-width:140px; background:#e3f2fd; border-radius:8px; padding:12px; text-align:center;">
            <div style="font-size:12px; color:#888;">总充电负荷</div>
            <div style="font-size:22px; font-weight:bold; color:#2196f3;">{total_load:.1f} kWh</div>
        </div>
        <div style="flex:1; min-width:140px; background:#e8f5e9; border-radius:8px; padding:12px; text-align:center;">
            <div style="font-size:12px; color:#888;">绿电替代率</div>
            <div style="font-size:22px; font-weight:bold; color:#4caf50;">{green_ratio:.1f}%</div>
        </div>
        <div style="flex:1; min-width:140px; background:#fce4ec; border-radius:8px; padding:12px; text-align:center;">
            <div style="font-size:12px; color:#888;">光伏峰值 / 负荷峰值</div>
            <div style="font-size:22px; font-weight:bold; color:#e91e63;">{solar_peak:.1f} / {load_peak:.1f} kW</div>
        </div>
    </div>
    """


def _status_html(model_type: str, ok: bool):
    """返回模型状态 HTML"""
    if model_type == "solar":
        icon = "✅" if ok else "❌"
        return f"<span>{icon} 光伏 LSTM {'就绪' if ok else '失败'}</span>"
    else:
        icon = "✅" if ok else "❌"
        return f"<span>{icon} 充电 TCN-Attention-LSTM {'就绪' if ok else '失败'}</span>"


def _empty_chart(msg: str):
    """返回空图表 (带消息)"""
    if not PLOTLY_AVAILABLE:
        return None
    fig = go.Figure()
    fig.add_annotation(x=0.5, y=0.5, text=msg, showarrow=False, font=dict(size=14, color="#888"))
    fig.update_layout(height=300, template="plotly_white")
    return fig


# ============================================================
# Tab 2: 气象监测 (与 Test4 一致)
# ============================================================
def build_weather_tab():
    """构建气象 Tab 的 UI 组件"""
    btn_refresh = gr.Button("🔄 刷新气象数据", variant="secondary")
    weather_rad_chart = gr.Plot(label="辐照度趋势")
    weather_forecast_table = gr.Dataframe(
        label="未来3小时气象预报",
        headers=["时间", "辐照度", "云量", "降雨量", "天气"]
    )
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
    return pd.DataFrame(columns=["时间", "辐照度 (W/m²)", "云量 (%)", "降雨量 (mm)", "天气"])


# ============================================================
# Tab 3: 数据探索 (基于内置数据，移除上传引用)
# ============================================================
def build_data_tab():
    """构建数据探索 Tab (v3: 移除上传功能)"""
    gr.Markdown("### 📋 数据集概览")
    overview_html = gr.HTML(value=get_dataset_overview())

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

    gr.Markdown("### 📊 特征相关性分析")
    corr_chart = gr.Plot(label="Pearson 相关系数", value=get_correlation_chart())

    gr.Markdown("### ⏰ 小时级负荷画像")
    gr.Markdown("""
    > **说明**: 小时级负荷画像展示 24 小时中每小时的平均负荷水平 ± 1 个标准差。  
    > 横轴为一天中的 24 个小时（0点 ~ 23点），纵轴为该时段内的平均负荷值。  
    > 帮助识别一天内负荷的高峰与低谷时段。
    """)
    hourly_chart = gr.Plot(label="均值 ± 1σ", value=get_hourly_profile_chart())

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
# Tab 4: 策略建议 (与 Test4 一致)
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
    result = get_last_prediction("combined") or get_last_prediction("solar")
    if result is None:
        return "### ⚠️ 尚未执行预测\n\n请先在 **预测核心** Tab 中执行预测，获取预测结果后再生成策略建议。"
    return generate_strategy(result)


# ============================================================
# Tab 5: 误差分析 (参照 Test4 设计)
# ============================================================
def build_error_tab():
    """构建误差分析 Tab (v3: 参照 Test4 设计)"""
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
    charging_error_dist_fig = build_charging_error_distribution_chart()
    charging_error_hour_fig = build_charging_error_by_hour_chart()

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
    """构建 Gradio Blocks 应用 (v3)"""
    # 合并 CSS
    combined_css = FIXED_WIDTH_CSS + "\n" + (CUSTOM_CSS or "")

    with gr.Blocks(
        title=APP_TITLE,
        theme=APP_THEME,
        css=combined_css,
    ) as app:
        # 标题
        gr.HTML(
            f"""
            <div style="text-align:center; margin-bottom:20px;">
                <h1 style="font-size:28px; margin-bottom:4px;">
                    ☀️🔋⚡ {APP_TITLE}
                </h1>
                <p style="font-size:14px; opacity:0.7;">
                    实时同步数据 → 光伏预测 (LSTM + GAN) | 充电负荷三模式预测 (TCN-Attention-LSTM) | 策略建议
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

        # Tab 页 (v3 新顺序，移除上传 Tab)
        with gr.Tabs(elem_classes="main-tabs"):
            # Tab 1: 预测核心
            with gr.TabItem("🎯 预测核心", id="tab_predict"):
                build_prediction_tab()

            # Tab 2: 气象监测
            with gr.TabItem("🌤️ 气象监测", id="tab_weather"):
                build_weather_tab()

            # Tab 3: 数据探索
            with gr.TabItem("📊 数据探索", id="tab_data"):
                build_data_tab()

            # Tab 4: 策略建议
            with gr.TabItem("💡 策略建议", id="tab_strategy"):
                build_strategy_tab()

            # Tab 5: 误差分析
            with gr.TabItem("🔬 误差分析", id="tab_error"):
                build_error_tab()

    return app


# ============================================================
# 启动
# ============================================================
if __name__ == "__main__":
    print("=" * 50)
    print(f"  {APP_TITLE} (v3)")
    print("=" * 50)
    print("[启动] 加载气象数据...")
    get_weather()

    app = create_app()
    app.queue(default_concurrency_limit=5)
    app.launch(
        server_port=APP_PORT,
        share=False,
        show_error=True,
    )