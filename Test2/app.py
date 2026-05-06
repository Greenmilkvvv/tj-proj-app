"""
光储充智能预测 Demo — Gradio 主应用
基于真实气象 API + 历史训练数据 + 预测模型权重
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
)
from prediction_service import run_prediction, generate_strategy

# ============================================================
# 全局缓存
# ============================================================
_weather_cache = None
_weather_cache_time = 0
WEATHER_CACHE_TTL = 300  # 5 分钟

# ============================================================
# CSS 样式
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
# Tab 1: 预测核心
# ============================================================
def build_prediction_tab():
    """构建预测 Tab 的 UI 组件"""
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ 预测参数")
            forecast_option = gr.Dropdown(
                choices=list(PREDICTION_OPTIONS.keys()),
                value="6 小时",
                label="预测时长",
            )
            btn_run = gr.Button("🚀 执行预测", variant="primary", size="lg")
            
            # 预测指标
            gr.Markdown("---")
            gr.Markdown("### 📈 预测指标")
            with gr.Row():
                solar_total = gr.Textbox(label="☀️ 光伏总出力 (kWh)", value="—")
                load_total = gr.Textbox(label="⚡ 充电总需求 (kWh)", value="—")
            with gr.Row():
                green_ratio = gr.Textbox(label="🌿 绿电替代率", value="—")
                model_status_md = gr.Markdown("")
            
            gr.Markdown("---")
            gr.Markdown("### 📊 峰谷指标")
            solar_peak_info = gr.Textbox(label="光伏峰值", value="—")
            load_peak_info = gr.Textbox(label="负荷峰值", value="—")

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
        status_md = "✅ 光伏 CNN-LSTM | ✅ 充电 TCN-Attention-LSTM"
    elif status["solar_ok"]:
        status_md = "✅ 光伏 CNN-LSTM | ⚠️ 充电模拟"
    elif status["charging_ok"]:
        status_md = "⚠️ 光伏模拟 | ✅ 充电 TCN-Attention-LSTM"
    else:
        status_md = "⚠️ 模型未加载，使用日模式模拟"

    return fig1, fig2, solar_total_txt, load_total_txt, green_ratio_txt, solar_peak_txt, load_peak_txt, status_md


# ============================================================
# Tab 2: 气象监测
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
        # 只取时间、辐照度、云量、降雨量、天气5列
        table = forecast_df[["时间", "辐照度 (W/m²)", "云量 (%)", "降雨量 (mm)", "天气"]]
    else:
        table = _empty_table()

    warning = data.get("warning_msg", "") if data else "⚠️ 气象数据获取失败"
    return summary, chart, table, warning


def _empty_table():
    import pandas as pd
    return pd.DataFrame(columns=["时间", "辐照度 (W/m²)", "云量 (%)", "降雨量 (mm)", "天气"])


# ============================================================
# Tab 3: 数据探索
# ============================================================
def build_data_tab():
    """构建数据探索 Tab 的 UI 组件"""
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
    hourly_chart = gr.Plot(label="均值 ± 1σ", value=get_hourly_profile_chart())

    # 事件绑定
    date_selector.change(
        fn=_update_daily_chart,
        inputs=[date_selector],
        outputs=[daily_chart],
    )

    # 初始加载
    return date_selector, daily_chart, corr_chart, hourly_chart


def _update_daily_chart(date_strs):
    if len(date_strs) > 3:
        date_strs = date_strs[:3]
    return plot_daily_load_curves(date_strs)


# ============================================================
# Tab 4: 策略建议
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
# 主应用
# ============================================================
def create_app():
    """构建 Gradio Blocks 应用"""
    with gr.Blocks(
        title=APP_TITLE,
        theme=APP_THEME,
        css=CUSTOM_CSS,
    ) as app:
        # 标题
        gr.HTML(
            f"""
            <div style="text-align:center; margin-bottom:20px;">
                <h1 style="font-size:28px; margin-bottom:4px;">
                    ☀️🔋⚡ {APP_TITLE}
                </h1>
                <p style="font-size:14px; opacity:0.7;">
                    光伏预测 (CNN-LSTM) | 充电负荷预测 (TCN-Attention-LSTM) | 历史数据探索
                </p>
            </div>
            """
        )

        # Tab 页
        with gr.Tabs():
            with gr.TabItem("🎯 预测核心", id="tab_predict"):
                build_prediction_tab()

            with gr.TabItem("🌤️ 气象监测", id="tab_weather"):
                build_weather_tab()

            with gr.TabItem("📊 数据探索", id="tab_data"):
                date_selector, daily_chart, corr_chart, hourly_chart = build_data_tab()

            with gr.TabItem("💡 策略建议", id="tab_strategy"):
                build_strategy_tab()

    return app


# ============================================================
# 启动
# ============================================================
if __name__ == "__main__":
    # 首次加载数据
    print("=" * 50)
    print(f"  {APP_TITLE}")
    print("=" * 50)
    print("[启动] 加载气象数据...")
    get_weather()

    # 启动 Gradio
    app = create_app()
    app.queue(default_concurrency_limit=5)
    app.launch(
        # server_name="0.0.0.0",
        server_port=APP_PORT,
        share=False,
        show_error=True,
    )