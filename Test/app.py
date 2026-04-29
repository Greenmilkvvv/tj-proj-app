"""
光储充一体化设备智能管理平台 - Gradio 测试版 Demo
基于 Demo_Requirement/demo_ui.md 需求设计
"""
import os
import sys
import gradio as gr
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import threading
import time
import queue
import random
import warnings
warnings.filterwarnings("ignore")

# 导入自定义模块
from model_service import solar_forecast_service, charging_forecast_service
from weather_service import update_weather_ui, get_weather_data

# ============================================================
# 全局状态
# ============================================================
SIMULATION_RUNNING = False
SIMULATION_THREAD = None
SIMULATION_DATA_QUEUE = queue.Queue(maxsize=200)
SIMULATION_HISTORY = {
    "time": [],
    "solar_power": [],
    "charging_power": [],
    "grid_power": [],
    "soc": [],
    "battery_power": [],
}

# 策略参数
STRATEGY_PARAMS = {
    "soc_min": 20.0,
    "soc_max": 90.0,
    "charge_rate": 60.0,  # kW
    "discharge_rate": 60.0,  # kW
    "grid_limit": 100.0,  # kW
    "feedback_enabled": True,
}


# ============================================================
# CSS 样式
# ============================================================
CUSTOM_CSS = """
.gradio-container { max-width: 1400px !important; margin: auto; }
.status-card { 
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 12px; padding: 20px; color: white; text-align: center;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
}
.status-card.green { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }
.status-card.orange { background: linear-gradient(135deg, #f2994a 0%, #f2c94c 100%); }
.status-card.red { background: linear-gradient(135deg, #cb2d3e 0%, #ef473a 100%); }
.value-display { font-size: 36px; font-weight: bold; margin: 10px 0; }
.label-display { font-size: 14px; opacity: 0.85; }
.warning-box {
    background: #fff3cd; border: 1px solid #ffc107; border-radius: 8px;
    padding: 12px; margin: 10px 0; color: #856404;
}
.title-section { text-align: center; margin-bottom: 30px; }
.title-section h1 { font-size: 32px; color: #2c3e50; margin-bottom: 8px; }
.title-section p { font-size: 16px; color: #7f8c8d; }
.tab-header { font-size: 18px; font-weight: bold; margin-bottom: 15px; color: #2c3e50; }
"""


# ============================================================
# Tab 1: 预测核心 - 光伏预测
# ============================================================
def solar_forecast_tab():
    def predict_solar(forecast_hours, weather_data=None):
        """执行光伏预测"""
        if weather_data is None:
            weather_data = get_weather_data()
        
        result = solar_forecast_service.predict(
            forecast_hours=int(forecast_hours),
            weather_data=weather_data
        )
        
        if result is None:
            return None, "⚠️ 模型加载失败，使用模拟数据", pd.DataFrame()
        
        # 构建图表
        times = result["times"]
        predictions = result["predictions"]
        confidence_lower = result.get("confidence_lower", [p * 0.85 for p in predictions])
        confidence_upper = result.get("confidence_upper", [p * 1.15 for p in predictions])
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=times, y=predictions, mode='lines+markers', name='光伏预测功率',
            line=dict(color='#FF9800', width=3),
            marker=dict(size=6)
        ))
        fig.add_trace(go.Scatter(
            x=times, y=confidence_lower, mode='lines', name='置信下限',
            line=dict(color='rgba(255,152,0,0.3)', width=1),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=times, y=confidence_upper, mode='lines', name='置信上限',
            line=dict(color='rgba(255,152,0,0.3)', width=1),
            fill='tonexty', fillcolor='rgba(255,152,0,0.1)',
            showlegend=False
        ))
        fig.update_layout(
            title=f"光伏发电功率预测 ({forecast_hours}h)",
            xaxis_title="时间", yaxis_title="功率 (kW)",
            height=400, hovermode="x unified",
        )
        
        # 统计信息
        total_energy = sum(predictions) * (forecast_hours / len(predictions))
        peak_power = max(predictions)
        avg_power = np.mean(predictions) if predictions else 0
        
        stats_html = f"""
        <div style='background:#f5f5f5; border-radius:8px; padding:15px;'>
        <h3 style='margin-top:0;'>📊 预测统计</h3>
        <table style='width:100%;'>
        <tr><td>🔆 峰值功率:</td><td><b>{peak_power:.1f} kW</b></td></tr>
        <tr><td>📈 平均功率:</td><td><b>{avg_power:.1f} kW</b></td></tr>
        <tr><td>⚡ 预计发电量:</td><td><b>{total_energy:.1f} kWh</b></td></tr>
        <tr><td>⏱️ 预测时长:</td><td><b>{forecast_hours}h</b></td></tr>
        </table>
        </div>
        """
        
        return fig, stats_html, create_solar_table(times, predictions)
    
    return predict_solar


def create_solar_table(times, predictions):
    """创建光伏预测数据表格"""
    df = pd.DataFrame({
        "时间": times,
        "预测功率 (kW)": [f"{p:.1f}" for p in predictions],
    })
    return df


# ============================================================
# Tab 1: 预测核心 - 充电负荷预测
# ============================================================
def charging_forecast_tab():
    def predict_charging(forecast_hours):
        """执行充电负荷预测"""
        result = charging_forecast_service.predict(forecast_hours=int(forecast_hours))
        
        if result is None:
            return None, "⚠️ 模型加载失败，使用模拟数据", pd.DataFrame()
        
        times = result["times"]
        predictions = result["predictions"]
        confidence_lower = result.get("confidence_lower", [p * 0.85 for p in predictions])
        confidence_upper = result.get("confidence_upper", [p * 1.15 for p in predictions])
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=times, y=predictions, mode='lines+markers', name='充电负荷预测',
            line=dict(color='#2196F3', width=3),
            marker=dict(size=6)
        ))
        fig.add_trace(go.Scatter(
            x=times, y=confidence_lower, mode='lines', name='置信下限',
            line=dict(color='rgba(33,150,243,0.3)', width=1),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=times, y=confidence_upper, mode='lines', name='置信上限',
            line=dict(color='rgba(33,150,243,0.3)', width=1),
            fill='tonexty', fillcolor='rgba(33,150,243,0.1)',
            showlegend=False
        ))
        fig.update_layout(
            title=f"充电负荷预测 ({forecast_hours}h)",
            xaxis_title="时间", yaxis_title="功率 (kW)",
            height=400, hovermode="x unified",
        )
        
        peak_load = max(predictions)
        avg_load = np.mean(predictions) if predictions else 0
        total_demand = sum(predictions) * (forecast_hours / len(predictions))
        
        stats_html = f"""
        <div style='background:#f5f5f5; border-radius:8px; padding:15px;'>
        <h3 style='margin-top:0;'>📊 预测统计</h3>
        <table style='width:100%;'>
        <tr><td>🔌 峰值负荷:</td><td><b>{peak_load:.1f} kW</b></td></tr>
        <tr><td>📈 平均负荷:</td><td><b>{avg_load:.1f} kW</b></td></tr>
        <tr><td>⚡ 预计充电量:</td><td><b>{total_demand:.1f} kWh</b></td></tr>
        <tr><td>⏱️ 预测时长:</td><td><b>{forecast_hours}h</b></td></tr>
        </table>
        </div>
        """
        
        return fig, stats_html, create_charging_table(times, predictions)
    
    return predict_charging


def create_charging_table(times, predictions):
    """创建充电负荷预测数据表格"""
    df = pd.DataFrame({
        "时间": times,
        "预测负荷 (kW)": [f"{p:.1f}" for p in predictions],
    })
    return df


# ============================================================
# Tab 2: 实时监控 - 实时数据获取
# ============================================================
def get_realtime_data():
    """获取实时监控数据"""
    now = datetime.now()
    
    # 模拟实时数据（实际应接入传感器/IoT设备）
    hour = now.hour + now.minute / 60
    if 6 <= hour <= 19:
        solar_base = 800 * np.sin(np.pi * (hour - 6) / 13)
        solar = max(0, solar_base + random.uniform(-30, 30))
    else:
        solar = 0
    
    # 充电负荷模拟（早晚高峰）
    if 7 <= hour <= 9 or 17 <= hour <= 20:
        charging_base = 80 + 40 * random.random()
    elif 10 <= hour <= 16:
        charging_base = 30 + 30 * random.random()
    else:
        charging_base = 10 + 20 * random.random()
    charging = max(0, charging_base)
    
    # 储能SOC
    soc = 50 + 20 * np.sin(np.pi * hour / 12) + random.uniform(-5, 5)
    soc = max(STRATEGY_PARAMS["soc_min"], min(STRATEGY_PARAMS["soc_max"], soc))
    
    # 电池功率 (正=充电, 负=放电)
    if solar > charging:
        battery_power = min(solar - charging, STRATEGY_PARAMS["charge_rate"])
        battery_power = min(battery_power, (STRATEGY_PARAMS["soc_max"] - soc) / 100 * 200)
    else:
        battery_power = -(charging - solar)
        battery_power = max(battery_power, -STRATEGY_PARAMS["discharge_rate"])
        battery_power = max(battery_power, (STRATEGY_PARAMS["soc_min"] - soc) / 100 * 200)
    
    # 电网交互功率 (正=取电, 负=馈电)
    grid_power = charging - solar - battery_power
    
    # 状态判定
    if grid_power > STRATEGY_PARAMS["grid_limit"]:
        status = "grid_overload"
    elif battery_power > 0:
        status = "charging"
    elif battery_power < 0:
        status = "discharging"
    else:
        status = "idle"
    
    return {
        "time": now.strftime("%H:%M:%S"),
        "solar_power": solar,
        "charging_power": charging,
        "soc": soc,
        "battery_power": battery_power,
        "grid_power": grid_power,
        "status": status,
    }


def create_realtime_dashboard():
    """创建实时监控仪表板"""
    data = get_realtime_data()
    
    # 状态卡片颜色
    status_colors = {
        "charging": "green",
        "discharging": "orange",
        "grid_overload": "red",
        "idle": "green",
    }
    status_names = {
        "charging": "⚡ 储能充电中",
        "discharging": "🔋 储能放电中",
        "grid_overload": "⚠️ 电网负荷超标",
        "idle": "✅ 系统待机",
    }
    card_color = status_colors.get(data["status"], "green")
    status_text = status_names.get(data["status"], "运行中")
    
    # 实时数据HTML卡片
    cards_html = f"""
    <div style='display:flex; gap:15px; flex-wrap:wrap; justify-content:center;'>
        <div class='status-card' style='flex:1; min-width:180px;'>
            <div class='label-display'>☀️ 光伏发电</div>
            <div class='value-display'>{data['solar_power']:.1f}</div>
            <div class='label-display'>kW</div>
        </div>
        <div class='status-card' style='flex:1; min-width:180px;'>
            <div class='label-display'>🔌 充电负荷</div>
            <div class='value-display'>{data['charging_power']:.1f}</div>
            <div class='label-display'>kW</div>
        </div>
        <div class='status-card {card_color}' style='flex:1; min-width:180px;'>
            <div class='label-display'>🔋 储能SOC</div>
            <div class='value-display'>{data['soc']:.1f}%</div>
            <div class='label-display'>{status_text}</div>
        </div>
        <div class='status-card' style='flex:1; min-width:180px;'>
            <div class='label-display'>⚡ 电网交互</div>
            <div class='value-display'>{data['grid_power']:.1f}</div>
            <div class='label-display'>kW</div>
        </div>
    </div>
    """
    
    # SOC仪表盘
    soc_fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=data["soc"],
        title={"text": "储能 SOC (%)"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#4CAF50" if data["soc"] > 30 else "#F44336"},
            "steps": [
                {"range": [0, 20], "color": "#ffcdd2"},
                {"range": [20, 50], "color": "#fff9c4"},
                {"range": [50, 80], "color": "#c8e6c9"},
                {"range": [80, 100], "color": "#a5d6a7"},
            ],
            "threshold": {
                "line": {"color": "red", "width": 3},
                "thickness": 0.75,
                "value": STRATEGY_PARAMS["soc_min"],
            },
        },
    ))
    soc_fig.update_layout(height=300)
    
    # 功率分布图
    power_fig = go.Figure(data=[
        go.Bar(name="光伏发电", x=["当前功率"], y=[data["solar_power"]],
               marker_color="#FF9800"),
        go.Bar(name="充电负荷", x=["当前功率"], y=[data["charging_power"]],
               marker_color="#2196F3"),
        go.Bar(name="电池功率", x=["当前功率"], y=[data["battery_power"]],
               marker_color="#4CAF50" if data["battery_power"] >= 0 else "#F44336"),
        go.Bar(name="电网交互", x=["当前功率"], y=[data["grid_power"]],
               marker_color="#9C27B0"),
    ])
    power_fig.update_layout(
        title="功率流分布",
        yaxis_title="功率 (kW)",
        height=350, barmode='group',
    )
    
    # 实时功率曲线 (模拟最近30分钟)
    now = datetime.now()
    times_30min = [(now - timedelta(seconds=30 * i)).strftime("%H:%M:%S") for i in range(60, -1, -1)]
    solar_curve = [data["solar_power"] * (0.85 + 0.15 * np.sin(i / 10)) for i in range(61)]
    charging_curve = [data["charging_power"] * (0.9 + 0.1 * np.cos(i / 8)) for i in range(61)]
    
    curve_fig = go.Figure()
    curve_fig.add_trace(go.Scatter(
        x=times_30min, y=solar_curve, name="光伏功率",
        line=dict(color="#FF9800", width=2)
    ))
    curve_fig.add_trace(go.Scatter(
        x=times_30min, y=charging_curve, name="充电负荷",
        line=dict(color="#2196F3", width=2)
    ))
    curve_fig.update_layout(
        title="实时功率曲线 (最近30分钟)",
        xaxis_title="时间", yaxis_title="功率 (kW)",
        height=300, hovermode="x unified",
    )
    
    return cards_html, soc_fig, power_fig, curve_fig


# ============================================================
# Tab 3: 气象环境 - 天气更新
# ============================================================
def weather_update():
    """更新气象环境模块"""
    radiation_plot, warning_html, forecast_df, realtime_info = update_weather_ui()
    
    # 创建额外图表: 温度和降水预报
    try:
        from weather_service import fetch_weather_data
        wd = fetch_weather_data()
        if wd:
            times = wd["full_times"]
            radiations = wd["full_radiations"]
            radiations = [r if r is not None else 0 for r in radiations]
            
            weather_fig = go.Figure()
            weather_fig.add_trace(go.Scatter(
                x=times, y=radiations, name="辐照度",
                line=dict(color="#FF9800", width=2),
                fill='tozeroy', fillcolor="rgba(255,152,0,0.15)"
            ))
            weather_fig.update_layout(
                title="辐照度变化趋势",
                xaxis_title="时间", yaxis_title="W/m²",
                height=300, hovermode="x unified",
            )
        else:
            weather_fig = go.Figure()
            weather_fig.update_layout(title="辐照度 (数据获取中...)")
    except Exception:
        weather_fig = go.Figure()
        weather_fig.update_layout(title="辐照度 (等待刷新)")
    
    return radiation_plot, warning_html, forecast_df, realtime_info, weather_fig


# ============================================================
# Tab 4: 策略模拟
# ============================================================
def run_strategy_simulation(soc_min, soc_max, charge_rate, discharge_rate,
                            grid_limit, feedback_enabled, sim_hours):
    """运行策略模拟"""
    global STRATEGY_PARAMS, SIMULATION_HISTORY
    
    STRATEGY_PARAMS.update({
        "soc_min": soc_min,
        "soc_max": soc_max,
        "charge_rate": charge_rate,
        "discharge_rate": discharge_rate,
        "grid_limit": grid_limit,
        "feedback_enabled": feedback_enabled,
    })
    
    # 获取预测数据
    solar_result = solar_forecast_service.predict(forecast_hours=int(sim_hours))
    charging_result = charging_forecast_service.predict(forecast_hours=int(sim_hours))
    
    if solar_result is None or charging_result is None:
        return None, "⚠️ 模型加载失败", pd.DataFrame()
    
    times = solar_result["times"]
    solar_pred = solar_result["predictions"]
    charging_pred = charging_result["predictions"]
    
    # 模拟储能运行策略
    n_steps = len(times)
    soc = 50.0  # 初始SOC
    battery_history = []
    grid_history = []
    soc_history = []
    
    for i in range(n_steps):
        solar = solar_pred[i]
        load = charging_pred[i]
        net = solar - load
        
        # 策略决策
        if net > 0:
            # 光伏盈余: 优先充电
            charge_power = min(net, charge_rate)
            charge_power = min(charge_power, (soc_max - soc) / 100 * 200)  # 假设200kWh容量
            battery_power = charge_power
            grid_power = net - charge_power
            if feedback_enabled and grid_power > 0:
                grid_power = -grid_power  # 馈电为负
        else:
            # 光伏不足: 优先放电
            deficit = abs(net)
            discharge_power = min(deficit, discharge_rate)
            discharge_power = min(discharge_power, (soc - soc_min) / 100 * 200)
            battery_power = -discharge_power
            grid_power = deficit - discharge_power
        
        # 更新SOC
        soc += battery_power / 200 * 100 / (60 / 15)  # 转换百分比 (简化)
        soc = max(soc_min, min(soc_max, soc))
        
        battery_history.append(battery_power)
        grid_history.append(grid_power)
        soc_history.append(soc)
    
    # 创建结果图表
    sim_fig = go.Figure()
    sim_fig.add_trace(go.Scatter(
        x=times, y=solar_pred, name="光伏预测", line=dict(color="#FF9800", width=2)
    ))
    sim_fig.add_trace(go.Scatter(
        x=times, y=charging_pred, name="充电负荷", line=dict(color="#2196F3", width=2)
    ))
    sim_fig.add_trace(go.Scatter(
        x=times, y=battery_history, name="电池功率", line=dict(color="#4CAF50", width=2)
    ))
    sim_fig.add_trace(go.Scatter(
        x=times, y=grid_history, name="电网交互", line=dict(color="#9C27B0", width=2, dash='dash')
    ))
    sim_fig.update_layout(
        title="策略模拟结果",
        xaxis_title="时间", yaxis_title="功率 (kW)",
        height=400, hovermode="x unified",
    )
    
    # SOC曲线
    soc_fig = go.Figure()
    soc_fig.add_trace(go.Scatter(
        x=times, y=soc_history, name="SOC",
        line=dict(color="#4CAF50", width=3),
        fill='tozeroy', fillcolor="rgba(76,175,80,0.1)"
    ))
    soc_fig.add_hline(y=soc_max, line_dash="dash", line_color="green",
                      annotation_text=f"SOC上限: {soc_max}%")
    soc_fig.add_hline(y=soc_min, line_dash="dash", line_color="red",
                      annotation_text=f"SOC下限: {soc_min}%")
    soc_fig.update_layout(
        title="储能SOC变化",
        xaxis_title="时间", yaxis_title="SOC (%)",
        height=300,
    )
    
    # 统计信息
    total_from_grid = sum(max(0, g) for g in grid_history) * (sim_hours / n_steps)
    total_to_grid = sum(max(0, -g) for g in grid_history) * (sim_hours / n_steps)
    total_solar = sum(solar_pred) * (sim_hours / n_steps)
    total_load = sum(charging_pred) * (sim_hours / n_steps)
    grid_peak = max(grid_history)
    soc_final = soc_history[-1]
    
    summary_html = f"""
    <div style='background:#f5f5f5; border-radius:8px; padding:15px;'>
    <h3>📊 模拟结果汇总</h3>
    <table style='width:100%;'>
    <tr><td>☀️ 总光伏发电:</td><td><b>{total_solar:.1f} kWh</b></td></tr>
    <tr><td>🔌 总充电需求:</td><td><b>{total_load:.1f} kWh</b></td></tr>
    <tr><td>⬇️ 从电网取电:</td><td><b>{total_from_grid:.1f} kWh</b></td></tr>
    <tr><td>⬆️ 向电网馈电:</td><td><b>{total_to_grid:.1f} kWh</b></td></tr>
    <tr><td>🔋 最终SOC:</td><td><b>{soc_final:.1f}%</b></td></tr>
    <tr><td>⚡ 电网峰值负荷:</td><td><b>{grid_peak:.1f} kW</b></td></tr>
    </table>
    </div>
    """
    
    # 数据表格
    df = pd.DataFrame({
        "时间": times,
        "光伏预测 (kW)": [f"{p:.1f}" for p in solar_pred],
        "充电负荷 (kW)": [f"{p:.1f}" for p in charging_pred],
        "电池功率 (kW)": [f"{p:.1f}" for p in battery_history],
        "电网交互 (kW)": [f"{p:.1f}" for p in grid_history],
        "SOC (%)": [f"{s:.1f}" for s in soc_history],
    })
    
    return sim_fig, soc_fig, summary_html, df


# ============================================================
# 主 UI 构建
# ============================================================
def create_app():
    """创建 Gradio 应用"""
    
    with gr.Blocks(
        title="光储充一体化设备智能管理平台",
        css=CUSTOM_CSS,
        theme=gr.themes.Soft(),
    ) as app:
        
        # 标题
        gr.HTML("""
        <div class='title-section'>
            <h1>⚡ 光储充一体化设备智能管理平台</h1>
            <p>Solar-Storage-Charging Integrated Smart Management Platform</p>
            <hr style='width:60%; margin:15px auto; border-color:#e0e0e0;'>
        </div>
        """)
        
        # ========================
        # Tab 1: 预测核心
        # ========================
        with gr.Tabs():
            with gr.TabItem("🔮 预测核心", elem_classes="tab-header"):
                gr.Markdown("## 预测核心模块\n基于深度学习的多尺度功率预测，支持光伏发电预测与充电负荷预测")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### ☀️ 光伏发电预测")
                        solar_hours = gr.Dropdown(
                            choices=[1, 2, 4, 6, 8, 12, 24],
                            value=4, label="预测时长 (小时)",
                            interactive=True
                        )
                        solar_btn = gr.Button("🚀 开始预测", variant="primary", size="lg")
                        solar_stats = gr.HTML(label="统计信息")
                    
                    with gr.Column(scale=2):
                        solar_plot = gr.Plot(label="光伏预测曲线")
                        solar_table = gr.Dataframe(label="预测数据明细", headers=["时间", "预测功率 (kW)"])
                
                gr.Markdown("---")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 🔌 充电负荷预测")
                        charging_hours = gr.Dropdown(
                            choices=[1, 2, 4, 6, 8, 12, 24],
                            value=4, label="预测时长 (小时)",
                            interactive=True
                        )
                        charging_btn = gr.Button("🚀 开始预测", variant="primary", size="lg")
                        charging_stats = gr.HTML(label="统计信息")
                    
                    with gr.Column(scale=2):
                        charging_plot = gr.Plot(label="充电负荷预测曲线")
                        charging_table = gr.Dataframe(label="预测数据明细", headers=["时间", "预测负荷 (kW)"])
                
                solar_btn.click(
                    fn=solar_forecast_tab(),
                    inputs=[solar_hours],
                    outputs=[solar_plot, solar_stats, solar_table],
                )
                
                charging_btn.click(
                    fn=charging_forecast_tab(),
                    inputs=[charging_hours],
                    outputs=[charging_plot, charging_stats, charging_table],
                )
            
            # ========================
            # Tab 2: 实时监控
            # ========================
            with gr.TabItem("📡 实时监控", elem_classes="tab-header"):
                gr.Markdown("## 实时监控\n系统运行状态实时展示，功率流分析与SOC监测")
                
                refresh_btn = gr.Button("🔄 刷新数据", variant="secondary")
                realtime_cards = gr.HTML()
                
                with gr.Row():
                    with gr.Column(scale=1):
                        soc_gauge = gr.Plot(label="储能SOC仪表盘")
                    with gr.Column(scale=1):
                        power_bar = gr.Plot(label="功率流分布")
                
                power_curve = gr.Plot(label="实时功率曲线")
                
                refresh_btn.click(
                    fn=create_realtime_dashboard,
                    inputs=[],
                    outputs=[realtime_cards, soc_gauge, power_bar, power_curve],
                )
            
            # ========================
            # Tab 3: 气象环境
            # ========================
            with gr.TabItem("🌤️ 气象环境", elem_classes="tab-header"):
                gr.Markdown("## 气象环境监测\n实时气象数据获取与天气预警")
                
                weather_refresh_btn = gr.Button("🔄 刷新气象数据", variant="secondary")
                weather_warning = gr.HTML(label="天气预警")
                realtime_weather_info = gr.Markdown()
                
                with gr.Row():
                    with gr.Column(scale=2):
                        weather_rad_plot = gr.Plot(label="辐照度与云量曲线")
                    with gr.Column(scale=1):
                        weather_forecast_table = gr.Dataframe(
                            label="未来2小时预报",
                            headers=["时间", "天气", "辐照度 (W/m²)", "降雨量 (mm)"]
                        )
                
                weather_extra_plot = gr.Plot(label="辐照度变化趋势")
                
                weather_refresh_btn.click(
                    fn=weather_update,
                    inputs=[],
                    outputs=[
                        weather_rad_plot,
                        weather_warning,
                        weather_forecast_table,
                        realtime_weather_info,
                        weather_extra_plot,
                    ],
                )
            
            # ========================
            # Tab 4: 策略模拟
            # ========================
            with gr.TabItem("🎯 策略模拟", elem_classes="tab-header"):
                gr.Markdown("## 储能策略模拟\n基于预测结果的储能充放电策略仿真")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 策略参数设置")
                        soc_min_slider = gr.Slider(
                            0, 50, value=20, step=5, label="SOC下限 (%)"
                        )
                        soc_max_slider = gr.Slider(
                            50, 100, value=90, step=5, label="SOC上限 (%)"
                        )
                        charge_rate_slider = gr.Slider(
                            10, 200, value=60, step=10, label="充电功率上限 (kW)"
                        )
                        discharge_rate_slider = gr.Slider(
                            10, 200, value=60, step=10, label="放电功率上限 (kW)"
                        )
                        grid_limit_slider = gr.Slider(
                            20, 500, value=100, step=10, label="电网交互功率上限 (kW)"
                        )
                        feedback_toggle = gr.Checkbox(
                            value=True, label="允许余电上网"
                        )
                        sim_hours_dropdown = gr.Dropdown(
                            choices=[4, 6, 8, 12, 24],
                            value=8, label="模拟时长 (小时)"
                        )
                        sim_btn = gr.Button("🎯 开始模拟", variant="primary", size="lg")
                        sim_summary = gr.HTML(label="模拟结果汇总")
                    
                    with gr.Column(scale=2):
                        sim_plot = gr.Plot(label="策略模拟功率曲线")
                        soc_sim_plot = gr.Plot(label="SOC变化曲线")
                        sim_table = gr.Dataframe(label="模拟数据明细")
                
                sim_btn.click(
                    fn=run_strategy_simulation,
                    inputs=[
                        soc_min_slider, soc_max_slider,
                        charge_rate_slider, discharge_rate_slider,
                        grid_limit_slider, feedback_toggle,
                        sim_hours_dropdown,
                    ],
                    outputs=[sim_plot, soc_sim_plot, sim_summary, sim_table],
                )
        
        # 底部信息
        gr.HTML("""
        <div style='text-align:center; margin-top:30px; padding:15px; color:#999; font-size:12px;'>
        <hr style='width:80%; border-color:#e0e0e0;'>
        <p>光储充一体化设备智能管理平台 v0.1 | 基于 Gradio 构建 | 测试版 Demo</p>
        <p>光伏预测: LSTM-GAN 模型 | 充电负荷预测: TCN-Attention-LSTM 模型</p>
        </div>
        """)
    
    return app


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  光储充一体化设备智能管理平台")
    print("  Solar-Storage-Charging Management Platform")
    print("=" * 60)
    print(f"  启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  光伏预测模型: Solar_Forecast/best_pth/")
    print(f"  充电预测模型: Charging_Forecast/best_pth/")
    print(f"  天气数据源: Open-Meteo API")
    print("=" * 60)
    print("\n正在启动 Gradio 应用...")
    
    app = create_app()
    app.queue(default_concurrency_limit=3).launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
    )