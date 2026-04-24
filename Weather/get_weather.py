# %%
import gradio as gr
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests

# 目标的经纬度
NINGBO_LAT = 29.866465
NINGBO_LON = 121.52707


# %%
# ==========  实时的气象数据获取  ==========
def fetch_current_weather(lat, lon):
    """获取最接近当前时刻的实时气象数据"""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "shortwave_radiation,cloudcover,rain,temperature,wind_speed_10m",
        "timezone": "Asia/Shanghai",
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    js = r.json()
    
    current = js.get("current", {})
    return {
        "time": current.get("time"),           # 当前数据对应的时间
        "radiation": current.get("shortwave_radiation"),   # W/m²
        "cloudcover": current.get("cloudcover"),           # %
        "rain": current.get("rain"),                       # mm/h
        "temperature": current.get("temperature_2m")       # °C
    }

# 测试
# data = fetch_current_weather(29.8683, 121.5440)
# print(f"数据时间: {data['time']}")
# print(f"辐射强度: {data['radiation']} W/m²")
# print(f"云量: {data['cloudcover']}%")


# %%
# ==========  定义气象数据获取函数 ==========
def fetch_weather_data():
    """从 Open-Meteo 获取实时 + 预报数据"""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": NINGBO_LAT,
        "longitude": NINGBO_LON,
        "minutely_15": "shortwave_radiation,cloudcover,rain",
        "forecast_days": 1,
        "timezone": "Asia/Shanghai",
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    js = r.json()
    
    times = pd.to_datetime(js["minutely_15"]["time"])
    radiations = js["minutely_15"]["shortwave_radiation"]
    cloudcovers = js["minutely_15"]["cloudcover"]
    rains = js["minutely_15"]["rain"]
    
    # 找到当前时刻（取最新数据点）
    now = pd.Timestamp.now(tz="Asia/Shanghai")
    now_hour = now.floor("15min")  # 对齐到整点
    
    # 找到最接近当前小时的数据索引
    idx = (times - now_hour).abs().argmin() if now_hour in times.values else 0
    
    current_radiation = radiations[idx] if radiations[idx] is not None else 0
    current_cloudcover = cloudcovers[idx] if cloudcovers[idx] is not None else 0
    
    # 构建未来2小时预报 DataFrame
    forecast_df = pd.DataFrame({
        "时间": [times[i].strftime("%H:%M") for i in range(idx, min(idx+3, len(times)))],
        "辐照度 (W/m²)": [radiations[i] if radiations[i] is not None else 0 for i in range(idx, min(idx+3, len(times)))],
        "云量 (%)": [cloudcovers[i] if cloudcovers[i] is not None else 0 for i in range(idx, min(idx+3, len(times)))],
        "降雨量 (mm)": [rains[i] if rains[i] is not None else 0 for i in range(idx, min(idx+3, len(times)))]
    })
    
    # 天气状况文字描述
    def get_weather_desc(cloud, rain):
        if rain > 1:
            return "🌧️ 大雨"
        elif rain > 0:
            return "🌦️ 小雨"
        elif cloud > 80:
            return "☁️ 阴天"
        elif cloud > 40:
            return "⛅ 多云"
        else:
            return "☀️ 晴天"
    
    forecast_df["天气"] = forecast_df.apply(
        lambda row: get_weather_desc(row["云量 (%)"], row["降雨量 (mm)"]), axis=1
    )
    
    # 预警逻辑：未来2小时有强降雨（降雨量 > 5mm/h）
    has_warning = any(rain > 5 for rain in forecast_df["降雨量 (mm)"].values)
    warning_msg = get_warning_message(has_warning, forecast_df)
    
    return {
        "current_radiation": current_radiation,
        "current_cloudcover": current_cloudcover,
        "forecast_df": forecast_df,
        "has_warning": has_warning,
        "warning_msg": warning_msg,
        "full_times": times,
        "full_radiations": radiations,
        "full_cloudcovers": cloudcovers
    }




def get_warning_message(has_warning, forecast_df):
    """生成预警文本"""
    if has_warning:
        # 找出降雨量最大的时段
        max_rain_idx = forecast_df["降雨量 (mm)"].idxmax()
        time_str = forecast_df.loc[max_rain_idx, "时间"]
        rain_val = forecast_df.loc[max_rain_idx, "降雨量 (mm)"]
        return f"""
        <div class='warning-banner' style='background-color:#ffebee; border-left-color:#f44336;'>
        🚨 <strong>天气预警</strong><br>
        ⚠️ 预计 {time_str} 有强降雨 ({rain_val:.1f}mm/h)<br>
        📉 光伏出力预计骤降，建议提前调整储能策略以应对充电负荷
        </div>
        """
    else:
        return """
        <div class='warning-banner' style='background-color:#e8f5e9; border-left-color:#4caf50;'>
        🟢 <strong>天气正常</strong><br>
        当前无恶劣天气预警，光伏出力条件良好
        </div>
        """


# %%
def create_radiation_plot(weather_data):
    """创建辐照度与云量曲线图"""
    times = weather_data["full_times"]
    radiations = weather_data["full_radiations"]
    cloudcovers = weather_data["full_cloudcovers"]
    
    fig = go.Figure()
    
    # 添加辐照度曲线（主 Y 轴）
    fig.add_trace(go.Scatter(
        x=times,
        y=radiations,
        name="太阳辐射强度",
        line=dict(color="orange", width=2),
        fill='tozeroy',
        fillcolor="rgba(255, 165, 0, 0.2)"
    ))
    
    # 添加云量曲线（次 Y 轴）
    fig.add_trace(go.Scatter(
        x=times,
        y=cloudcovers,
        name="云层覆盖率",
        line=dict(color="gray", width=2, dash="dash"),
        yaxis="y2"
    ))
    
    # 设置双 Y 轴
    fig.update_layout(
        title="太阳辐射强度与云层覆盖率",
        xaxis=dict(title="时间"),
        yaxis=dict(title="辐射强度 (W/m²)", titlefont=dict(color="orange")),
        yaxis2=dict(
            title="云量 (%)",
            titlefont=dict(color="gray"),
            overlaying="y",
            side="right",
            range=[0, 100]
        ),
        legend=dict(x=0.01, y=0.99),
        hovermode="x unified"
    )
    
    return fig


# %%
# ========== 2. 创建 UI 并绑定更新函数 ==========
def update_weather_ui():
    """刷新整个气象模块的 UI"""
    weather_data = fetch_weather_data()
    
    # 更新辐照度曲线图
    radiation_plot = create_radiation_plot(weather_data)
    
    # 更新预警框（通过 HTML 组件）
    warning_html = weather_data["warning_msg"]
    
    # 更新未来2小时预报表格
    forecast_display = weather_data["forecast_df"][["时间", "天气", "辐照度 (W/m²)", "降雨量 (mm)"]]
    
    # 同时更新实时辐照度和云量显示（如果需要）
    realtime_info = f"""
    📍 实时辐照度: {weather_data['current_radiation']:.0f} W/m² &nbsp;|&nbsp; 
    云量: {weather_data['current_cloudcover']:.0f}%
    """
    
    return radiation_plot, warning_html, forecast_display, realtime_info


# %%
# ========== 3. Gradio UI 代码 ==========
# 在你的主 UI 中添加以下内容

# 假设你已经在之前的代码中创建了 demo
with gr.Blocks(title="光储充一体化智能预测平台", css=custom_css, theme=gr.themes.Soft()) as demo:
    # ... 前面的代码（左侧边栏、Tab1、Tab2等）...
    
    # ---------- Tab 3: 气象与环境 ----------
    with gr.TabItem("🌤️ 气象与环境"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("### ☀️ 辐照度与云量")
                # 替换为动态更新的 Plot
                radiation_plot = gr.Plot(label="辐照度与云量曲线")
                
            with gr.Column():
                gr.Markdown("### ⚠️ 天气预警")
                # 替换为动态更新的 HTML
                warning_html = gr.HTML(value="加载中...")
                
                gr.Markdown("---")
                gr.Markdown("### 📅 未来2小时预报")
                # 替换为动态更新的 DataFrame
                forecast_table = gr.Dataframe(
                    headers=["时间", "天气", "辐照度 (W/m²)", "降雨量 (mm)"],
                    interactive=False
                )
                
                # 可选：显示实时数据的小卡片
                gr.Markdown("---")
                realtime_info = gr.Markdown("### 实时数据加载中...")
    
    # 添加刷新按钮到气象模块内部（或复用左侧的全局刷新按钮）
    with gr.Row():
        refresh_weather_btn = gr.Button("🔄 刷新气象数据", variant="secondary")
    
    # 绑定更新事件
    refresh_weather_btn.click(
        fn=update_weather_ui,
        inputs=[],
        outputs=[radiation_plot, warning_html, forecast_table, realtime_info]
    )
    
    # 页面加载时自动刷新一次
    demo.load(fn=update_weather_ui, outputs=[radiation_plot, warning_html, forecast_table, realtime_info])

# 启动
if __name__ == "__main__":
    demo.launch()