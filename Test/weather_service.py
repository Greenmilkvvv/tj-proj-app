"""
天气服务适配模块 - 封装 Open-Meteo API 天气数据获取
适配 Weather/get_weather.py 中的功能到 Gradio 应用
"""
import os
import sys
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests

# 添加 Weather 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Weather'))

# 目标坐标 (宁波)
LAT = 29.866465
LON = 121.52707


def fetch_current_weather(lat=LAT, lon=LON):
    """获取最接近当前时刻的实时气象数据"""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "shortwave_radiation,cloudcover,rain,temperature,wind_speed_10m",
        "timezone": "Asia/Shanghai",
    }
    try:
        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()
        js = r.json()
        
        current = js.get("current", {})
        return {
            "time": current.get("time"),
            "radiation": current.get("shortwave_radiation", 0),
            "cloudcover": current.get("cloudcover", 0),
            "rain": current.get("rain", 0),
            "temperature": current.get("temperature_2m", 0),
            "wind_speed": current.get("wind_speed_10m", 0),
        }
    except Exception as e:
        print(f"[WARNING] 实时天气获取失败: {e}")
        return None


def fetch_weather_data(lat=LAT, lon=LON):
    """从 Open-Meteo 获取实时 + 15分钟预报数据"""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "shortwave_radiation,cloudcover,rain",
        "minutely_15": "shortwave_radiation,cloudcover,rain",
        "forecast_days": 1,
        "timezone": "Asia/Shanghai",
    }
    try:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        js = r.json()
        
        # 当前数据
        current = js.get("current", {})
        
        # 获取15分钟预报数据
        times = pd.to_datetime(js["minutely_15"]["time"])
        radiations = js["minutely_15"]["shortwave_radiation"]
        cloudcovers = js["minutely_15"]["cloudcover"]
        rains = js["minutely_15"]["rain"]
        
        # 对齐当前时间
        now = pd.Timestamp.now(tz="Asia/Shanghai")
        now_time = now.floor("15min")
        
        # 处理时区
        times = pd.Series(times)
        try:
            times = times.dt.tz_localize("Asia/Shanghai")
        except Exception:
            # 可能已有时区
            pass
        
        # 找到最接近当前时间的数据索引
        try:
            idx = (times - now_time).abs().argmin()
        except Exception:
            idx = 0
        
        current_radiation = radiations[idx] if radiations[idx] is not None else 0
        current_cloudcover = cloudcovers[idx] if cloudcovers[idx] is not None else 0
        
        # 构建未来2小时预报 DataFrame
        end_idx = min(idx + 8, len(times))  # 8个15分钟 = 2小时
        forecast_df = pd.DataFrame({
            "时间": [times[i].strftime("%H:%M") for i in range(idx, end_idx)],
            "辐照度 (W/m²)": [radiations[i] if radiations[i] is not None else 0 for i in range(idx, end_idx)],
            "云量 (%)": [cloudcovers[i] if cloudcovers[i] is not None else 0 for i in range(idx, end_idx)],
            "降雨量 (mm)": [rains[i] if rains[i] is not None else 0 for i in range(idx, end_idx)],
        })
        
        # 天气状况文字描述
        def get_weather_desc(cloud, rain):
            if rain is None or (isinstance(rain, float) and np.isnan(rain)):
                rain = 0
            if cloud is None or (isinstance(cloud, float) and np.isnan(cloud)):
                cloud = 50
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
        
        # 预警逻辑
        rain_values = forecast_df["降雨量 (mm)"].values
        has_warning = any(r is not None and r > 5 for r in rain_values)
        warning_msg = get_warning_message(has_warning, forecast_df)
        
        return {
            "current_radiation": current_radiation,
            "current_cloudcover": current_cloudcover,
            "forecast_df": forecast_df,
            "has_warning": has_warning,
            "warning_msg": warning_msg,
            "full_times": times,
            "full_radiations": radiations,
            "full_cloudcovers": cloudcovers,
        }
    except Exception as e:
        print(f"[WARNING] 天气数据获取失败: {e}")
        return None


def get_warning_message(has_warning, forecast_df):
    """生成预警文本"""
    if has_warning:
        try:
            max_rain_idx = forecast_df["降雨量 (mm)"].idxmax()
            time_str = forecast_df.loc[max_rain_idx, "时间"]
            rain_val = forecast_df.loc[max_rain_idx, "降雨量 (mm)"]
            return f"""
            <div style='background-color:#ffebee; border-left:4px solid #f44336; padding:12px; margin:8px 0; color:#333333;'>
            🚨 <strong>天气预警</strong><br>
            ⚠️ 预计 {time_str} 有强降雨 ({rain_val:.1f}mm/h)<br>
            📉 光伏出力预计骤降，建议提前调整储能策略以应对充电负荷
            </div>
            """
        except Exception:
            pass
    return """
    <div style='background-color:#e8f5e9; border-left:4px solid #4caf50; padding:12px; margin:8px 0; color:#333333;'>
    🟢 <strong>天气正常</strong><br>
    当前无恶劣天气预警，光伏出力条件良好
    </div>
    """


def create_radiation_plot(weather_data):
    """创建辐照度与云量曲线图"""
    times = weather_data["full_times"]
    radiations = weather_data["full_radiations"]
    cloudcovers = weather_data["full_cloudcovers"]
    
    # 处理 None 值
    radiations = [r if r is not None else 0 for r in radiations]
    cloudcovers = [c if c is not None else 0 for c in cloudcovers]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=times,
        y=radiations,
        name="太阳辐射强度",
        line=dict(color="orange", width=2),
        fill='tozeroy',
        fillcolor="rgba(255, 165, 0, 0.2)"
    ))
    
    fig.add_trace(go.Scatter(
        x=times,
        y=cloudcovers,
        name="云层覆盖率",
        line=dict(color="gray", width=2, dash="dash"),
        yaxis="y2"
    ))
    
    fig.update_layout(
        title="太阳辐射强度与云层覆盖率",
        xaxis=dict(title="时间"),
        yaxis=dict(title="辐射强度 (W/m²)", tickfont=dict(color="orange")),
        yaxis2=dict(
            title="云量 (%)",
            tickfont=dict(color="gray"),
            overlaying="y",
            side="right",
            range=[0, 100]
        ),
        legend=dict(x=0.01, y=0.99),
        hovermode="x unified",
        height=350,
    )
    
    return fig


def update_weather_ui():
    """刷新整个气象模块的 UI - 供 Gradio 调用
    
    Returns:
        tuple: (radiation_plot, warning_html, forecast_df, realtime_info)
    """
    try:
        weather_data = fetch_weather_data()
        
        if weather_data is None:
            raise ValueError("无法获取天气数据")
        
        # 辐照度曲线图
        radiation_plot = create_radiation_plot(weather_data)
        
        # 预警 HTML
        warning_html = weather_data["warning_msg"]
        
        # 未来2小时预报表格
        forecast_display = weather_data["forecast_df"][
            ["时间", "天气", "辐照度 (W/m²)", "降雨量 (mm)"]
        ]
        
        # 实时信息
        realtime_info = f"""
        📍 实时辐照度: {weather_data['current_radiation']:.0f} W/m² &nbsp;|&nbsp; 
        云量: {weather_data['current_cloudcover']:.0f}%
        """
        
        return radiation_plot, warning_html, forecast_display, realtime_info
        
    except Exception as e:
        print(f"[WARNING] update_weather_ui 失败: {e}")
        # 返回默认值
        return _mock_weather_update()


def _mock_weather_update():
    """天气数据获取失败时的回退模拟数据"""
    now = datetime.now()
    times = pd.date_range(now - timedelta(hours=6), now + timedelta(hours=12), freq='15min')
    
    radiations = []
    cloudcovers = []
    for t in times:
        hour = t.hour + t.minute / 60
        if 5 <= hour <= 19:
            rad = 800 * np.sin(np.pi * (hour - 5) / 14)
            cloud = 30 + 20 * np.sin(np.pi * hour / 12)
        else:
            rad = 0
            cloud = 50
        radiations.append(rad + np.random.normal(0, 20))
        cloudcovers.append(cloud + np.random.normal(0, 5))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times, y=radiations, name="太阳辐射强度",
        line=dict(color="orange", width=2),
        fill='tozeroy', fillcolor="rgba(255, 165, 0, 0.2)"
    ))
    fig.add_trace(go.Scatter(
        x=times, y=cloudcovers, name="云层覆盖率",
        line=dict(color="gray", width=2, dash="dash"), yaxis="y2"
    ))
    fig.update_layout(
        title="太阳辐射强度与云层覆盖率 (模拟数据)",
        xaxis=dict(title="时间"),
        yaxis=dict(title="辐射强度 (W/m²)", tickfont=dict(color="orange")),
        yaxis2=dict(title="云量 (%)", tickfont=dict(color="gray"), overlaying="y", side="right", range=[0, 100]),
        legend=dict(x=0.01, y=0.99),
        hovermode="x unified",
        height=350,
    )
    
    warning_html = """
    <div style='background-color:#fff3e0; border-left:4px solid #ff9800; padding:12px; margin:8px 0; color:#333333;'>
    🟡 <strong>降级模式</strong><br>
    天气API暂不可用，使用模拟数据
    </div>
    """
    
    future_times = pd.date_range(now, now + timedelta(hours=2), freq='15min')
    forecast_df = pd.DataFrame({
        "时间": [t.strftime("%H:%M") for t in future_times],
        "天气": ["⛅ 多云"] * 8,
        "辐照度 (W/m²)": [800, 780, 750, 720, 680, 640, 600, 550],
        "降雨量 (mm)": [0] * 8,
    })
    
    realtime_info = "📍 实时辐照度: N/A  |  云量: N/A (模拟模式)"
    
    return fig, warning_html, forecast_df, realtime_info


def get_weather_data():
    """获取天气数据用于模型输入
    
    Returns:
        dict: 包含 current_radiation, current_cloudcover, temperature 等
    """
    current = fetch_current_weather()
    if current:
        return {
            "current_radiation": current["radiation"],
            "current_cloudcover": current["cloudcover"],
            "temperature": current["temperature"],
            "rain": current["rain"],
            "wind_speed": current["wind_speed"],
        }
    
    # 回退默认值
    return {
        "current_radiation": 600.0,
        "current_cloudcover": 30.0,
        "temperature": 25.0,
        "rain": 0.0,
        "wind_speed": 3.0,
    }