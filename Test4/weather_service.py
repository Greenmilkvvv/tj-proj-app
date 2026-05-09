"""
气象服务模块 — 封装 Open-Meteo API
提供实时气象数据获取 + 未来预报 + 预警判断
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import warnings
warnings.filterwarnings("ignore")

from config import LATITUDE, LONGITUDE, WEATHER_API_URL, WEATHER_TIMEOUT


# ============================================================
# 核心数据获取
# ============================================================
def fetch_weather_data(lat=LATITUDE, lon=LONGITUDE):
    """
    从 Open-Meteo 获取实时 + 15min 预报数据
    Returns: dict 或 None (失败时)
    """
    url = WEATHER_API_URL
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,shortwave_radiation,cloudcover,rain",
        "minutely_15": "shortwave_radiation,cloudcover,rain",
        "forecast_days": 1,
        "timezone": "Asia/Shanghai",
    }
    try:
        r = requests.get(url, params=params, timeout=WEATHER_TIMEOUT)
        r.raise_for_status()
        js = r.json()
    except Exception as e:
        print(f"[WARNING] 气象 API 请求失败: {e}")
        return None

    try:
        current = js.get("current", {})
        # 15分钟预报
        times = pd.to_datetime(js["minutely_15"]["time"])
        radiations = js["minutely_15"]["shortwave_radiation"]
        cloudcovers = js["minutely_15"]["cloudcover"]
        rains = js["minutely_15"]["rain"]

        # 对齐当前时间
        now = pd.Timestamp.now(tz="Asia/Shanghai")
        now_time = now.floor("15min")

        times = pd.Series(times)
        try:
            times = times.dt.tz_localize("Asia/Shanghai")
        except Exception:
            pass  # 可能已有时区

        # 找到当前索引
        idx = (times - now_time).abs().argmin()

        current_radiation = radiations[idx] if radiations[idx] is not None else 0
        current_cloudcover = cloudcovers[idx] if cloudcovers[idx] is not None else 0
        current_rain = rains[idx] if rains[idx] is not None else 0
        current_temp = current.get("temperature_2m")
        if current_temp is None:
            current_temp = "—"

        # 构建未来3小时预报 (12步)
        end_idx = min(idx + 12, len(times))
        forecast_df = pd.DataFrame({
            "时间": [times[i].strftime("%H:%M") for i in range(idx, end_idx)],
            "辐照度 (W/m²)": [radiations[i] if radiations[i] is not None else 0 for i in range(idx, end_idx)],
            "云量 (%)": [cloudcovers[i] if cloudcovers[i] is not None else 0 for i in range(idx, end_idx)],
            "降雨量 (mm)": [rains[i] if rains[i] is not None else 0 for i in range(idx, end_idx)],
        })

        # 天气描述
        def weather_desc(cloud, rain):
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
            lambda r: weather_desc(r["云量 (%)"], r["降雨量 (mm)"]), axis=1
        )

        # 预警判断
        has_warning = any(r > 5 for r in forecast_df["降雨量 (mm)"].values)
        warning_msg = _build_warning(has_warning, forecast_df)

        # 今日辐照度曲线（全天）
        day_idx = max(0, idx - (now.hour * 4 + now.minute // 15))
        day_times = times[day_idx:day_idx + 96]  # 96步 = 24h
        day_rads = radiations[day_idx:day_idx + 96]
        day_clouds = cloudcovers[day_idx:day_idx + 96]

        return {
            "current_time": current.get("time", str(now)),
            "current_temp": current_temp,
            "current_radiation": current_radiation,
            "current_cloudcover": current_cloudcover,
            "current_rain": current_rain,
            "forecast_df": forecast_df,
            "has_warning": has_warning,
            "warning_msg": warning_msg,
            "day_times": [t.strftime("%H:%M") for t in day_times],
            "day_radiations": [r if r is not None else 0 for r in day_rads],
            "day_cloudcovers": [c if c is not None else 0 for c in day_clouds],
        }
    except Exception as e:
        print(f"[WARNING] 气象数据解析失败: {e}")
        return None


def _build_warning(has_warning, forecast_df):
    """生成预警 HTML 文本 (theme-adaptive: no hardcoded text color)"""
    if has_warning:
        max_row = forecast_df.loc[forecast_df["降雨量 (mm)"].idxmax()]
        return (
            f"<div style='background:rgba(255,193,7,0.12); border-left:4px solid #ffc107; "
            f"padding:12px; margin:8px 0; border-radius:4px;'>"
            f"🚨 <strong>天气预警</strong><br>"
            f"⚠️ 预计 {max_row['时间']} 有强降雨 ({max_row['降雨量 (mm)']:.1f} mm/h)<br>"
            f"📉 光伏出力可能骤降，请关注充电策略调整"
            f"</div>"
        )
    else:
        return (
            "<div style='background:rgba(40,167,69,0.12); border-left:4px solid #28a745; "
            "padding:12px; margin:8px 0; border-radius:4px;'>"
            "🟢 <strong>天气正常</strong> — 当前无恶劣天气预警，光伏出力条件良好"
            "</div>"
        )


# ============================================================
# 图表构建
# ============================================================
def build_radiation_chart(weather_data):
    """辐照度与云量双轴曲线"""
    if weather_data is None:
        return go.Figure()

    fig = go.Figure()

    # 辐照度
    fig.add_trace(go.Scatter(
        x=weather_data["day_times"],
        y=weather_data["day_radiations"],
        mode="lines",
        name="辐照度 (W/m²)",
        line=dict(color="#ff9800", width=2),
        fill="tozeroy",
        fillcolor="rgba(255,152,0,0.15)",
        yaxis="y1",
    ))

    # 云量
    fig.add_trace(go.Scatter(
        x=weather_data["day_times"],
        y=weather_data["day_cloudcovers"],
        mode="lines",
        name="云量 (%)",
        line=dict(color="#90a4ae", width=1.5, dash="dot"),
        yaxis="y2",
    ))

    fig.update_layout(
        title="今日辐照度与云量趋势",
        xaxis=dict(title="时间", tickangle=45),
        yaxis=dict(title="辐照度 (W/m²)", side="left"),
        yaxis2=dict(title="云量 (%)", side="right", overlaying="y", range=[0, 100]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=40, b=40),
        template="plotly_white",
        height=350,
    )
    return fig


def get_current_weather_summary(weather_data):
    """当前气象概览 HTML (theme-adaptive: no hardcoded dark text colors)"""
    if weather_data is None:
        return "<div style='color:#dc3545;'>⚠️ 气象数据获取失败，请检查网络连接</div>"

    # 温度格式化
    current_temp = weather_data.get("current_temp", "—")
    if isinstance(current_temp, (int, float)):
        temp_display = f"{current_temp:.1f}°C"
    else:
        temp_display = f"{current_temp}°C"

    return f"""
    <div style='display:grid; grid-template-columns: repeat(4, 1fr); gap:12px;'>
        <div style='background:rgba(25,118,210,0.10); border-radius:8px; padding:14px; text-align:center;'>
            <div style='font-size:12px; opacity:0.7;'>🌡️ 温度</div>
            <div style='font-size:24px; font-weight:bold; color:#1976d2;'>{temp_display}</div>
        </div>
        <div style='background:rgba(230,81,0,0.10); border-radius:8px; padding:14px; text-align:center;'>
            <div style='font-size:12px; opacity:0.7;'>☁️ 云量</div>
            <div style='font-size:24px; font-weight:bold; color:#e65100;'>{weather_data['current_cloudcover']}%</div>
        </div>
        <div style='background:rgba(245,127,23,0.10); border-radius:8px; padding:14px; text-align:center;'>
            <div style='font-size:12px; opacity:0.7;'>☀️ 辐照度</div>
            <div style='font-size:24px; font-weight:bold; color:#f57f17;'>{weather_data['current_radiation']}</div>
            <div style='font-size:11px; opacity:0.5;'>W/m²</div>
        </div>
        <div style='background:rgba(46,125,50,0.10); border-radius:8px; padding:14px; text-align:center;'>
            <div style='font-size:12px; opacity:0.7;'>🌧️ 降雨</div>
            <div style='font-size:24px; font-weight:bold; color:#2e7d32;'>{weather_data['current_rain']}</div>
            <div style='font-size:11px; opacity:0.5;'>mm/h</div>
        </div>
    </div>
    <div style='margin-top:8px; font-size:12px; opacity:0.5; text-align:right;'>
        📅 数据时间: {weather_data['current_time']}
    </div>
    """