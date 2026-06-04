"""
BentoML 主服务文件 — 光储充智能预测 Demo
把所有子服务包装成 BentoML Runner，提供 RESTful API + 静态前端
"""
from __future__ import annotations

import os, sys, io, json, base64
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import bentoml
from bentoml.io import JSON, Text

# 让子模块可 import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    SERVICE_PORT, SERVICE_TITLE,
    PEAK_PRICE, MID_PRICE, VALLEY_PRICE,
    PREDICTION_OPTIONS, DEFAULT_PREDICTION_STEPS,
)

# BentoML 服务定义
svc = bentoml.Service(
    name="solar_charging_demo",
    runners=[],
)


# ============================================================
# 工具函数
# ============================================================
def _get_price(hour):
    if 8 <= hour < 11 or 18 <= hour < 21:
        return 0.85
    elif 6 <= hour < 8 or 11 <= hour < 18 or 21 <= hour < 22:
        return 0.63
    else:
        return 0.25


def _make_plotly_json(fig):
    """把 matplotlib/plotly figure 转为 JSON 可序列化对象"""
    if hasattr(fig, 'to_json'):
        return json.loads(fig.to_json())
    return None


# ============================================================
# API: 健康检查
# ============================================================
@svc.api(route="/api/health_check", input=JSON(), output=JSON())
async def health_check(payload: dict = None):
    from data_service import get_health
    return get_health()


# ============================================================
# API: 获取实时气象
# ============================================================
@svc.api(route="/api/weather", input=JSON(), output=JSON())
async def weather(payload: dict = None):
    from weather_service import fetch_weather_data
    data = fetch_weather_data()
    if data is None:
        return {"error": "气象数据获取失败"}
    return data


# ============================================================
# API: 执行预测
# ============================================================
@svc.api(route="/api/predict", input=JSON(), output=JSON())
async def predict(payload: dict):
    n_steps = payload.get("n_steps", DEFAULT_PREDICTION_STEPS)
    current_price = payload.get("current_price", None)
    current_load = payload.get("current_load", None)
    import_export_price = payload.get("import_export_price", None)

    from weather_service import fetch_weather_data
    weather_data = fetch_weather_data() or {}
    weather_forcast = {
        "radiation": float(weather_data.get("current_radiation", 400)),
        "cloud": float(weather_data.get("current_cloudcover", 0)),
        "rain": float(weather_data.get("current_rain", 0)),
    }

    from prediction_service import run_prediction, generate_strategy
    result = run_prediction(
        n_steps=n_steps,
        weather=weather_forcast,
        current_price=current_price,
        current_load=current_load,
    )
    strategy = generate_strategy(result)

    # 价格曲线
    now = datetime.now()
    hours = list(range(now.hour, now.hour + 24))
    prices = [_get_price(h % 24) for h in hours]
    price_labels = [f"{(h % 24):02d}:00" for h in hours]

    return {
        "prediction": result,
        "strategy": strategy,
        "price_curve": {"labels": price_labels, "values": prices},
        "weather": weather_data,
    }


# ============================================================
# API: 仅光伏预测
# ============================================================
@svc.api(route="/api/solar_predict", input=JSON(), output=JSON())
async def solar_predict(payload: dict):
    n_steps = payload.get("n_steps", 96)
    from weather_service import fetch_weather_data
    w = fetch_weather_data() or {}
    weather_forcast = {"radiation": float(w.get("current_radiation", 400))}

    from prediction_service import run_solar_only
    return run_solar_only(n_steps, weather=weather_forcast)


# ============================================================
# API: 仅充电预测
# ============================================================
@svc.api(route="/api/charging_predict", input=JSON(), output=JSON())
async def charging_predict(payload: dict):
    n_steps = payload.get("n_steps", 96)
    current_price = payload.get("current_price", 0.8)
    current_load = payload.get("current_load", 20.0)

    from prediction_service import run_charging_only
    return run_charging_only(n_steps, current_price=current_price, current_load=current_load)


# ============================================================
# API: 生成 Plotly 图表 (作为 JSON 对象返回)
# ============================================================
@svc.api(route="/api/plot_prediction", input=JSON(), output=JSON())
async def plot_prediction(payload: dict):
    n_steps = payload.get("n_steps", 96)

    from weather_service import fetch_weather_data
    weather_data = fetch_weather_data() or {}
    weather_forcast = {
        "radiation": float(weather_data.get("current_radiation", 400)),
        "cloud": float(weather_data.get("current_cloudcover", 0)),
        "rain": float(weather_data.get("current_rain", 0)),
    }

    from prediction_service import run_prediction
    result = run_prediction(n_steps=n_steps, weather=weather_forcast)

    times = result["times"]
    solar = result["solar"]
    load_m = result["load_mean"]
    load_l = result["load_lower"]
    load_u = result["load_upper"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times, y=solar, mode="lines", name="光伏出力 (kW)",
        line=dict(color="#F59E0B", width=2),
        fill="tozeroy", fillcolor="rgba(245,158,11,0.15)",
    ))
    fig.add_trace(go.Scatter(
        x=times, y=load_m, mode="lines", name="充电负荷 (kW)",
        line=dict(color="#3B82F6", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=times + times[::-1],
        y=load_u.tolist() + load_l.tolist()[::-1],
        fill="toself", fillcolor="rgba(59,130,246,0.1)",
        line=dict(color="rgba(59,130,246,0)"),
        hoverinfo="skip",
        name="负荷置信区间",
    ))

    fig.update_layout(
        title="光伏出力 vs 充电负荷 预测",
        xaxis_title="时间",
        yaxis_title="功率 (kW)",
        hovermode="x unified",
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=60, b=40),
        height=400,
    )
    return _make_plotly_json(fig)


# ============================================================
# API: 生成天气图表
# ============================================================
@svc.api(route="/api/plot_weather", input=JSON(), output=JSON())
async def plot_weather(payload: dict = None):
    from weather_service import fetch_weather_data
    data = fetch_weather_data()
    if data is None:
        return {"error": "无气象数据"}

    dt = data.get("day_times", [])
    rads = data.get("day_radiations", [])
    clouds = data.get("day_cloudcovers", [])

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=dt, y=rads, name="辐射 (W/m²)",
        marker_color="#F59E0B", yaxis="y",
    ))
    fig.add_trace(go.Scatter(
        x=dt, y=clouds, name="云量 (%)",
        line=dict(color="#94A3B8", width=2), yaxis="y2",
    ))

    fig.update_layout(
        title="今日辐射 / 云量趋势",
        xaxis=dict(title="时间"),
        yaxis=dict(title="辐射 (W/m²)", side="left"),
        yaxis2=dict(title="云量 (%)", side="right", overlaying="y", range=[0, 100]),
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=50, r=50, t=60, b=40),
        height=350,
    )
    return _make_plotly_json(fig)


# ============================================================
# API: 电价曲线
# ============================================================
@svc.api(route="/api/price_curve", input=JSON(), output=JSON())
async def price_curve(payload: dict = None):
    from data_service import get_price_curve
    return get_price_curve()


# ============================================================
# API: 历史数据
# ============================================================
@svc.api(route="/api/historical", input=JSON(), output=JSON())
async def historical(payload: dict = None):
    from data_service import get_historical_summary, get_charging_history, get_price_curve
    solar = get_historical_summary()
    charging = get_charging_history()
    price = get_price_curve()
    return {
        "solar_history": solar,
        "charging_history": charging,
        "price_curve": price,
    }


# ============================================================
# 模型状态
# ============================================================
@svc.api(route="/api/model_status", input=JSON(), output=JSON())
async def model_status(payload: dict = None):
    from prediction_service import load_all_models
    status = load_all_models()
    return status


# ============================================================
# 静态文件服务 (前端 HTML/JS/CSS)
# 使用 Starlette ASGI 应用挂载，确保正确 MIME 类型
# ============================================================
_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

from starlette.applications import Starlette
from starlette.staticfiles import StaticFiles
from starlette.responses import FileResponse

static_app = Starlette()

# 挂载 /static 目录
static_app.mount(
    "/static",
    StaticFiles(directory=_STATIC_DIR, html=True),
    name="static",
)

# 根路径返回 index.html
@static_app.route("/")
async def homepage(request):
    index_path = os.path.join(_STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path, media_type="text/html")
    from starlette.responses import PlainTextResponse
    return PlainTextResponse("index.html not found", status_code=404)

# 挂载到 BentoML 服务
svc.mount_asgi_app(static_app, path="/")


# ============================================================
# BentoML 启动配置 (通过 bentofile.yaml 或者命令行)
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print(f"  {SERVICE_TITLE}")
    print(f"  服务地址: http://0.0.0.0:{SERVICE_PORT}")
    print("=" * 60)
    # 启动方式: bentoml serve service:svc --port 7863