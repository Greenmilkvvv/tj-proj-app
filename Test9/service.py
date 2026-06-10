"""
BentoML 服务入口 — 纯 HTML+CSS+JS 前端 (5-Tab)
==========================================
基于 Test5 的 5-Tab 界面设计，通过 BentoML 同时提供:
  - 静态文件 (HTML/CSS/JS)
  - REST API 端点 (预测、气象、数据探索、误差分析)

路由设计:
  /            -> Starlette StaticFiles (index.html, css/, js/)
  /api/*       -> BentoML 原生 API 端点

前端使用 Plotly JS 图表库。
"""
from __future__ import annotations

import os
import sys
import json
import traceback
from datetime import datetime
from typing import Optional

import bentoml

# 让子模块可 import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DEFAULT_PREDICTION_STEPS, PREDICTION_OPTIONS

# ============================================================
# Fallback 错误图表
# ============================================================
def _error_chart(message: str):
    """生成一个包含错误信息的 fallback Plotly 图表 JSON"""
    import plotly.graph_objects as go
    from plotly.utils import PlotlyJSONEncoder
    fig = go.Figure()
    fig.add_annotation(
        x=0.5, y=0.5, text=f"⚠️ {message}",
        showarrow=False, font=dict(size=14, color="#ef4444")
    )
    fig.update_layout(
        height=300, template="plotly_white",
        margin=dict(l=40, r=40, t=20, b=40),
    )
    return json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))


# ============================================================
# 静态文件目录
# ============================================================
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


# ============================================================
# BentoML 服务定义 (所有 API 端点使用 /api/ 前缀)
# ============================================================
@bentoml.service(name="solar_charging_demo_v5")
class SolarChargingDemo:
    """光储充智能预测服务 — 纯 HTML 5-Tab UI"""

    # ============================================================
    # 生命周期
    # ============================================================
    @bentoml.on_startup
    async def startup(self):
        print("[BentoML] 预加载气象数据...")
        try:
            from weather_service import get_weather
            get_weather()
            print("[BentoML] 气象数据加载完成")
        except Exception as e:
            print(f"[BentoML] 气象数据加载失败 (非致命): {e}")

    # ============================================================
    # API: 气象数据
    # ============================================================
    @bentoml.api(route="/api/api_weather")
    async def get_api_weather(self, _body: str = "") -> dict:
        try:
            from weather_service import fetch_weather_data, get_current_weather_summary, build_radiation_chart
            from plotly.utils import PlotlyJSONEncoder
            wd = fetch_weather_data()
            summary_html = get_current_weather_summary(wd)

            # 手动构建 HTML 表格，精确控制列宽比例
            forecast_html = ""
            if wd and "forecast_df" in wd:
                df = wd["forecast_df"]
                cols = list(df.columns)
                rows = [[str(cell) for cell in row] for row in df.values.tolist()]
                # colgroup: 时间8% | 辐照度12% | 云量10% | 降雨量14% | 天气56%
                col_widths = ["8%", "12%", "10%", "14%", "56%"]
                col_html = "".join(f'<col style="width:{w}">' for w in col_widths)
                thead_html = "<tr>" + "".join(f"<th>{c}</th>" for c in cols) + "</tr>"
                tbody_html = "".join("<tr>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>" for row in rows)
                forecast_html = (
                    f'<table id="forecast-table" class="weather-table">'
                    f'<colgroup>{col_html}</colgroup>'
                    f'<thead>{thead_html}</thead>'
                    f'<tbody>{tbody_html}</tbody>'
                    f'</table>'
                )

            # 辐照度趋势图
            rad_chart = build_radiation_chart(wd) if wd else None
            rad_chart_json = json.loads(json.dumps(rad_chart, cls=PlotlyJSONEncoder)) if rad_chart else None

            return {
                "success": True,
                "weather": {
                    "current_time": wd["current_time"] if wd else "",
                    "current_temp": wd["current_temp"] if wd else "--",
                    "current_radiation": float(wd["current_radiation"]) if wd else 0,
                    "current_cloudcover": int(wd["current_cloudcover"]) if wd else 0,
                    "current_rain": float(wd["current_rain"]) if wd else 0,
                    "has_warning": wd["has_warning"] if wd else False,
                    "warning_msg": wd["warning_msg"] if wd else "",
                    "forecast_html": forecast_html,
                },
                "summary_html": summary_html,
                "chart": rad_chart_json,
            }
        except Exception as e:
            traceback.print_exc()
            return {"success": False, "error": str(e)}

    # ============================================================
    # API: 预测
    # ============================================================
    @bentoml.api(route="/api/api_predict")
    async def get_api_predict(
        self,
        model: str = "combined",
        n_steps: int = DEFAULT_PREDICTION_STEPS,
        price: float = 0.63,
        load: float = 875.0,
    ) -> dict:
        try:
            from prediction_service import run_prediction
            weather = {"radiation": 400}
            result = run_prediction(n_steps, weather, current_price=price, current_load=load)

            times_iso = [t.isoformat() if hasattr(t, "isoformat") else str(t) for t in result["times"]]
            times_str = [t.strftime("%H:%M") if hasattr(t, "strftime") else str(t)[-8:-3] for t in result["times"]]

            solar_list = [float(v) for v in result["solar"]]
            load_mean_list = [float(v) for v in result["load_mean"]]
            load_lower_list = [float(v) for v in result["load_lower"]]
            load_upper_list = [float(v) for v in result["load_upper"]]

            from prediction_service import generate_strategy
            strategy = generate_strategy(result)

            return {
                "success": True,
                "times": times_str,
                "times_iso": times_iso,
                "solar": solar_list,
                "load_mean": load_mean_list,
                "load_lower": load_lower_list,
                "load_upper": load_upper_list,
                "total_solar": float(result["total_solar"]),
                "total_load": float(result["total_load"]),
                "green_ratio": float(result["green_ratio"]),
                "solar_peak": float(result["solar_peak"]),
                "solar_peak_time": result["solar_peak_time"].strftime("%H:%M") if hasattr(result["solar_peak_time"], "strftime") else str(result["solar_peak_time"]),
                "load_peak": float(result["load_peak"]),
                "model_status": result.get("model_status", {"solar_ok": False, "charging_ok": False}),
                "strategy": strategy,
            }
        except Exception as e:
            traceback.print_exc()
            return {"success": False, "error": str(e)}

    # ============================================================
    # API: 数据探索
    # ============================================================
    @bentoml.api(route="/api/api_data_overview")
    async def get_api_data_overview(self, _body: str = "") -> dict:
        try:
            from data_service import get_dataset_overview, get_date_range
            overview_html = get_dataset_overview()
            date_range = get_date_range()
            return {"success": True, "overview_html": overview_html, "date_range": date_range}
        except Exception as e:
            traceback.print_exc()
            return {"success": False, "error": str(e), "chart": _error_chart(str(e))}

    @bentoml.api(route="/api/api_single_day_load")
    async def get_api_single_day_load(self, date: str = "") -> dict:
        try:
            from data_service import plot_single_day_load
            fig = plot_single_day_load(date)
            from plotly.utils import PlotlyJSONEncoder
            return {"success": True, "chart": json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))}
        except Exception as e:
            traceback.print_exc()
            return {"success": False, "error": str(e), "chart": _error_chart(str(e))}

    @bentoml.api(route="/api/api_aggregated_load")
    async def get_api_aggregated_load(self, date_start: str = "", date_end: str = "") -> dict:
        try:
            from data_service import plot_aggregated_load
            fig = plot_aggregated_load(date_start, date_end)
            from plotly.utils import PlotlyJSONEncoder
            return {"success": True, "chart": json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))}
        except Exception as e:
            traceback.print_exc()
            return {"success": False, "error": str(e), "chart": _error_chart(str(e))}

    @bentoml.api(route="/api/api_daily_load")
    async def get_api_daily_load(self, date_start: str = "", date_end: str = "") -> dict:
        try:
            from data_service import plot_daily_load_curves
            fig = plot_daily_load_curves(date_start, date_end)
            from plotly.utils import PlotlyJSONEncoder
            return {"success": True, "chart": json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))}
        except Exception as e:
            traceback.print_exc()
            return {"success": False, "error": str(e), "chart": _error_chart(str(e))}

    @bentoml.api(route="/api/api_correlation")
    async def get_api_correlation(self, _body: str = "") -> dict:
        try:
            from data_service import get_correlation_chart
            fig = get_correlation_chart()
            from plotly.utils import PlotlyJSONEncoder
            return {"success": True, "chart": json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))}
        except Exception as e:
            traceback.print_exc()
            return {"success": False, "error": str(e), "chart": _error_chart(str(e))}

    @bentoml.api(route="/api/api_hourly_profile")
    async def get_api_hourly_profile(self, _body: str = "") -> dict:
        try:
            from data_service import get_hourly_profile_chart
            fig = get_hourly_profile_chart()
            from plotly.utils import PlotlyJSONEncoder
            return {"success": True, "chart": json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))}
        except Exception as e:
            traceback.print_exc()
            return {"success": False, "error": str(e), "chart": _error_chart(str(e))}

    # ============================================================
    # API: 误差分析
    # ============================================================
    @bentoml.api(route="/api/api_backtest_charging")
    async def get_api_backtest_charging(self, _body: str = "") -> dict:
        try:
            from data_service import run_backtest_charging
            fig, summary = run_backtest_charging()
            from plotly.utils import PlotlyJSONEncoder
            return {"success": True, "chart": json.loads(json.dumps(fig, cls=PlotlyJSONEncoder)), "summary": summary}
        except Exception as e:
            traceback.print_exc()
            return {"success": False, "error": str(e), "chart": _error_chart(str(e)), "summary": f"⚠️ 回测异常: {e}"}

    @bentoml.api(route="/api/api_backtest_solar")
    async def get_api_backtest_solar(self, _body: str = "") -> dict:
        try:
            from data_service import run_backtest_solar
            fig, summary = run_backtest_solar()
            from plotly.utils import PlotlyJSONEncoder
            return {"success": True, "chart": json.loads(json.dumps(fig, cls=PlotlyJSONEncoder)), "summary": summary}
        except Exception as e:
            traceback.print_exc()
            return {"success": False, "error": str(e), "chart": _error_chart(str(e)), "summary": f"⚠️ 回测异常: {e}"}

    @bentoml.api(route="/api/api_charging_error_dist")
    async def get_api_charging_error_dist(self, _body: str = "") -> dict:
        try:
            from data_service import build_charging_error_distribution_chart
            fig = build_charging_error_distribution_chart()
            from plotly.utils import PlotlyJSONEncoder
            return {"success": True, "chart": json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))}
        except Exception as e:
            traceback.print_exc()
            return {"success": False, "error": str(e), "chart": _error_chart(str(e))}

    @bentoml.api(route="/api/api_charging_error_hourly")
    async def get_api_charging_error_hourly(self, _body: str = "") -> dict:
        try:
            from data_service import build_charging_error_by_hour_chart
            fig = build_charging_error_by_hour_chart()
            from plotly.utils import PlotlyJSONEncoder
            return {"success": True, "chart": json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))}
        except Exception as e:
            traceback.print_exc()
            return {"success": False, "error": str(e), "chart": _error_chart(str(e))}

    @bentoml.api(route="/api/api_solar_error_dist")
    async def get_api_solar_error_dist(self, _body: str = "") -> dict:
        try:
            from data_service import build_solar_error_distribution_chart
            fig = build_solar_error_distribution_chart()
            from plotly.utils import PlotlyJSONEncoder
            return {"success": True, "chart": json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))}
        except Exception as e:
            traceback.print_exc()
            return {"success": False, "error": str(e), "chart": _error_chart(str(e))}

    @bentoml.api(route="/api/api_solar_error_hourly")
    async def get_api_solar_error_hourly(self, _body: str = "") -> dict:
        try:
            from data_service import build_solar_error_by_hour_chart
            fig = build_solar_error_by_hour_chart()
            from plotly.utils import PlotlyJSONEncoder
            return {"success": True, "chart": json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))}
        except Exception as e:
            traceback.print_exc()
            return {"success": False, "error": str(e), "chart": _error_chart(str(e))}

    # ============================================================
    # 挂载静态文件 ASGI 应用 (路径: /)
    # ============================================================
try:
    from starlette.staticfiles import StaticFiles
    from starlette.applications import Starlette
    from starlette.responses import FileResponse

    static_app = Starlette()

    static_app.mount(
        "/css",
        StaticFiles(directory=os.path.join(STATIC_DIR, "css"), html=False),
        name="css",
    )
    static_app.mount(
        "/js",
        StaticFiles(directory=os.path.join(STATIC_DIR, "js"), html=False),
        name="js",
    )

    @static_app.route("/")
    async def homepage(request):
        index_path = os.path.join(STATIC_DIR, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path, media_type="text/html")
        from starlette.responses import PlainTextResponse
        return PlainTextResponse("index.html not found", status_code=404)

    SolarChargingDemo.mount_asgi_app(static_app, path="/")
    print(f"[BentoML] 静态文件已挂载: {STATIC_DIR}")
    print(f"[BentoML] 路由: / -> StaticFiles, /api/* -> BentoML APIs")
except ImportError as e:
    print(f"[BentoML] 静态文件挂载失败: {e}")

# BentoML 服务入口
svc = SolarChargingDemo