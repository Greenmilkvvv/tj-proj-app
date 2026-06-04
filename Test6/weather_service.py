"""
气象服务模块 — BentoML 版
提供实时气象数据获取 + 预报 + 预警判断
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import warnings
warnings.filterwarnings("ignore")

from config import LATITUDE, LONGITUDE, WEATHER_API_URL, WEATHER_TIMEOUT, WEATHER_CACHE_TTL

_cache = {"data": None, "ts": 0}

def fetch_weather_data(lat=LATITUDE, lon=LONGITUDE):
    """从 Open-Meteo 获取实时 + 15min 预报数据 (带缓存)"""
    now_ts = datetime.now().timestamp()
    if _cache["data"] is not None and (now_ts - _cache["ts"]) < WEATHER_CACHE_TTL:
        return _cache["data"]

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
        times = pd.to_datetime(js["minutely_15"]["time"])
        radiations = js["minutely_15"]["shortwave_radiation"]
        cloudcovers = js["minutely_15"]["cloudcover"]
        rains = js["minutely_15"]["rain"]

        now = pd.Timestamp.now(tz="Asia/Shanghai")
        now_time = now.floor("15min")
        times = pd.Series(times)
        try:
            times = times.dt.tz_localize("Asia/Shanghai")
        except Exception:
            pass

        idx = (times - now_time).abs().argmin()

        current_radiation = radiations[idx] if radiations[idx] is not None else 0
        current_cloudcover = cloudcovers[idx] if cloudcovers[idx] is not None else 0
        current_rain = rains[idx] if rains[idx] is not None else 0
        current_temp = current.get("temperature_2m", "—")

        end_idx = min(idx + 12, len(times))
        forecast_list = []
        for i in range(idx, end_idx):
            rt = times[i].strftime("%H:%M")
            rv = radiations[i] if radiations[i] is not None else 0
            cv = cloudcovers[i] if cloudcovers[i] is not None else 0
            rn = rains[i] if rains[i] is not None else 0
            if rn > 1: desc = "大雨"
            elif rn > 0: desc = "小雨"
            elif cv > 80: desc = "阴天"
            elif cv > 40: desc = "多云"
            else: desc = "晴天"
            forecast_list.append({"time": rt, "radiation": rv, "cloud": cv, "rain": rn, "weather": desc})

        has_warning = any(r > 5 for r in forecasts["rain"] for forecasts in forecast_list)

        day_idx = max(0, idx - (now.hour * 4 + now.minute // 15))
        day_end = min(day_idx + 96, len(times))
        day_times = [times[i].strftime("%H:%M") for i in range(day_idx, day_end)]
        day_rads = [radiations[i] if radiations[i] is not None else 0 for i in range(day_idx, day_end)]
        day_clouds = [cloudcovers[i] if cloudcovers[i] is not None else 0 for i in range(day_idx, day_end)]

        data = {
            "current_time": current.get("time", str(now)),
            "current_temp": current_temp if isinstance(current_temp, str) else round(float(current_temp), 1),
            "current_radiation": current_radiation,
            "current_cloudcover": current_cloudcover,
            "current_rain": current_rain,
            "forecast": forecast_list,
            "has_warning": has_warning,
            "day_times": day_times,
            "day_radiations": day_rads,
            "day_cloudcovers": day_clouds,
        }
        _cache["data"] = data
        _cache["ts"] = now_ts
        return data
    except Exception as e:
        print(f"[WARNING] 气象数据解析失败: {e}")
        return None


def get_weather():
    """别名"""
    return fetch_weather_data()