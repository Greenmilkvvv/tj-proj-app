"""
数据服务模块 — BentoML 版
图表数据提取 + 历史曲线等
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

from config import DATA_ALIGNED

_ALIGNED_DF_CACHE = None


def _get_aligned_df():
    global _ALIGNED_DF_CACHE
    if _ALIGNED_DF_CACHE is not None:
        return _ALIGNED_DF_CACHE
    try:
        df = pd.read_csv(DATA_ALIGNED, parse_dates=["datetime"])
        df.dropna(inplace=True)
        df.sort_values("datetime", inplace=True)
        _ALIGNED_DF_CACHE = df
        return df
    except Exception:
        return None


def get_historical_summary():
    """返回最近24h的历史光伏+负荷摘要"""
    df = _get_aligned_df()
    if df is None:
        return None
    end = df["datetime"].max()
    start = end - timedelta(hours=24)
    recent = df[(df["datetime"] >= start) & (df["datetime"] <= end)].copy()
    if recent.empty:
        recent = df.tail(96).copy()

    solar = recent["power"].values if "power" in recent.columns else np.zeros(len(recent))
    times = recent["datetime"].dt.strftime("%H:%M").tolist()
    return {
        "times": times,
        "solar": [round(float(v), 2) for v in solar],
    }


def get_charging_history():
    """返回充电负荷历史"""
    from config import DATA_SELECTED_TEST
    try:
        df = pd.read_csv(DATA_SELECTED_TEST, parse_dates=["timestamp"])
        df.dropna(inplace=True)
        df.sort_values("timestamp", inplace=True)
        last_96 = df.tail(96)
        times = last_96["timestamp"].dt.strftime("%Y-%m-%d %H:%M").tolist()
        load = last_96["load_kw"].values.tolist()
        return {"times": times, "load": load}
    except Exception:
        return None


def get_price_curve():
    """当天电价曲线"""
    hours = list(range(24))
    prices = []
    for h in hours:
        if 8 <= h < 11 or 18 <= h < 21:
            prices.append(0.85)
        elif 6 <= h < 8 or 11 <= h < 18 or 21 <= h < 22:
            prices.append(0.63)
        else:
            prices.append(0.25)
    return {"hours": [f"{h:02d}:00" for h in hours], "prices": prices}


def get_health():
    """健康检查"""
    return {
        "status": "ok",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "aligned_data_available": _get_aligned_df() is not None,
    }