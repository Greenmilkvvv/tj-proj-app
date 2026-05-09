"""
生成示例模拟数据 (2026-03-01 ~ 当前日期)
提供充电负荷和光伏出力两组模拟数据，可直接注入 upload_service 缓存。
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def _get_default_end():
    """默认结束日期为今天"""
    return datetime.now().strftime("%Y-%m-%d")


def generate_charging_sample(start: str = "2026-03-01", end: str = None) -> pd.DataFrame:
    """
    生成模拟充电负荷数据。

    参数
    ----
    start : str  起始日期 (包含)
    end   : str  结束日期 (包含), 默认为今天

    返回
    ----
    pd.DataFrame  列: timestamp, load_kw
    """
    if end is None:
        end = _get_default_end()

    freq = "15min"
    ts_range = pd.date_range(start=start, end=end + " 23:45:00", freq=freq)

    # ---- 日负荷模式 (kW) ----
    # 每小时基准负荷，反映典型充电站日规律
    hourly_base = {
        0: 300,  1: 250,  2: 200,  3: 180,  4: 170,  5: 160,
        6: 200,  7: 400,  8: 700,  9: 850,  10: 900, 11: 880,
        12: 820, 13: 780, 14: 800, 15: 830, 16: 880, 17: 950,
        18: 1000, 19: 920, 20: 780, 21: 650, 22: 500, 23: 400,
    }

    # 季节系数：3月=0.85, 4月=0.92, 5月=1.0
    def seasonal(hour_ts):
        m = hour_ts.month
        if m == 3:
            return 0.85
        elif m == 4:
            return 0.92
        else:
            return 1.0

    # 工作日系数：工作日=1.0, 周末=0.65
    def weekday_factor(hour_ts):
        if hour_ts.weekday() >= 5:
            return 0.65
        return 1.0

    loads = []
    rng = np.random.RandomState(42)
    for ts in ts_range:
        base = hourly_base.get(ts.hour, 500)
        # 15分钟内微调
        minute_adjust = 1.0 + 0.05 * np.sin(2 * np.pi * ts.minute / 60)
        seasonal_f = seasonal(ts)
        wd_f = weekday_factor(ts)
        noise = rng.normal(0, 40)
        load = base * minute_adjust * seasonal_f * wd_f + noise
        load = max(load, 50)  # 下限
        loads.append(round(load, 2))

    df = pd.DataFrame({
        "timestamp": ts_range,
        "load_kw": loads,
    })
    return df


def generate_solar_sample(start: str = "2026-03-01", end: str = None) -> pd.DataFrame:
    """
    生成模拟光伏数据。

    参数
    ----
    start : str  起始日期 (包含)
    end   : str  结束日期 (包含), 默认为今天

    返回
    ----
    pd.DataFrame  列: datetime, power, shortwave_radiation (W/m2)
    """
    if end is None:
        end = _get_default_end()

    freq = "15min"
    ts_range = pd.date_range(start=start, end=end + " 23:45:00", freq=freq)

    # 季节系数 (3月 → 5月辐照度逐渐增强)
    def seasonal_radiation(ts):
        m = ts.month
        # 3月: 0.65, 4月: 0.82, 5月: 1.0
        if m == 3:
            return 0.65
        elif m == 4:
            return 0.82
        else:
            return 1.0

    powers = []
    radiations = []
    rng = np.random.RandomState(42)
    for ts in ts_range:
        hour = ts.hour + ts.minute / 60.0

        # 光伏仅在 6:00 ~ 18:00 有出力
        if 6 <= hour <= 18:
            # 中午峰值曲线
            noon_offset = abs(hour - 12.0) / 6.0  # 0 (中午) → 1 (6点/18点)
            shape = max(0, 1 - noon_offset ** 1.5)  # 钟形曲线
            peak_power = 500  # kW 峰值
            radiation_peak = 800  # W/m² 峰值

            # 模拟云量波动
            cloud = rng.uniform(0, 0.5)  # 0~0.5 随机云量衰减
            seasonal_f = seasonal_radiation(ts)
            noise = rng.normal(0, 10)

            power = peak_power * shape * seasonal_f * (1 - cloud) + noise
            power = max(power, 0)
            radiation = radiation_peak * shape * seasonal_f * (1 - cloud) + noise * 0.5
            radiation = max(radiation, 0)
        else:
            power = 0.0
            radiation = 0.0

        powers.append(round(power, 2))
        radiations.append(round(radiation, 2))

    df = pd.DataFrame({
        "datetime": ts_range,
        "power": powers,
        "shortwave_radiation (W/m2)": radiations,
    })

    # 添加其他辐射列（填充 0，满足 upload_service 验证）
    df["direct_radiation (W/m2)"] = (np.array(radiations) * 0.7).round(2)
    df["diffuse_radiation (W/m2)"] = (np.array(radiations) * 0.3).round(2)
    df["direct_normal_irradiance (W/m2)"] = (np.array(radiations) * 0.65).round(2)

    return df


def inject_sample_data():
    """
    生成并注入示例数据到 upload_service 缓存。

    返回
    ----
    tuple  (charging_df, solar_df)
    """
    end_date = _get_default_end()

    print(f"[示例数据] 生成 2026-03-01 ~ {end_date} 充电数据...")
    charging_df = generate_charging_sample(end=end_date)
    print(f"  → {len(charging_df)} 行")

    print(f"[示例数据] 生成 2026-03-01 ~ {end_date} 光伏数据...")
    solar_df = generate_solar_sample(end=end_date)
    print(f"  → {len(solar_df)} 行")

    return charging_df, solar_df


# ====== 直接运行测试 ======
if __name__ == "__main__":
    c, s = inject_sample_data()
    print("\n充电数据预览:")
    print(c.head(10))
    print(f"\n光伏数据预览:")
    print(s.head(10))
    print(f"\n充电: {c['timestamp'].min()} ~ {c['timestamp'].max()}")
    print(f"光伏: {s['datetime'].min()} ~ {s['datetime'].max()}")