"""
生成示例数据 — 充电负荷 + 光伏
充电负荷数据从真实训练数据 dataset_selected_features_test.csv 复制，
避免模拟数据分布偏差导致预测为平的问题。
"""
import pandas as pd
import numpy as np
import os
import sys

# ============================================================
# 路径配置
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(BASE_DIR, "sample_data")

CHARGING_SRC = os.path.join(ROOT_DIR, "Data", "dataset_selected_features_test.csv")
CHARGING_OUT = os.path.join(DATA_DIR, "charging_load_sample.csv")
SOLAR_OUT = os.path.join(DATA_DIR, "solar_pv_sample.csv")


def generate_solar_data(days: int = 60) -> pd.DataFrame:
    """
    生成光伏模拟数据。
    日照遵循日周期正弦波（6:00–18:00 有日照），叠加随机噪声。
    列名与 upload_service.py 的 SOLAR_REQUIRED_COLS / SOLAR_RADIATION_COLS 一致。
    """
    steps = days * 96  # 15 分钟步长
    datetimes = pd.date_range(
        start="2026-03-01 00:00:00",
        periods=steps,
        freq="15min",
        tz=None,
    )

    np.random.seed(42)
    hour_of_day = datetimes.hour + datetimes.minute / 60.0

    # 日照强度：6–18 点为半正弦波，夜间为 0
    sunrise, sunset = 6, 18
    sun_mask = (hour_of_day >= sunrise) & (hour_of_day <= sunset)
    intensity = np.zeros(steps)
    intensity[sun_mask] = (
        np.sin(np.pi * (hour_of_day[sun_mask] - sunrise) / (sunset - sunrise)) ** 0.8
    )

    # 叠加天气噪声 + 小幅趋势
    base_power = 500 * intensity  # 峰值 500 kW
    noise = np.random.normal(0, 20, steps)
    trend = 0.02 * np.arange(steps)  # 轻微上升趋势
    power = np.clip(base_power + noise + trend, 0, None)

    # 辐射列 (模拟)
    shortwave = 800 * intensity + np.random.normal(0, 30, steps)
    shortwave = np.clip(shortwave, 0, None)

    direct_radiation = 600 * intensity + np.random.normal(0, 25, steps)
    direct_radiation = np.clip(direct_radiation, 0, None)

    diffuse_radiation = 200 * intensity + np.random.normal(0, 15, steps)
    diffuse_radiation = np.clip(diffuse_radiation, 0, None)

    dni = 700 * intensity + np.random.normal(0, 20, steps)
    dni = np.clip(dni, 0, None)

    df = pd.DataFrame({
        "datetime": datetimes,
        "power": power.round(2),
        "shortwave_radiation (W/m2)": shortwave.round(1),
        "direct_radiation (W/m2)": direct_radiation.round(1),
        "diffuse_radiation (W/m2)": diffuse_radiation.round(1),
        "direct_normal_irradiance (W/m2)": dni.round(1),
    })
    return df


def generate_charging_data_from_real(total_days: int = 60) -> pd.DataFrame:
    """
    从真实训练数据中提取逐日负荷模板，循环复制生成充电负荷数据。
    保留真实 load_kw 的日周期模式（900~2400 kW），避免预测为平。
    """
    # ---- 1. 读取训练数据 ----
    if not os.path.exists(CHARGING_SRC):
        print(f"[ERROR] 训练数据不存在: {CHARGING_SRC}")
        sys.exit(1)

    df_src = pd.read_csv(CHARGING_SRC, parse_dates=["timestamp"])

    # ---- 2. 抽取日模板 ----
    # 直接按日期分组（不筛选 price，用全量数据保留日负荷模式）
    df_src["date"] = df_src["timestamp"].dt.date
    daily_groups = df_src.groupby("date")

    templates = []
    for date, group in daily_groups:
        if len(group) >= 80:  # 至少 80 步才视为完整天
            # 取 load_kw 时序，不足 96 步则用最后一个值填充
            vals = group["load_kw"].values[:96]
            if len(vals) < 96:
                pad = np.full(96 - len(vals), vals[-1])
                vals = np.concatenate([vals, pad])
            templates.append(vals)

    if not templates:
        print("[ERROR] 训练数据中没有找到完整的一天数据")
        sys.exit(1)

    print(f"[INFO] 从训练数据中提取了 {len(templates)} 个日模板")

    # ---- 3. 循环复制生成数据 ----
    steps = total_days * 96
    timestamps = pd.date_range(
        start="2026-03-01 00:00:00",
        periods=steps,
        freq="15min",
        tz=None,
    )

    load_kw = np.zeros(steps)
    np.random.seed(123)

    for i in range(total_days):
        day_start = i * 96
        day_end = day_start + 96
        template = templates[i % len(templates)]

        # 加微小噪声（标准差 ~ 负荷均值的 1%），避免完全重复
        noise_std = max(np.mean(template) * 0.01, 5.0)
        noise = np.random.normal(0, noise_std, 96)
        load_kw[day_start:day_end] = np.clip(template + noise, 0, None)

    # price 全部设为 0.25（与训练数据一致）
    price = np.full(steps, 0.25)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "price": price,
        "load_kw": load_kw.round(2),
    })
    return df


def main():
    print("=" * 60)
    print("生成示例数据...")
    print("=" * 60)

    os.makedirs(DATA_DIR, exist_ok=True)

    # ---- 光伏数据 ----
    print("\n[1/2] 生成光伏数据...")
    df_solar = generate_solar_data(days=60)
    df_solar.to_csv(SOLAR_OUT, index=False)
    print(f"  -> 保存到: {SOLAR_OUT}")
    print(f"     行数: {len(df_solar)}, 列: {list(df_solar.columns)}")

    # ---- 充电负荷数据 (从真实数据复制) ----
    print("\n[2/2] 生成充电负荷数据 (基于真实训练数据)...")
    df_charging = generate_charging_data_from_real(total_days=60)
    df_charging.to_csv(CHARGING_OUT, index=False)
    print(f"  -> 保存到: {CHARGING_OUT}")
    print(f"     行数: {len(df_charging)}, 列: {list(df_charging.columns)}")
    print(f"     load_kw 范围: {df_charging['load_kw'].min():.0f} ~ {df_charging['load_kw'].max():.0f} kW")
    print(f"     load_kw 均值: {df_charging['load_kw'].mean():.0f} kW")
    print(f"     price 唯一值: {df_charging['price'].unique().tolist()}")

    print("\n" + "=" * 60)
    print("完成！请重启 app.py 并重新上传数据。")
    print("=" * 60)


def inject_sample_data():
    """
    读取已生成的示例 CSV 文件，返回 (充电负荷DataFrame, 光伏DataFrame)。
    供 app.py 在启动时自动注入示例数据。
    """
    if not os.path.exists(CHARGING_OUT) or not os.path.exists(SOLAR_OUT):
        print("[WARN] 示例数据文件不存在，请先运行 generate_sample_data.py")
        return None, None

    df_charging = pd.read_csv(CHARGING_OUT, parse_dates=["timestamp"])
    df_solar = pd.read_csv(SOLAR_OUT, parse_dates=["datetime"])
    print(f"[INFO] 注入示例数据: 充电 {len(df_charging)} 行, 光伏 {len(df_solar)} 行")
    return df_charging, df_solar


if __name__ == "__main__":
    main()
