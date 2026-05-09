"""
上传数据管理服务
提供 CSV 文件上传验证、lag 特征自动计算、数据合并等功能
"""
import os
import pandas as pd
import numpy as np
import shutil
from datetime import datetime

from config import UPLOAD_DIR, BASE_DIR

# ========== 全局上传缓存 ==========
_upload_cache = {}  # {"charging": DataFrame, "solar": DataFrame}

# ========== 充电数据列定义 ==========
CHARGING_REQUIRED_COLS = ["timestamp", "load_kw"]
CHARGING_ALL_COLS = ["timestamp", "price", "lag_1", "lag_96", "lag_672",
                     "rolling_std_4", "rolling_mean_4", "load_kw"]

# ========== 光伏数据列定义 ==========
SOLAR_REQUIRED_COLS = ["datetime", "power"]
SOLAR_RADIATION_COLS = [
    "shortwave_radiation (W/m2)", "direct_radiation (W/m2)",
    "diffuse_radiation (W/m2)", "direct_normal_irradiance (W/m2)",
]
SOLAR_OPTIONAL_COLS = SOLAR_RADIATION_COLS


def get_upload_dir():
    """获取上传目录路径，确保存在"""
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    return UPLOAD_DIR


def _sanitize_csv(raw_content):
    """检测并统一 CSV 编码"""
    # 尝试 UTF-8, 回退 GBK
    try:
        return raw_content.decode("utf-8")
    except UnicodeDecodeError:
        try:
            return raw_content.decode("gbk")
        except Exception:
            return raw_content.decode("utf-8", errors="replace")


def _compute_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """为充电数据自动计算缺失的 lag 特征"""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    load = df["load_kw"].values

    # lag_1: 上一行 (15min前)
    if "lag_1" not in df.columns:
        l1 = np.roll(load, 1)
        l1[0] = load[0]  # 第一行用自身填充
        df["lag_1"] = l1

    # lag_96: 96步前 (24小时前)
    if "lag_96" not in df.columns:
        l96 = np.roll(load, 96)
        l96[:96] = load[:96]
        df["lag_96"] = l96

    # lag_672: 672步前 (7天前)
    if "lag_672" not in df.columns:
        l672 = np.roll(load, 672)
        l672[:672] = load[:672]
        df["lag_672"] = l672

    # rolling_std_4 / rolling_mean_4: 4步滚动
    if "rolling_std_4" not in df.columns or "rolling_mean_4" not in df.columns:
        rstd = np.zeros_like(load)
        rmean = np.zeros_like(load)
        for i in range(len(load)):
            start = max(0, i - 3)
            window = load[start:i + 1]
            rstd[i] = float(np.std(window))
            rmean[i] = float(np.mean(window))
        if "rolling_std_4" not in df.columns:
            df["rolling_std_4"] = rstd
        if "rolling_mean_4" not in df.columns:
            df["rolling_mean_4"] = rmean

    # price: 如果缺失，按峰谷时段填充
    if "price" not in df.columns:
        def _price(h):
            if 8 <= h < 11 or 18 <= h < 21:
                return 1.0
            elif 6 <= h < 8 or 11 <= h < 18 or 21 <= h < 22:
                return 0.6
            else:
                return 0.3
        df["price"] = df["timestamp"].dt.hour.apply(_price)

    return df


def upload_charging(file_obj):
    """上传充电 CSV，返回 (成功标志, 信息摘要)"""
    upload_dir = get_upload_dir()

    try:
        raw = file_obj.read()
        content = _sanitize_csv(raw)
    except Exception as e:
        return False, f"读取文件失败: {e}"

    from io import StringIO
    try:
        df = pd.read_csv(StringIO(content))
    except Exception as e:
        return False, f"CSV 解析失败: {e}"

    # 检查必要列
    for col in CHARGING_REQUIRED_COLS:
        if col not in df.columns:
            return False, f"缺少必要列: '{col}'。充电数据必须包含 {CHARGING_REQUIRED_COLS}"

    # 自动计算 lag 特征
    df = _compute_lag_features(df)

    # 只保留需要的列，去除额外列
    keep_cols = [c for c in df.columns if c not in ["timestamp", "load_kw"]
                 and c in CHARGING_ALL_COLS]  # 特征列
    # 实际保留: timestamp + 6特征 + load_kw，但允许额外列存在（如原始特征）
    # 简化：删除明显多余列
    # 不强制删除，data_service 只会用需要的列

    # 保存
    save_path = os.path.join(upload_dir, "charging_upload.csv")
    df.to_csv(save_path, index=False)

    # 缓存
    _upload_cache["charging"] = df

    # 时间范围
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        dmin = df["timestamp"].min()
        dmax = df["timestamp"].max()
        date_range = f"{dmin} ~ {dmax}"
    else:
        date_range = "—"

    info = (
        f"✅ 充电数据上传成功\n"
        f"📅 时间范围: {date_range}\n"
        f"📊 共 {len(df)} 行, {len(df.columns)} 列"
    )
    return True, info


def upload_solar(file_obj):
    """上传光伏 CSV，返回 (成功标志, 信息摘要)"""
    upload_dir = get_upload_dir()

    try:
        raw = file_obj.read()
        content = _sanitize_csv(raw)
    except Exception as e:
        return False, f"读取文件失败: {e}"

    from io import StringIO
    try:
        df = pd.read_csv(StringIO(content))
    except Exception as e:
        return False, f"CSV 解析失败: {e}"

    # 检查必要列
    for col in SOLAR_REQUIRED_COLS:
        if col not in df.columns:
            return False, f"缺少必要列: '{col}'。光伏数据必须包含 {SOLAR_REQUIRED_COLS}"

    # 检查辐射列 (至少有一列)
    has_radiation = any(c in df.columns for c in SOLAR_RADIATION_COLS)
    if not has_radiation:
        return False, f"缺少辐射列。至少需要以下之一: {SOLAR_RADIATION_COLS}"

    # 时间解析
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime")

    # 缺失辐射列用 0 填充
    for c in SOLAR_RADIATION_COLS:
        if c not in df.columns:
            df[c] = 0.0

    # 计算时间特征
    df["hour"] = df["datetime"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # 保存
    save_path = os.path.join(upload_dir, "solar_upload.csv")
    df.to_csv(save_path, index=False)

    # 缓存
    _upload_cache["solar"] = df

    dmin = df["datetime"].min()
    dmax = df["datetime"].max()

    info = (
        f"✅ 光伏数据上传成功\n"
        f"📅 时间范围: {dmin} ~ {dmax}\n"
        f"📊 共 {len(df)} 行, {len(df.columns)} 列"
    )
    return True, info


def get_charging_data():
    """获取合并后的充电数据 (历史 + 上传)"""
    from data_service import get_test_data, _load_csv
    from config import DATA_SELECTED_TEST

    if "charging" in _upload_cache and _upload_cache["charging"] is not None:
        upload_df = _upload_cache["charging"].copy()

        # 加载历史测试数据
        try:
            hist_df = _load_csv(DATA_SELECTED_TEST)
            if hist_df is not None and "timestamp" in hist_df.columns:
                hist_df["timestamp"] = pd.to_datetime(hist_df["timestamp"])
                # 确保 lag 特征存在
                upload_df["timestamp"] = pd.to_datetime(upload_df["timestamp"])
                # 合并：历史 + 上传（按时间排序去重）
                combined = pd.concat([hist_df, upload_df], ignore_index=True)
                combined["timestamp"] = pd.to_datetime(combined["timestamp"])
                combined = combined.drop_duplicates(subset=["timestamp"], keep="last")
                combined = combined.sort_values("timestamp").reset_index(drop=True)
                return combined
        except Exception as e:
            print(f"[WARN] 历史充电数据加载失败，仅使用上传数据: {e}")

        upload_df["timestamp"] = pd.to_datetime(upload_df["timestamp"])
        upload_df = upload_df.sort_values("timestamp").reset_index(drop=True)
        return upload_df

    # 回退：仅历史数据
    return _load_csv(DATA_SELECTED_TEST)


def get_solar_data():
    """获取合并后的光伏数据 (历史 + 上传)"""
    from config import DATA_ALIGNED

    aligned_path = DATA_ALIGNED

    if "solar" in _upload_cache and _upload_cache["solar"] is not None:
        upload_df = _upload_cache["solar"].copy()
        upload_df["datetime"] = pd.to_datetime(upload_df["datetime"])

        # 确保特征列存在
        if "hour" not in upload_df.columns:
            upload_df["hour"] = upload_df["datetime"].dt.hour
        if "hour_sin" not in upload_df.columns:
            upload_df["hour_sin"] = np.sin(2 * np.pi * upload_df["hour"] / 24)
        if "hour_cos" not in upload_df.columns:
            upload_df["hour_cos"] = np.cos(2 * np.pi * upload_df["hour"] / 24)

        # 加载历史数据
        if os.path.exists(aligned_path):
            try:
                hist_df = pd.read_csv(aligned_path, parse_dates=["datetime"])
                hist_df = hist_df.sort_values("datetime")
                # 确保特征列
                if "hour" not in hist_df.columns:
                    hist_df["hour"] = hist_df["datetime"].dt.hour
                if "hour_sin" not in hist_df.columns:
                    hist_df["hour_sin"] = np.sin(2 * np.pi * hist_df["hour"] / 24)
                if "hour_cos" not in hist_df.columns:
                    hist_df["hour_cos"] = np.cos(2 * np.pi * hist_df["hour"] / 24)

                combined = pd.concat([hist_df, upload_df], ignore_index=True)
                combined["datetime"] = pd.to_datetime(combined["datetime"])
                combined = combined.drop_duplicates(subset=["datetime"], keep="last")
                combined = combined.sort_values("datetime").reset_index(drop=True)
                return combined
            except Exception as e:
                print(f"[WARN] 历史光伏数据加载失败，仅使用上传数据: {e}")

        upload_df = upload_df.sort_values("datetime").reset_index(drop=True)
        return upload_df

    # 回退：仅历史数据
    if os.path.exists(aligned_path):
        df = pd.read_csv(aligned_path, parse_dates=["datetime"])
        df = df.sort_values("datetime")
        if "hour" not in df.columns:
            df["hour"] = df["datetime"].dt.hour
        if "hour_sin" not in df.columns:
            df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        if "hour_cos" not in df.columns:
            df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        return df

    return None


def has_uploaded_data(data_type="charging"):
    """检查是否已上传数据"""
    return data_type in _upload_cache and _upload_cache[data_type] is not None


def get_upload_status():
    """获取上传状态摘要 HTML"""
    lines = ["<table style='width:100%; border-collapse:collapse; font-size:13px;'>"]
    lines.append("<tr style='background:#f0f0f0;'><th style='padding:8px; text-align:left;'>数据类型</th>"
                  "<th style='padding:8px; text-align:left;'>状态</th>"
                  "<th style='padding:8px; text-align:left;'>时间范围</th>"
                  "<th style='padding:8px; text-align:right;'>行数</th></tr>")

    # 充电
    if has_uploaded_data("charging"):
        df = _upload_cache["charging"]
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        dmin, dmax = df["timestamp"].min(), df["timestamp"].max()
        lines.append(f"<tr><td>⚡ 充电数据</td><td style='color:#28a745;'>✅ 已上传</td>"
                     f"<td>{dmin} ~ {dmax}</td><td style='text-align:right;'>{len(df)}</td></tr>")
    else:
        lines.append(f"<tr><td>⚡ 充电数据</td><td style='color:#dc3545;'>❌ 未上传</td>"
                     f"<td>—</td><td style='text-align:right;'>—</td></tr>")

    # 光伏
    if has_uploaded_data("solar"):
        df = _upload_cache["solar"]
        dmin, dmax = df["datetime"].min(), df["datetime"].max()
        lines.append(f"<tr><td>☀️ 光伏数据</td><td style='color:#28a745;'>✅ 已上传</td>"
                     f"<td>{dmin} ~ {dmax}</td><td style='text-align:right;'>{len(df)}</td></tr>")
    else:
        lines.append(f"<tr><td>☀️ 光伏数据</td><td style='color:#dc3545;'>❌ 未上传</td>"
                     f"<td>—</td><td style='text-align:right;'>—</td></tr>")

    lines.append("</table>")

    # 判断是否两数据都已上传
    if has_uploaded_data("charging") and has_uploaded_data("solar"):
        lines.append('<div class="success-box">🎉 两项数据均已上传，可在「误差分析」Tab 中基于真实数据评估模型</div>')
    elif has_uploaded_data("charging") or has_uploaded_data("solar"):
        lines.append('<div class="info-box">ℹ️ 建议同时上传充电和光伏数据以获得完整评估</div>')

    return "\n".join(lines)


# ============================================================
# 适配函数 — 匹配 app.py 的导入接口
# ============================================================

def save_uploaded_charging(filepath: str):
    """适配 app.py 的上传接口: 接受文件路径，返回 (消息, 成功标志)"""
    if not filepath or not os.path.exists(filepath):
        return "文件不存在", False
    try:
        with open(filepath, "rb") as f:
            raw = f.read()
        from io import BytesIO
        file_obj = BytesIO(raw)
        ok, msg = upload_charging(file_obj)
        return msg, ok
    except Exception as e:
        return f"上传失败: {str(e)}", False


def save_uploaded_solar(filepath: str):
    """适配 app.py 的上传接口: 接受文件路径，返回 (消息, 成功标志)"""
    if not filepath or not os.path.exists(filepath):
        return "文件不存在", False
    try:
        with open(filepath, "rb") as f:
            raw = f.read()
        from io import BytesIO
        file_obj = BytesIO(raw)
        ok, msg = upload_solar(file_obj)
        return msg, ok
    except Exception as e:
        return f"上传失败: {str(e)}", False


def clear_uploaded_data():
    """清除所有上传数据缓存，返回状态消息"""
    global _upload_cache
    _upload_cache = {}
    return "已清除所有上传数据，恢复使用内置数据集"
