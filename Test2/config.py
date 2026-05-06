"""
光储充预测 Demo — 全局配置文件
所有可调参数集中在此，修改后重启应用即可生效。
"""
import os

# ============================================================
# 项目根目录
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

# ============================================================
# 模型权重路径
# ============================================================
SOLAR_MODEL_PTH = os.path.join(ROOT_DIR, "Solar_Forecast", "best_pth", "best_generator.pth")
CHARGING_MODEL_PTH = os.path.join(ROOT_DIR, "Charging_Forecast", "best_pth", "final_best_hybrid_model.pth")

# ============================================================
# 数据文件路径
# ============================================================
DATA_ALL_TRAIN = os.path.join(ROOT_DIR, "Data", "dataset_all_features_train.csv")
DATA_ALL_TEST = os.path.join(ROOT_DIR, "Data", "dataset_all_features_test.csv")
DATA_SELECTED_TRAIN = os.path.join(ROOT_DIR, "Data", "dataset_selected_features_train.csv")
DATA_SELECTED_TEST = os.path.join(ROOT_DIR, "Data", "dataset_selected_features_test.csv")
DATA_ALIGNED = os.path.join(ROOT_DIR, "Data", "aligned_2026_01_02.csv")

# ============================================================
# 气象 API 配置 (宁波坐标)
# ============================================================
LATITUDE = 29.866465
LONGITUDE = 121.52707
WEATHER_API_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER_TIMEOUT = 20  # 秒

# ============================================================
# 模型参数
# ============================================================
SOLAR_LOOKBACK = 24        # 光伏模型输入窗口 (步, 15min/步 = 6h)
CHARGING_LOOKBACK = 96     # 充电模型输入窗口 (步, 15min/步 = 24h)
MC_DROPOUT_SAMPLES = 30    # MC Dropout 采样次数
SOLAR_FEATURE_DIM = 5      # 光伏模型输入特征数
CHARGING_FEATURE_DIM = 6   # 充电模型输入特征数

# 光伏模型使用的特征列 (aligned CSV)
SOLAR_FEATURE_COLS = [
    "power",
    "hour_sin",
    "hour_cos",
    "shortwave_radiation (W/m2)",
    "direct_radiation (W/m2)",
]

# ============================================================
# 电价配置 (元/kWh)
# ============================================================
PEAK_PRICE = 1.0
MID_PRICE = 0.6
VALLEY_PRICE = 0.3

# 电价时段 (hours)
PRICE_PERIODS = {
    "peak":   [(8, 11), (18, 21)],
    "mid":    [(6, 8), (11, 18), (21, 22)],
    "valley": [(0, 6), (22, 24)],
}

# ============================================================
# 预测步数选项 (供UI下拉使用)
# ============================================================
PREDICTION_OPTIONS = {
    "1 小时": 4,
    "3 小时": 12,
    "6 小时": 24,
    "12 小时": 48,
}
DEFAULT_PREDICTION_STEPS = 24  # 默认 6h

# ============================================================
# UI 配置
# ============================================================
APP_TITLE = "光储充智能预测 Demo"
APP_PORT = 7860
APP_THEME = "soft"