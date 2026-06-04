"""
光储充预测 Demo — 全局配置文件 (Test6 BentoML 版)
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
CHARGING_MODEL_PTH = os.path.join(ROOT_DIR, "Charging_Retraining", "best_pth", "final_best_hybrid_model.pth")

# ============================================================
# 数据文件路径
# ============================================================
DATA_ALL_TRAIN = os.path.join(ROOT_DIR, "Data", "dataset_all_features_train.csv")
DATA_ALL_TEST = os.path.join(ROOT_DIR, "Data", "dataset_all_features_test.csv")
DATA_SELECTED_TRAIN = os.path.join(ROOT_DIR, "Data", "dataset_selected_features_train.csv")
DATA_SELECTED_TEST = os.path.join(ROOT_DIR, "Data", "dataset_selected_features_test.csv")
DATA_ALIGNED = os.path.join(ROOT_DIR, "Data", "aligned_2026_01_02.csv")

# ============================================================
# 气象 API 配置
# ============================================================
LATITUDE = 29.866465
LONGITUDE = 121.52707
WEATHER_API_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER_TIMEOUT = 20
WEATHER_CACHE_TTL = 300

# ============================================================
# 模型参数
# ============================================================
SOLAR_LOOKBACK = 24
CHARGING_LOOKBACK = 96
MC_DROPOUT_SAMPLES = 30
SOLAR_FEATURE_DIM = 7
CHARGING_FEATURE_DIM = 6

SOLAR_FEATURE_COLS = [
    "power", "hour_sin", "hour_cos",
    "shortwave_radiation (W/m2)", "direct_radiation (W/m2)",
]

# ============================================================
# 电价配置
# ============================================================
PEAK_PRICE = 1.0
MID_PRICE = 0.6
VALLEY_PRICE = 0.3

PRICE_PERIODS = {
    "peak":   [(8, 11), (18, 21)],
    "mid":    [(6, 8), (11, 18), (21, 22)],
    "valley": [(0, 6), (22, 24)],
}

# ============================================================
# 预测步数选项
# ============================================================
PREDICTION_OPTIONS = {
    "15min 单步预测": 1,
    "1h 预测": 4,
    "24h 预测": 96,
}
DEFAULT_PREDICTION_STEPS = 4

# ============================================================
# BentoML 服务配置
# ============================================================
SERVICE_PORT = 7863
SERVICE_TITLE = "光储充智能预测 Demo"