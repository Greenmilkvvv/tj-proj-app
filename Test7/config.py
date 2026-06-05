"""
光储充预测 Demo — 全局配置文件 (v4)
=====================================
基于 BentoML 部署，移除 Gradio 专有配置。
所有可调参数集中在此，修改后重新构建 Bento 即可生效。
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
SOLAR_FEATURE_DIM = 7      # 光伏模型输入特征数 (由 best_generator.pth weight_ih_l0 [512,7] 确认)
CHARGING_FEATURE_DIM = 6   # 充电模型输入特征数 (由 tcn.0.conv.weight [64,6,3] 确认)

# 光伏模型使用的特征列 (aligned CSV, 训练时7维但 aligned CSV 可能只有5维可用列)
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
    "15min 单步预测": 1,
    "1h 预测": 4,
    "24h 预测": 96,
}
DEFAULT_PREDICTION_STEPS = 4  # 默认 1h 预测

# ============================================================
# 服务配置 (BentoML v4)
# ============================================================
BENTOML_PORT = 3000
APP_TITLE = "光储充智能预测 Demo v4"