"""
光储充预测 Demo — 全局配置文件 (v3)
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
# 预测步数选项 (供UI下拉使用) — v3: 三种模式
# ============================================================
PREDICTION_OPTIONS = {
    "15min 单步预测": 1,
    "1h 预测": 4,
    "24h 预测": 96,
}
DEFAULT_PREDICTION_STEPS = 4  # 默认 1h 预测

# ============================================================
# UI 配置
# ============================================================
APP_TITLE = "光储充智能预测 Demo v3"
APP_PORT = 7862
APP_THEME = "soft"

CUSTOM_CSS = """
.gradio-container { max-width: 1400px !important; margin: auto; }
.warning-box {
    background: #fff3cd; border: 1px solid #ffc107; border-radius: 8px;
    padding: 12px; margin: 10px 0; color: #856404;
}
.info-box {
    background: #d1ecf1; border: 1px solid #17a2b8; border-radius: 8px;
    padding: 12px; margin: 10px 0; color: #0c5460;
}
.success-box {
    background: #d4edda; border: 1px solid #28a745; border-radius: 8px;
    padding: 12px; margin: 10px 0; color: #155724;
}
.title-section { text-align: center; margin-bottom: 20px; }
.title-section h1 { font-size: 28px; color: #2c3e50; margin-bottom: 6px; }
.title-section p { font-size: 14px; color: #7f8c8d; }
.card-grid {
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 12px 0;
}
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 10px; padding: 16px; color: white; text-align: center;
    box-shadow: 0 2px 10px rgba(0,0,0,0.15);
}
.metric-card.green { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }
.metric-card.orange { background: linear-gradient(135deg, #f2994a 0%, #f2c94c 100%); }
.metric-value { font-size: 28px; font-weight: bold; }
.metric-label { font-size: 13px; opacity: 0.9; margin-top: 4px; }
footer { visibility: hidden; }

/* ---- Dark mode ---- */
.dark .upload-card h3,
.dark .upload-card p,
.dark .upload-card .prose {
    color: #e0e0e0 !important;
}
.dark .upload-card {
    background: #1e1e1e !important;
    border-color: #444 !important;
}

.dark table thead th,
.dark .table-wrap thead th,
.dark table th {
    color: #e0e0e0 !important;
    background-color: #2a2a2a !important;
}
.dark table tbody td,
.dark .table-wrap tbody td {
    color: #d0d0d0 !important;
}

.dark .gauge text,
.dark .gauge .value {
    fill: #e0e0e0 !important;
}
.dark .gauge .adjust {
    fill: #999 !important;
}

.dark .tabs > .tab-nav > button {
    color: #bbb !important;
}
.dark .tabs > .tab-nav > button.selected {
    color: #fff !important;
    background-color: #444 !important;
}

/* ---- ui-resizable-handle 隐藏，防止拖拽改变页面宽度 ---- */
.ui-resizable-handle {
    display: none !important;
}
"""
