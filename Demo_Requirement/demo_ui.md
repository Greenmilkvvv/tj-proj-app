# 光储充预测 Demo — 需求文档 v3.0

> **修订说明**：基于项目现有资源（光伏预测模型权重、充电负荷预测模型权重、历史训练数据、气象 API）重构需求。剔除了所有无法通过现有资源实现的功能（实时充放电数据、储能 SOC、变压器监测等），聚焦于可交付的核心能力：实时气象展示、模型驱动的光伏/负荷预测、历史数据探索、电价策略建议。

---

## 一、项目定位

构建一个 **「光储充智能预测 Demo」** Gradio Web 应用，面向项目演示与学术交流，展示以下核心能力：

- 🌤️ 基于 Open-Meteo 免费 API 的实时气象数据仪表板
- ☀️ 基于 CNN-LSTM 模型的光伏出力滚动预测
- ⚡ 基于 TCN-Attention-LSTM Hybrid 模型的充电负荷预测（含 MC Dropout 不确定性估计）
- 📊 历史训练数据探索与多维度可视化分析
- 💰 分时电价策略建议（规则引擎）

**目标用户**：项目演示评审、学术交流、能源管理概念验证。

---

## 二、项目现有资源清单

在开始开发前，确认以下资源已就位：

| 资源 | 路径 | 用途 |
|------|------|------|
| 光伏模型权重 | `Solar_Forecast/best_pth/best_generator.pth` | CNN-LSTM 光伏出力预测 |
| 光伏模型定义 | `Solar_Forecast/NN.py` | GeneratorWithFeatures / CNN_LSTM |
| 充电模型权重 | `Charging_Forecast/best_pth/final_best_hybrid_model.pth` | TCN-Attention-LSTM 负荷预测 |
| 充电模型训练代码 | `Charging_Forecast/Bys-TCN-Attention-LSTM_model.ipynb` | HybridModel 结构定义 |
| 气象 API 模块 | `Weather/get_weather.py` | Open-Meteo 实时气象 + 15min 预报 |
| 全特征训练集 | `Data/dataset_all_features_train.csv` | 数据探索与模型输入构建 |
| 全特征测试集 | `Data/dataset_all_features_test.csv` | 数据探索 |
| 精选特征训练集 | `Data/dataset_selected_features_train.csv` | 充电模型输入构建 |
| 精选特征测试集 | `Data/dataset_selected_features_test.csv` | 充电模型验证 |
| 光伏训练数据 | `Data/aligned_2026_01_02.csv` | 光伏模型训练参考 |

---

## 三、技术架构

```
┌──────────────────────────────────────────────────────────────────┐
│                       Gradio Web UI                               │
├────────────────┬──────────────────┬────────────────┬─────────────┤
│  Tab1: 预测    │  Tab2: 预测曲线   │  Tab3: 数据探索 │  Tab4: 气象  │
│  总览仪表板    │  光伏/负荷/净负荷 │  历史分析+画像  │  实时+预报   │
├────────────────┴──────────────────┴────────────────┴─────────────┤
│                          服务层 (Python)                          │
├──────────────┬──────────────┬──────────────┬────────────────────┤
│ 气象服务      │ 光伏预测服务  │ 负荷预测服务  │ 数据探索服务        │
│ (Open-Meteo) │ (CNN-LSTM)   │ (HybridModel │ (pandas+plotly)    │
│              │              │  +MC Dropout) │                    │
├──────────────┴──────────────┴──────────────┴────────────────────┤
│                          资源层                                    │
├──────────────┬──────────────┬──────────────┬────────────────────┤
│ Open-Meteo   │ Solar_Forecast│ Charging_    │ Data/               │
│ API (免费)   │ /best_pth/   │ Forecast/    │ (CSV 数据集)        │
│              │              │ best_pth/    │                    │
└──────────────┴──────────────┴──────────────┴────────────────────┘
```

---

## 四、页面功能详述

### Tab 1 — 预测总览仪表板（首页）

#### 4.1.1 顶部指标卡片

| 指标 | 说明 | 数据来源 |
|------|------|----------|
| ☀️ 光伏预测峰值 (kW) | 预测窗口内光伏出力最大值 | 光伏模型滚动预测 |
| ⚡ 充电负荷峰值 (kW) | 预测窗口内充电负荷最大值 | 充电模型 MC Dropout 预测 |
| 🌿 绿电替代率 (%) | min(光伏总出力 / 充电总需求 × 100, 100) | 两个模型预测结果计算 |

#### 4.1.2 电价状态栏

- 显示 **当前电价**（元/kWh）和 **当前时段标签**（峰/平/谷）
- 内置分时电价表，所有参数集中在 `config.py` 中：
  - 峰时：1.0 元/kWh（默认时段 8:00-11:00, 18:00-21:00）
  - 平时：0.6 元/kWh（默认时段 6:00-8:00, 11:00-18:00, 21:00-22:00）
  - 谷时：0.3 元/kWh（默认时段 22:00-次日 6:00）
- 时段划分和电价均可通过配置文件修改

#### 4.1.3 预测控制

- 下拉选择预测时长：**1h / 3h / 6h / 12h**（15min 粒度，即 4 / 12 / 24 / 48 步）
- 「🚀 运行预测」按钮触发预测流水线：
  1. 调用气象 API 获取当前气象数据（辐照度、云量、温度、降雨）
  2. 从历史 CSV 中提取最近时间窗口数据，构造模型输入序列
  3. 光伏模型滚动推理 → 光伏出力预测序列
  4. 充电模型 MC Dropout 推理（多次采样）→ 负荷均值 + 置信区间
  5. 汇总仪表板指标 + 生成策略建议报告

> **实现说明**：模型需要历史序列作为输入（光伏模型 lookback=24 步即 6h，充电模型 lookback=96 步即 24h）。预测采用滚动方式：每预测一步，将其追加到输入窗口末尾、丢弃最旧一步，再预测下一步。

#### 4.1.4 策略建议报告

基于预测结果自动生成文本建议（规则引擎，仅供参考）：

- **预测总览**：光伏总出力(kWh)、充电总需求(kWh)、绿电替代率(%)
- **供需平衡分析**：光伏盈余时段与不足时段的统计
- **电价引导策略**：
  - 若谷时且光伏有盈余 → "建议在谷时集中充电，利用低价+绿电双重优势"
  - 若峰时且光伏不足 → "建议减少峰时充电，优先使用谷时或光伏盈余时段"
  - 若光伏充足 → "光伏出力充裕，可优先消纳绿电"
- **风险提示**：若气象 API 获取失败，提示使用默认值；若模型加载失败，提示回退到模拟模式

> ⚠️ 策略建议基于简单规则，不涉及实时优化调度算法。

---

### Tab 2 — 预测曲线（可视化）

#### 4.2.1 光伏出力预测曲线

- X 轴：时间（15min 粒度）
- Y 轴：光伏出力 (kW)
- 折线图展示预测序列
- 标注峰值功率点

#### 4.2.2 充电负荷预测曲线

- 折线图展示负荷均值预测 (kW)
- 半透明带状区域展示 95% 置信区间（基于 MC Dropout 多次采样的标准差）

#### 4.2.3 净负荷曲线

- 净负荷 = 充电负荷预测 − 光伏预测
- 正值区域（红色）= 光伏不足以覆盖充电需求，需从电网取电
- 负值区域（绿色）= 光伏盈余，可储能或上网
- 标注零线便于直观判断

#### 4.2.4 分时电价背景

- 图表背景按峰/平/谷时段着色：
  - 🔴 峰时 (红色浅底)
  - 🟡 平时 (黄色浅底)
  - 🟢 谷时 (绿色浅底)
- 帮助直观判断各时段的电价策略

---

### Tab 3 — 数据探索

#### 4.3.1 数据集概览

- 展示各 CSV 数据集的基本信息：
  - 行数、列数、日期范围
  - 缺失值统计
  - 特征列名列表
- 数据来源：`Data/dataset_all_features_train.csv`、`Data/dataset_all_features_test.csv`

#### 4.3.2 历史负荷曲线浏览

- 下拉选择日期，查看该日的 15min 粒度假负荷曲线（load_kw）
- 支持叠加对比多条曲线（选择多日）
- 数据来源：Data 文件夹中的 CSV 文件

#### 4.3.3 特征相关性分析

- 计算各数值特征与 `load_kw` 的 Pearson 相关系数
- 以水平条形图展示，按相关系数绝对值降序排列
- 帮助理解哪些特征对负荷预测最重要

#### 4.3.4 小时级负荷画像

- 按 24 小时分组，统计历史负荷的均值 ± 标准差
- 以折线图 + 误差带展示
- 揭示负荷的典型日模式（如早晚高峰）

---

### Tab 4 — 实时气象

#### 4.4.1 当前气象概览

| 指标 | 说明 |
|------|------|
| 🌡️ 温度 | 当前温度 (°C) |
| ☁️ 云量 | 当前云层覆盖率 (%) |
| ☀️ 辐照度 | 当前短波辐射 (W/m²) |
| 🌧️ 降雨 | 当前降雨量 (mm/h) |
| 📅 数据时间 | 气象数据对应的时间戳 |

数据来源：Open-Meteo API (`api.open-meteo.com`)，免费无需 API Key。

#### 4.4.2 辐照度与云量双轴曲线

- X 轴：时间（15min 粒度）
- 左 Y 轴：辐照度 (W/m²)，橙色填充曲线
- 右 Y 轴：云量 (%)，灰色虚线
- 展示今日辐照度变化趋势及云量关系

#### 4.4.3 未来 3 小时预报表

- 表格展示未来 3 小时（12 个 15min 步）的：
  - 时间、天气描述（☀️晴天 / ⛅多云 / ☁️阴天 / 🌧️降雨）
  - 辐照度 (W/m²)
  - 降雨量 (mm)

#### 4.4.4 天气预警

- 若未来 2 小时内有强降雨（> 5mm/h）：🚨 红色预警框，提示光伏出力可能骤降
- 若天气正常：🟢 绿色提示框

---

### Tab 5 — 系统状态检查

#### 4.5.1 依赖检查

- Python 版本
- 关键包及其版本：`torch`, `gradio`, `pandas`, `numpy`, `plotly`, `requests`, `scikit-learn`

#### 4.5.2 模型文件检查

| 检查项 | 预期 |
|--------|------|
| 光伏模型文件 | `Solar_Forecast/best_pth/best_generator.pth` 存在 + 可加载 |
| 充电模型文件 | `Charging_Forecast/best_pth/final_best_hybrid_model.pth` 存在 + 可加载 |
| 模型参数量 | 显示两个模型的参数总数 |

#### 4.5.3 数据文件检查

- 各 CSV 文件存在性检查
- `load_kw` 列是否存在（充电负荷数据）
- `power` 列是否存在（光伏数据，aligned CSV）
- 特征维度校验

#### 4.5.4 气象 API 连通性检查

- 向 Open-Meteo API 发送测试请求
- 显示连通状态 + 响应时间

---

## 五、预测流水线数据流

```
用户点击「运行预测」
       │
       ▼
① 调用气象 API → 获取当前辐照度/云量/温度/降雨
       │
       ▼
② 从历史 CSV 提取最近 lookback 条记录
   光伏: 最近 24 条 (6h) → 构造 (24, 5) 输入张量
   充电: 最近 96 条 (24h) → 构造 (96, 6+) 输入张量
       │
       ▼
③ 光伏 CNN-LSTM 滚动推理 (每次输出 1 步，追加后继续)
   → solar_forecast[n_steps]
       │
       ▼
④ 充电 Hybrid 模型 MC Dropout 推理 (采样 30 次)
   → load_mean[n_steps] + load_std[n_steps] (95% CI)
       │
       ▼
⑤ 计算汇总指标 → 生成策略建议 → 渲染到 UI
```

---

## 六、模型信息

| 模型 | 架构 | 权重路径 | 输入维度 | 输出 | 输入窗口 |
|------|------|----------|----------|------|----------|
| 光伏预测 | CNN-LSTM (GeneratorWithFeatures) | `Solar_Forecast/best_pth/best_generator.pth` | (batch, 24, 5) | 光伏出力单步预测 (kW) | 24 步 (6h) |
| 充电负荷预测 | TCN-Attention-LSTM Hybrid | `Charging_Forecast/best_pth/final_best_hybrid_model.pth` | (batch, 96, 6+) | 负荷单步预测 (kW) | 96 步 (24h) |

**MC Dropout**：充电模型的 Dropout 层在推理时保持开启，多次前向传播采样得到预测分布，用于计算置信区间。采样次数建议 30 次（可通过 `config.py` 调整）。

**滚动预测**：两个模型均为单步预测器（输入 N 个历史步 → 输出 1 个未来步）。多步预测通过滚动方式实现：每预测一步，将预测值追加到输入序列、移除最早一步，再进行下一步预测。

---

## 七、数据文件说明

| 文件 | 行数 | 特征数 | 用途 |
|------|------|--------|------|
| `Data/dataset_all_features_train.csv` | ~4000+ | 18 | 全特征训练集（数据探索 + 模型输入构建） |
| `Data/dataset_all_features_test.csv` | ~1000+ | 18 | 全特征测试集（数据探索） |
| `Data/dataset_selected_features_train.csv` | ~4000+ | 7 | 精选特征训练集（充电模型训练参考） |
| `Data/dataset_selected_features_test.csv` | ~1000+ | 7 | 精选特征测试集 |
| `Data/aligned_2026_01_02.csv` | — | — | 光伏模型训练对齐数据 |

全特征列包含：`timestamp, hour, minute, day, month, dayofweek, is_weekend, hour_sin, hour_cos, dow_sin, dow_cos, lag_1, lag_4, lag_96, lag_672, rolling_mean_4, rolling_std_4, price, load_kw`

精选特征列：`timestamp, lag_1, lag_96, lag_672, rolling_mean_4, rolling_std_4, price, load_kw`

aligned CSV 包含光伏相关列：`datetime, power, shortwave_radiation, direct_radiation, diffuse_radiation, direct_normal_irradiance, temperature_2m` 等。

---

## 八、技术约束

1. **全部本地运行**：不依赖外部数据库或云端 GPU 服务
2. **CPU 兼容**：模型推理优先 CUDA，无 GPU 时自动回退 CPU
3. **低延迟要求**：单次预测流水线 < 15 秒（含 MC Dropout 30 次采样 + 滚动预测）
4. **容错设计**：模型加载失败时回退到基于历史统计的模拟数据，确保 UI 始终可运行
5. **可配置**：电价、时段划分、预测步数、MC Dropout 采样次数等参数集中在 `config.py`

---

## 九、排除范围（明确不实现）

| 功能 | 排除原因 |
|------|----------|
| 实时充放电功率监控 | 无充电桩数据采集接口，无法获取实时充放电数据 |
| 储能 SOC 监测 | 无储能系统接入，无电池状态数据 |
| 储能优化调度算法 | 无实时 SOC 和功率数据，不具备调度条件 |
| 变压器实时负载率 | 无变压器监测数据 |
| 充电站远程控制指令 | Demo 仅做预测展示，不涉及控制闭环 |
| 用户登录/权限管理 | Demo 场景无需认证 |
| 数据库持久化 | 全量数据来自静态 CSV + 实时 API，无需数据库 |
| 历史预测结果回溯 | Demo 不存储历史预测记录 |

---

## 十、运行方式

```bash
# 安装依赖
pip install gradio torch pandas numpy plotly requests scikit-learn

# 启动应用
python app.py
```

浏览器打开 `http://localhost:7860` 即可访问。

---

## 十一、配置文件说明 (`config.py`)

```python
# ========== 模型路径 ==========
SOLAR_MODEL_PATH = "Solar_Forecast/best_pth/best_generator.pth"
CHARGING_MODEL_PATH = "Charging_Forecast/best_pth/final_best_hybrid_model.pth"

# ========== 数据路径 ==========
DATA_ALL_TRAIN = "Data/dataset_all_features_train.csv"
DATA_ALL_TEST = "Data/dataset_all_features_test.csv"
DATA_SELECTED_TRAIN = "Data/dataset_selected_features_train.csv"
DATA_SELECTED_TEST = "Data/dataset_selected_features_test.csv"
DATA_ALIGNED = "Data/aligned_2026_01_02.csv"

# ========== 气象 API ==========
LATITUDE = 29.866465
LONGITUDE = 121.52707
WEATHER_API_URL = "https://api.open-meteo.com/v1/forecast"

# ========== 电价配置 (元/kWh) ==========
PEAK_PRICE = 1.0
MID_PRICE = 0.6
VALLEY_PRICE = 0.3

# ========== 电价时段 ==========
PRICE_PERIODS = {
    "peak":   [(8, 11), (18, 21)],
    "mid":    [(6, 8), (11, 18), (21, 22)],
    "valley": [(0, 6), (22, 24)],
}

# ========== 预测参数 ==========
SOLAR_LOOKBACK = 24       # 光伏模型输入窗口 (步)
CHARGING_LOOKBACK = 96    # 充电模型输入窗口 (步)
MC_DROPOUT_SAMPLES = 30   # MC Dropout 采样次数
PREDICTION_STEPS = [4, 12, 24, 48]  # 可选预测步数 (对应 1h/3h/6h/12h)
```

---

## 十二、开发优先级

| 优先级 | 功能模块 | 说明 |
|--------|----------|------|
| P0 | 气象 Tab | 已有 `Weather/get_weather.py` 可直接复用 |
| P0 | 系统检查 Tab | 纯文件/环境检查，无外部依赖 |
| P1 | 数据探索 Tab | 基于静态 CSV，逻辑清晰 |
| P1 | 预测总览 Tab | 核心功能，需整合模型推理 |
| P2 | 预测曲线 Tab | 依赖预测结果，基于 Plotly 绘制 |
| P2 | 策略建议报告 | 规则引擎，依赖预测结果 |