# 光储充预测 Demo — 需求文档 v5.1

> **修订说明（v5.1）**：新增「误差分析」Tab，补充充电模型回测与残差分析功能描述；更新整体架构图；与 `Test2/` 实现保持严格对齐。

---

## 一、项目定位

构建一个 **「光储充智能预测 Demo」** Gradio Web 应用，面向项目演示与学术交流，展示以下核心能力：

- 🌤️ 基于 Open-Meteo 免费 API 的实时气象数据仪表板
- ☀️ 基于 LSTM 模型的光伏出力滚动预测
- ⚡ 基于 TCN-Attention-LSTM Hybrid 模型的充电负荷预测（含置信区间估计）
- 📊 历史训练数据探索与多维度可视化分析
- 💡 基于预测结果的供需平衡分析与策略建议（规则引擎）
- 🔬 充电模型误差分析与回测评估

**目标用户**：项目演示评审、学术交流、能源管理概念验证。

---

## 二、项目现有资源清单

| 资源 | 路径 | 用途 |
|------|------|------|
| 光伏模型权重 | `Solar_Forecast/best_pth/best_generator.pth` | LSTM 光伏出力预测 (GeneratorWithFeatures) |
| 光伏模型定义 | `Solar_Forecast/NN.py` | LSTMPredictor + GeneratorWithFeatures 结构 |
| 充电模型权重 | `Charging_Forecast/best_pth/final_best_hybrid_model.pth` | TCN-Attention-LSTM 负荷预测 |
| 充电模型训练代码 | `Charging_Forecast/Bys-TCN-Attention-LSTM_model.ipynb` | HybridModel 结构定义 |
| 充电模型训练历史 | `Charging_Forecast/final_train_history.pkl` | 训练 Loss 曲线回放 |
| 气象 API 模块 | `Weather/get_weather.py` | Open-Meteo 实时气象 + 15min 预报 |
| 全特征训练集 | `Data/dataset_all_features_train.csv` | 数据探索 |
| 全特征测试集 | `Data/dataset_all_features_test.csv` | 数据探索 + 回测验证 |
| 精选特征训练集 | `Data/dataset_selected_features_train.csv` | 充电模型训练参考 |
| 精选特征测试集 | `Data/dataset_selected_features_test.csv` | 充电模型验证 + 残差分析 |
| 光伏训练对齐数据 | `Data/aligned_2026_01_02.csv` | 光伏模型训练及预测输入构建 |

---

## 三、技术架构

```
┌──────────────────────────────────────────────────────────────────────┐
│                         Gradio Web UI                                 │
├──────────────┬──────────────┬──────────────┬──────────────┬──────────┤
│ Tab1: 预测核心│ Tab2: 气象监测│ Tab3: 数据探索│ Tab4: 策略建议│Tab5: 误差│
│ 光伏+负荷预测 │ 实时+预报+预警│ 历史分析+画像 │ 供需平衡分析  │ 回测+残差│
├──────────────┴──────────────┴──────────────┴──────────────┴──────────┤
│                          服务层 (Python)                               │
├────────────────┬────────────────┬───────────────┬────────────────────┤
│　weather_service│ prediction_   │ data_service  │                    │
│   (Open-Meteo) │ service        │ (pandas +     │                    │
│                │ (LSTM +        │  plotly)      │                    │
│                │  TCN-Attn +    │               │                    │
│                │  LSTM)         │               │                    │
├────────────────┴────────────────┴───────────────┴────────────────────┤
│                           资源层                                       │
├────────────────┬────────────────┬───────────────┬────────────────────┤
│ Open-Meteo     │ Solar_Forecast │ Charging_     │ Data/              │
│ API (免费)     │ /best_pth/     │ Forecast/     │ (CSV 数据集)        │
│                │                │ best_pth/     │                    │
└────────────────┴────────────────┴───────────────┴────────────────────┘
```

---

## 四、页面功能详述

### Tab 1 — 预测核心（首页）

整合预测参数、指标卡片、预测曲线、供需平衡分析于一个视图中。

#### 4.1.1 预测参数控制

- 下拉选择预测时长：**1h / 3h / 6h / 12h**（15min 粒度，即 4 / 12 / 24 / 48 步）
- 「🚀 执行预测」按钮触发预测流水线：
  1. 调用气象 API 获取当前气象数据（辐照度、云量、温度、降雨）
  2. 从 `aligned_2026_01_02.csv` 提取最近时间窗口数据，构造模型输入序列
  3. 光伏 LSTM 滚动推理 → 光伏出力预测序列
  4. 充电 Hybrid 模型推理（含置信区间）→ 负荷均值 + 上限/下限
  5. 汇总指标 + 渲染图表

#### 4.1.2 核心指标卡片

| 指标 | 说明 | 数据来源 |
|------|------|----------|
| ☀️ 光伏总出力 (kWh) | 预测窗口内光伏累计出力 | 光伏模型滚动预测求和 |
| ⚡ 充电总需求 (kWh) | 预测窗口内充电累计需求 | 充电模型预测求和 |
| 🌿 绿电替代率 (%) | min(光伏总出力 / 充电总需求 × 100, 100) | 两个模型预测结果计算 |
| 模型加载状态 | 显示两个模型是否正常加载 | 模型文件存在性 + 加载结果 |

#### 4.1.3 光伏出力 & 充电负荷预测曲线

- X 轴：时间（15min 粒度）
- Y 轴：功率 (kW)
- 橙色曲线：光伏出力预测（折线 + 半透明填充）
- 蓝色曲线：充电负荷均值预测（折线）
- 蓝色半透明带：置信区间（均值 ± 15%）

#### 4.1.4 供需平衡图

- 柱状图展示 净功率 = 光伏预测 − 负荷预测
- 绿色柱（光伏盈余，正值）：光伏可覆盖充电需求
- 红色柱（光伏不足，负值）：需从电网取电
- 零线参考线

#### 4.1.5 峰谷指标

| 指标 | 说明 |
|------|------|
| 光伏峰值 | 预测窗口内光伏出力最大值 + 对应时间 |
| 负荷峰值 | 预测窗口内充电负荷最大值 |

> **实现说明**：光伏模型 lookback=24 步（6h），充电模型 lookback=96 步（24h）。预测采用滚动方式：每预测一步，更新输入窗口的时间编码（sin/cos），将预测值追加到输入末尾、丢弃最旧一步，再预测下一步。

---

### Tab 2 — 气象监测

#### 4.2.1 当前气象概览卡片

以 HTML 卡片展示当前时刻的关键气象指标：

| 指标 | 说明 |
|------|------|
| 🌡️ 温度 | 当前温度 (°C) |
| ☁️ 云量 | 当前云层覆盖率 (%) |
| ☀️ 辐照度 | 当前短波辐射 (W/m²) |
| 🌧️ 降雨 | 当前降雨量 (mm) |
| 📅 数据时间 | 气象数据对应的时间戳 |

数据来源：Open-Meteo API (`api.open-meteo.com`)，免费无需 API Key。

#### 4.2.2 辐照度趋势图

- 展示当日辐照度变化趋势（Plotly 图表）
- 与云量数据关联展示（双轴图）

#### 4.2.3 未来 3 小时预报表

- 表格展示未来 3 小时（12 个 15min 步）的：
  - 时间
  - 辐照度 (W/m²)
  - 云量 (%)
  - 降雨量 (mm)
  - 天气描述

#### 4.2.4 天气预警

- 若有降雨风险：⚠️ 黄色警告框，提示光伏出力可能受影响
- 若气象 API 获取失败：🔴 错误提示
- 若天气正常：🟢 绿色正常提示

#### 4.2.5 气象数据缓存

- 气象数据缓存 5 分钟（`WEATHER_CACHE_TTL = 300`），避免频繁 API 调用
- 「🔄 刷新气象数据」按钮可手动刷新，忽略缓存

---

### Tab 3 — 数据探索

#### 4.3.1 数据集概览

- 展示训练集和测试集的基本信息：
  - 行数、列数、日期范围
  - 缺失值统计
- 数据来源：`Data/dataset_all_features_train.csv`、`Data/dataset_all_features_test.csv`

#### 4.3.2 历史日负荷曲线

- 多选下拉框选择日期（最多 3 个），查看该日的 15min 粒度负荷曲线
- 支持叠加对比多条曲线

#### 4.3.3 特征相关性分析

- 计算各数值特征与负荷的 Pearson 相关系数
- 以水平条形图展示，按相关系数绝对值降序排列
- 帮助理解哪些特征对负荷预测最重要

#### 4.3.4 小时级负荷画像

- 按 24 小时分组，统计历史负荷的均值 ± 1σ 标准差
- 以折线图 + 误差带展示
- 揭示负荷的典型日模式（如早晚高峰）

---

### Tab 4 — 策略建议

#### 4.4.1 触发方式

- 用户需先在「预测核心」Tab 执行预测
- 点击「📋 生成策略建议」按钮，基于最近一次预测结果生成

#### 4.4.2 策略内容

基于预测结果自动生成文本建议（规则引擎，仅供参考）：

- **预测总览**：光伏总出力 (kWh)、充电总需求 (kWh)、绿电替代率 (%)
- **供需平衡分析**：
  - 🟢 储能充电建议：列出光伏盈余 > 5kW 的时段，建议向储能系统充电
  - 🔴 储能放电建议：列出缺口 > 5kW 的时段，可调度储能放电补充
- **风险提示**：若未执行预测，提示用户先执行预测

> ⚠️ 策略建议基于简单规则（净功率阈值判断），不涉及实时优化调度算法。分时电价配置已在 `config.py` 中预留，可后续扩展电价引导策略。

---

### Tab 5 — 误差分析

提供充电模型的离线评估与误差分析，帮助理解模型性能。

#### 4.5.1 充电模型回测

- 对测试集执行 24 步滚动预测
- 双轴图对比真实值 vs 预测值，含 MAPE / MAE / RMSE 指标
- 回测结果汇总为 Markdown 文本

#### 4.5.2 残差分布分析

- **残差分布直方图**：展示预测残差（真实值 − 预测值）的频率分布，叠加正态分布参考线
- **按小时误差箱线图**：展示不同小时段（0-23h）的误差分布，帮助识别模型的薄弱时段

#### 4.5.3 光伏模型信息

- 展示光伏模型的结构概述：
  - 模型架构 (LSTMPredictor + GeneratorWithFeatures)
  - 权重文件路径
  - 输入/输出维度
  - 参数量统计

---

## 五、预测流水线数据流

```
用户点击「执行预测」
       │
       ▼
① 调用气象 API → 获取当前辐照度/云量/温度/降雨
       │
       ▼
② 从 aligned_2026_01_02.csv 提取最近 N 条记录
    光伏: 最近 24 条 (6h) → 构造 (1, 24, 7) 输入张量
    充电: 最近 96 条 (24h) → 构造 (1, 96, 6) 输入张量
    特征窗口动态构建：power 列 + sin/cos 时间编码 + 气象特征补齐
       │
       ▼
③ 光伏 LSTM (GeneratorWithFeatures) 滚动推理
   每步输出 1 步预测，更新窗口后继续
   → solar_forecast[n_steps]
       │
       ▼
④ 充电 Hybrid 模型 (TCN-Attention-LSTM) 滚动推理
   → load_mean[n_steps] + 置信区间 (mean ± 15%)
       │
       ▼
⑤ 计算汇总指标 → 渲染图表到 UI
```

---

## 六、模型信息

| 模型 | 架构 | 权重路径 | 输入维度 | 输出 | 输入窗口 |
|------|------|----------|----------|------|----------|
| 光伏预测 | LSTM (GeneratorWithFeatures) | `Solar_Forecast/best_pth/best_generator.pth` | (batch, 24, 7) | 光伏出力单步预测 (kW) | 24 步 (6h) |
| 充电负荷预测 | TCN-Attention-LSTM Hybrid | `Charging_Forecast/best_pth/final_best_hybrid_model.pth` | (batch, 96, 6) | 负荷单步预测 (kW) | 96 步 (24h) |

### 光伏模型结构

- `LSTMPredictor`: 2 层单向 LSTM，hidden_size=128，dropout=0.2
- `GeneratorWithFeatures`: 包装 LSTMPredictor，直接输出单步光伏功率预测

### 充电模型结构

- `TCNBlock`: 5 层膨胀卷积（dilations=[1,2,4,8,16]），kernel_size=3，含残差连接
- `SimpleAttention`: 可学习注意力权重，对 TCN 输出加权
- `HybridModel`: TCN → Attention → LSTM(hidden_size=121) → 拼接原始最后一步 → FC → ReLU

### 滚动预测

两个模型均为单步预测器（输入 N 个历史步 → 输出 1 个未来步）。多步预测通过滚动方式实现：每预测一步，更新输入序列的时间编码列（sin/cos），将预测值写入 power 列，追加到序列末尾、移除最早一步，再进行下一步预测。

---

## 七、数据文件说明

| 文件 | 用途 | 关键列 |
|------|------|--------|
| `Data/dataset_all_features_train.csv` | 数据探索（全特征训练集） | timestamp, hour, load_kw, price, lag_1, lag_96 等 |
| `Data/dataset_all_features_test.csv` | 数据探索（全特征测试集） | 同上 |
| `Data/dataset_selected_features_train.csv` | 充电模型训练参考 | timestamp, lag_1, lag_96, lag_672, rolling_mean_4, rolling_std_4, price, load_kw |
| `Data/dataset_selected_features_test.csv` | 充电模型验证 + 回测 + 残差分析 | 同上 |
| `Data/aligned_2026_01_02.csv` | 光伏模型训练及预测输入 | datetime, power, shortwave_radiation, direct_radiation, diffuse_radiation, temperature_2m 等 35 列 |

### aligned CSV 实际列名（部分）

`datetime`, `power`, `shortwave_radiation (W/m²)`, `direct_radiation (W/m²)`, `diffuse_radiation (W/m²)`, `direct_normal_irradiance (W/m²)`, `temperature_2m (°C)`, `cloudcover (%)`, `windspeed_10m (m/s)`, `precipitation (mm)` 等。

> **注意**：aligned CSV 不含 `hour_sin`/`hour_cos` 时间编码列，预测服务在构建输入窗口时动态生成这些特征。

---

## 八、技术约束

1. **全部本地运行**：不依赖外部数据库或云端 GPU 服务
2. **CPU 兼容**：模型推理优先 CUDA，无 GPU 时自动回退 CPU
3. **低延迟要求**：单次预测流水线 < 15 秒（含双模型滚动预测）
4. **容错设计**：
   - 模型加载失败时回退到基于数学模型的模拟数据（日模式正弦曲线），确保 UI 始终可运行
   - 气象 API 失败时使用默认辐照度值，并显示警告提示
5. **可配置**：电价、时段划分、预测步数、气象坐标等参数集中在 `config.py`
6. **模型兼容性**：
   - 充电模型 hidden_size 从权重文件动态推断（`lstm.weight_ih_l0` shape / 4）
   - 光伏模型固定 hidden_size=128，input_size 通过 `config.py` 的 `SOLAR_FEATURE_DIM` 配置

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
| 分时电价动态优化调度 | Demo 阶段仅做供需平衡分析，电价引导策略已预留配置，可在后续版本扩展 |

---

## 十、运行方式

```bash
# 安装依赖
pip install gradio torch pandas numpy plotly requests scikit-learn

# 启动应用
cd Test2 && python app.py
```

浏览器打开 `http://localhost:7860` 即可访问。

---

## 十一、配置文件说明 (`config.py`)

```python
# ========== 应用基础 ==========
APP_TITLE = "光储充智能预测 Demo"
APP_PORT = 7860

# ========== 模型路径 ==========
SOLAR_MODEL_PTH = "Solar_Forecast/best_pth/best_generator.pth"
CHARGING_MODEL_PTH = "Charging_Forecast/best_pth/final_best_hybrid_model.pth"

# ========== 数据路径 ==========
DATA_ALL_TRAIN = "Data/dataset_all_features_train.csv"
DATA_ALL_TEST = "Data/dataset_all_features_test.csv"
DATA_ALIGNED = "Data/aligned_2026_01_02.csv"

# ========== 气象 API ==========
LATITUDE = 29.866465
LONGITUDE = 121.52707
WEATHER_API_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER_CACHE_TTL = 300  # 气象数据缓存时间 (秒)

# ========== 模型输入参数 ==========
SOLAR_FEATURE_DIM = 7      # 光伏模型输入特征维度
CHARGING_FEATURE_DIM = 6   # 充电模型输入特征维度
SOLAR_LOOKBACK = 24        # 光伏模型输入窗口 (步)
CHARGING_LOOKBACK = 96     # 充电模型输入窗口 (步)

# ========== 电价配置 (元/kWh) [预留，策略建议可扩展] ==========
PEAK_PRICE = 1.0
MID_PRICE = 0.6
VALLEY_PRICE = 0.3

# ========== 电价时段 ==========
PRICE_PERIODS = {
    "peak":   [(8, 11), (18, 21)],
    "mid":    [(6, 8), (11, 18), (21, 22)],
    "valley": [(0, 6), (22, 24)],
}

# ========== 预测选项 ==========
PREDICTION_OPTIONS = {
    "1 小时": 4,
    "3 小时": 12,
    "6 小时": 24,
    "12 小时": 48,
}
```

---

## 十二、UI 交互规范

| 特性 | 规范 |
|------|------|
| 主题切换 | Dark / Light 模式切换按钮，状态通过 JS 管理 `body.classList` |
| 响应式布局 | `max-width: 1400px`，指标卡片 4 列网格布局 |
| 颜色方案 | 光伏橙色 (#ff9800)、负荷蓝色 (#2196f3)、盈余绿色 (#4caf50)、缺口红色 (#f44336) |
| 图表库 | Plotly（交互式图表，支持悬停提示、缩放） |
| 数据表格 | Gradio DataFrame 组件，支持排序和滚动 |

---

## 十三、版本历史

| 版本 | 日期 | 主要变更 |
|------|------|----------|
| v4.0 | — | 基于 Test 文件夹实现重构，剔除不可实现功能 |
| v5.0 | 2026-05 | 修正模型架构描述（光伏 LSTM 非 CNN-LSTM）、更新输入维度（7/6）、对齐 aligned CSV 列名、精简策略建议 |
| v5.1 | 2026-05 | 新增误差分析 Tab（回测 + 残差分布 + 光伏模型信息）、更新架构图为 5 Tab、补充精选特征测试集的回测用途 |