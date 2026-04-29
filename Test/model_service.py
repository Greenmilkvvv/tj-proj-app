"""
模型服务模块 - 封装光伏预测和充电负荷预测模型的加载与推理
支持真实模型加载和模拟数据回退
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import sys

# 尝试加载 PyTorch
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[WARNING] PyTorch 不可用，将使用模拟数据")

# 添加项目根目录到路径，以便导入 NN 模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Solar_Forecast'))

# ==================== 配置常量 ====================
TRANSFORMER_CAPACITY_KW = 500  # 变压器额定容量 (kW)
PEAK_PRICE = 1.28  # 峰时电价 (元/kWh)
MID_PRICE = 0.85   # 平时电价 (元/kWh)
VALLEY_PRICE = 0.35  # 谷时电价 (元/kWh)

# 电价时段定义
PRICE_PERIODS = {
    "peak": [(8, 12), (17, 21)],     # 峰时段
    "mid": [(12, 17), (21, 22)],     # 平时段
    "valley": [(22, 24), (0, 8)],    # 谷时段
}


def get_current_price_period():
    """根据当前时间返回电价时段"""
    now = datetime.now()
    hour = now.hour
    for period, ranges in PRICE_PERIODS.items():
        for start, end in ranges:
            if start <= hour < end:
                return period
    return "mid"


def get_price_by_period(period):
    """根据时段返回电价"""
    prices = {"peak": PEAK_PRICE, "mid": MID_PRICE, "valley": VALLEY_PRICE}
    return prices.get(period, MID_PRICE)


def get_price_tag(period):
    """返回电价时段的中文标签"""
    tags = {"peak": "🔴 峰时段", "mid": "🟡 平时段", "valley": "🟢 谷时段"}
    return tags.get(period, "⚪ 未知")


# ==================== 光伏预测模型 ====================
class SolarPredictor:
    """光伏出力预测模型封装"""
    
    def __init__(self, model_path=None):
        self.model = None
        self.model_loaded = False
        self.device = torch.device('cuda' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu')
        
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), '..', 'Solar_Forecast', 'best_pth', 'best_generator.pth')
        
        if os.path.exists(model_path) and TORCH_AVAILABLE:
            try:
                # 尝试加载 Solar_Forecast.NN 中的模型
                from NN import LSTMPredictor, CNN_LSTM, GeneratorWithFeatures
                
                # 创建与训练时相同的模型结构
                # 根据 forecast_LSTM_GAN.ipynb 中的配置
                input_size = 5  # 辐照度、云量、温度、风速、时间特征
                hidden_size = 64
                num_layers = 2
                
                base_model = CNN_LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    output_size=1,
                    dropout=0.2,
                    bidirectional=False,
                    cnn_channels=[64, 64]
                )
                
                self.model = GeneratorWithFeatures(base_model)
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.to(self.device)
                self.model.eval()
                self.model_loaded = True
                print(f"[INFO] 光伏预测模型已加载: {model_path}")
            except Exception as e:
                print(f"[WARNING] 光伏模型加载失败 ({e})，将使用模拟数据")
                self.model_loaded = False
        else:
            if not TORCH_AVAILABLE:
                print("[INFO] PyTorch 不可用，光伏模型使用模拟数据")
            else:
                print(f"[INFO] 模型文件未找到: {model_path}，使用模拟数据")
    
    def predict(self, weather_data, lookback=96):
        """
        预测未来24小时光伏出力 (96个15分钟点)
        
        Args:
            weather_data: dict with radiation, cloudcover, temperature, etc.
            lookback: 输入序列长度
        
        Returns:
            dict with timestamps, predictions, confidence_lower, confidence_upper
        """
        now = datetime.now().replace(minute=0, second=0, microsecond=0)
        future_times = [now + timedelta(minutes=15*i) for i in range(96)]
        
        if self.model_loaded and TORCH_AVAILABLE:
            try:
                # 构建输入序列 (用历史+预报数据)
                # 这里简化处理：使用天气数据构建特征序列
                features = self._build_features(weather_data, lookback)
                with torch.no_grad():
                    input_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
                    predictions = self.model(input_tensor).cpu().numpy().flatten()
                
                # 生成置信区间 (基于经验噪声)
                std = np.std(predictions) * 0.1
                confidence_upper = predictions + 1.96 * std
                confidence_lower = np.clip(predictions - 1.96 * std, 0, None)
                
                return {
                    "timestamps": future_times,
                    "predictions": predictions.tolist(),
                    "confidence_lower": confidence_lower.tolist(),
                    "confidence_upper": confidence_upper.tolist(),
                    "peak_value": float(np.max(predictions)),
                    "peak_time": future_times[np.argmax(predictions)],
                    "total_energy": float(np.sum(predictions) * 0.25),  # kWh
                }
            except Exception as e:
                print(f"[ERROR] 光伏预测推理失败: {e}")
        
        # 回退到模拟数据
        return self._mock_predict(future_times, weather_data)
    
    def _build_features(self, weather_data, lookback):
        """构建模型输入特征"""
        # 简化：用天气数据填充 lookback 长度的序列
        radiation = weather_data.get("current_radiation", 500)
        cloudcover = weather_data.get("current_cloudcover", 30)
        temperature = weather_data.get("temperature", 25)
        
        features = np.zeros((lookback, 5))
        features[:, 0] = radiation
        features[:, 1] = cloudcover
        features[:, 2] = temperature
        features[:, 3] = np.sin(np.linspace(0, 2*np.pi, lookback))  # 模拟日周期
        features[:, 4] = np.cos(np.linspace(0, 2*np.pi, lookback))
        return features
    
    def _mock_predict(self, future_times, weather_data):
        """生成模拟光伏预测数据"""
        radiation = weather_data.get("current_radiation", 500)
        cloudcover = weather_data.get("current_cloudcover", 30)
        cloud_factor = 1 - cloudcover / 100 * 0.7
        
        predictions = []
        for t in future_times:
            hour = t.hour + t.minute / 60
            # 模拟光伏出力曲线：正弦形状，峰值在中午
            if 6 <= hour <= 18:
                base = radiation * cloud_factor * np.sin(np.pi * (hour - 6) / 12)
                noise = np.random.normal(0, base * 0.05)
                pred = max(0, base + noise)
            else:
                pred = max(0, np.random.normal(0, 5))
            predictions.append(pred)
        
        predictions = np.array(predictions)
        std = np.std(predictions) * 0.15
        confidence_upper = predictions + 1.96 * std
        confidence_lower = np.clip(predictions - 1.96 * std, 0, None)
        
        return {
            "timestamps": future_times,
            "predictions": predictions.tolist(),
            "confidence_lower": confidence_lower.tolist(),
            "confidence_upper": confidence_upper.tolist(),
            "peak_value": float(np.max(predictions)),
            "peak_time": future_times[np.argmax(predictions)],
            "total_energy": float(np.sum(predictions) * 0.25),
        }


# ==================== 充电负荷预测模型 ====================
class ChargingLoadPredictor:
    """充电负荷预测模型封装"""
    
    def __init__(self, model_path=None):
        self.model = None
        self.model_loaded = False
        self.device = torch.device('cuda' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu')
        
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), '..', 'Charging_Forecast', 'best_pth', 'final_best_hybrid_model.pth')
        
        if os.path.exists(model_path) and TORCH_AVAILABLE:
            try:
                # 尝试加载模型
                # 充电负荷模型是 Bys-TCN-Attention-LSTM
                # 由于模型结构复杂，这里简化处理
                from NN import CNN_LSTM_Advanced
                
                # 根据数据集特征列
                input_size = 6  # price, lag_1, lag_96, lag_672, rolling_std_4, rolling_mean_4
                hidden_size = 64
                num_layers = 2
                
                self.model = CNN_LSTM_Advanced(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    output_size=1,
                    dropout=0.2,
                    bidirectional=False,
                    cnn_channels=[64, 128],
                    use_attention=True
                )
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.to(self.device)
                self.model.eval()
                self.model_loaded = True
                print(f"[INFO] 充电负荷模型已加载: {model_path}")
            except Exception as e:
                print(f"[WARNING] 充电负荷模型加载失败 ({e})，将使用模拟数据")
                self.model_loaded = False
        else:
            if not TORCH_AVAILABLE:
                print("[INFO] PyTorch 不可用，充电负荷模型使用模拟数据")
            else:
                print(f"[INFO] 模型文件未找到: {model_path}，使用模拟数据")
        
        # 加载历史数据用于构建输入特征
        self._load_historical_data()
    
    def _load_historical_data(self):
        """加载历史数据用于特征构建"""
        data_path = os.path.join(os.path.dirname(__file__), '..', 'Data', 'dataset_selected_features_test.csv')
        if os.path.exists(data_path):
            try:
                self.historical_df = pd.read_csv(data_path)
                self.historical_df['timestamp'] = pd.to_datetime(self.historical_df['timestamp'])
                print(f"[INFO] 已加载历史数据: {len(self.historical_df)} 行")
            except Exception as e:
                print(f"[WARNING] 历史数据加载失败: {e}")
                self.historical_df = None
        else:
            print(f"[WARNING] 历史数据文件未找到: {data_path}")
            self.historical_df = None
    
    def predict(self, current_load=None, lookback=96):
        """
        预测未来24小时充电负荷 (96个15分钟点)
        
        Args:
            current_load: 当前负荷值 (kW)，None则自动获取
            lookback: 输入序列长度
        
        Returns:
            dict with timestamps, predictions, confidence_lower, confidence_upper
        """
        now = datetime.now().replace(minute=0, second=0, microsecond=0)
        future_times = [now + timedelta(minutes=15*i) for i in range(96)]
        
        if self.model_loaded and self.historical_df is not None and TORCH_AVAILABLE:
            try:
                features = self._build_features(lookback)
                if features is not None:
                    with torch.no_grad():
                        input_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
                        predictions = self.model(input_tensor).cpu().numpy().flatten()
                    
                    std = np.std(predictions) * 0.1
                    confidence_upper = predictions + 1.96 * std
                    confidence_lower = np.clip(predictions - 1.96 * std, 0, None)
                    
                    return {
                        "timestamps": future_times,
                        "predictions": predictions.tolist(),
                        "confidence_lower": confidence_lower.tolist(),
                        "confidence_upper": confidence_upper.tolist(),
                        "peak_value": float(np.max(predictions)),
                        "peak_time": future_times[np.argmax(predictions)],
                        "total_energy": float(np.sum(predictions) * 0.25),
                    }
            except Exception as e:
                print(f"[ERROR] 充电负荷预测推理失败: {e}")
        
        # 回退到模拟数据
        return self._mock_predict(future_times, current_load)
    
    def _build_features(self, lookback):
        """从历史数据构建输入特征"""
        if self.historical_df is None:
            return None
        try:
            # 取最近的 lookback 行
            recent = self.historical_df.tail(lookback)
            feature_cols = ['price', 'lag_1', 'lag_96', 'lag_672', 'rolling_std_4', 'rolling_mean_4']
            features = recent[feature_cols].values.astype(np.float32)
            return features
        except Exception as e:
            print(f"[ERROR] 特征构建失败: {e}")
            return None
    
    def _mock_predict(self, future_times, current_load=None):
        """生成模拟充电负荷预测数据"""
        if current_load is None:
            current_load = 150.0
        
        predictions = []
        for t in future_times:
            hour = t.hour + t.minute / 60
            # 模拟充电负荷曲线：早晚高峰
            if 7 <= hour <= 9:  # 早高峰
                base = current_load * 1.3
            elif 11 <= hour <= 13:  # 午间
                base = current_load * 0.9
            elif 17 <= hour <= 20:  # 晚高峰
                base = current_load * 1.5
            elif 22 <= hour <= 24 or 0 <= hour <= 6:  # 夜间
                base = current_load * 0.5
            else:
                base = current_load * 0.8
            
            noise = np.random.normal(0, base * 0.08)
            pred = max(0, base + noise)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        std = np.std(predictions) * 0.12
        confidence_upper = predictions + 1.96 * std
        confidence_lower = np.clip(predictions - 1.96 * std, 0, None)
        
        return {
            "timestamps": future_times,
            "predictions": predictions.tolist(),
            "confidence_lower": confidence_lower.tolist(),
            "confidence_upper": confidence_upper.tolist(),
            "peak_value": float(np.max(predictions)),
            "peak_time": future_times[np.argmax(predictions)],
            "total_energy": float(np.sum(predictions) * 0.25),
        }


# ==================== 综合预测服务 ====================
class PredictionService:
    """综合预测服务，协调光伏和负荷预测"""
    
    def __init__(self):
        self.solar_predictor = SolarPredictor()
        self.charging_predictor = ChargingLoadPredictor()
    
    def get_all_predictions(self, weather_data, current_load=None):
        """获取完整的预测结果"""
        solar_result = self.solar_predictor.predict(weather_data)
        charging_result = self.charging_predictor.predict(current_load)
        
        # 计算净负荷
        solar_preds = np.array(solar_result["predictions"])
        charging_preds = np.array(charging_result["predictions"])
        net_load = charging_preds - solar_preds
        
        # 计算关键指标
        now = datetime.now()
        period = get_current_price_period()
        price = get_price_by_period(period)
        
        # 绿电替代率 (光伏发电占充电总量的比例)
        total_charging = np.sum(charging_preds) * 0.25  # kWh
        total_solar = np.sum(solar_preds) * 0.25
        green_ratio = min(total_solar / total_charging * 100, 100) if total_charging > 0 else 0
        
        # 节省电费
        saved_cost = total_solar * price  # 假设光伏直供节省了电网购电
        
        return {
            "solar": solar_result,
            "charging": charging_result,
            "net_load": net_load.tolist(),
            "current_price_period": period,
            "current_price": price,
            "price_tag": get_price_tag(period),
            "green_ratio": green_ratio,
            "saved_cost": saved_cost,
            "transformer_capacity": TRANSFORMER_CAPACITY_KW,
            "capacity_ratio": float(np.max(charging_preds) / TRANSFORMER_CAPACITY_KW * 100),
        }


# ==================== 预警与策略 ====================
class AlertService:
    """预警与策略服务"""
    
    def __init__(self, prediction_service):
        self.prediction_service = prediction_service
    
    def check_alerts(self, predictions):
        """检查预警条件"""
        alerts = []
        
        charging_preds = np.array(predictions["charging"]["predictions"])
        solar_preds = np.array(predictions["solar"]["predictions"])
        net_load = np.array(predictions["net_load"])
        
        # 1. 负荷超阈值预警 (变压器80%容量)
        capacity_80 = TRANSFORMER_CAPACITY_KW * 0.8
        future_1h = charging_preds[:4]  # 未来1小时 (4个15分钟点)
        if np.max(future_1h) > capacity_80:
            peak_idx = np.argmax(future_1h)
            peak_time = predictions["charging"]["timestamps"][peak_idx]
            alerts.append({
                "type": "overload",
                "level": "danger",
                "message": f"⚠️ 未来1小时负荷预计达到 {np.max(future_1h):.0f}kW，超过变压器80%容量 ({capacity_80:.0f}kW)",
                "time": peak_time.strftime("%H:%M"),
            })
        
        # 2. 光伏盈余提醒
        future_4h_solar = solar_preds[:16]  # 未来4小时
        future_4h_charging = charging_preds[:16]
        surplus = future_4h_solar - future_4h_charging
        max_surplus = np.max(surplus)
        if max_surplus > 50:
            surplus_idx = np.argmax(surplus)
            surplus_time = predictions["solar"]["timestamps"][surplus_idx]
            alerts.append({
                "type": "solar_surplus",
                "level": "info",
                "message": f"💡 预计 {surplus_time.strftime('%H:%M')} 光伏盈余 {max_surplus:.0f}kW，建议开启储能充电或有序充电",
                "time": surplus_time.strftime("%H:%M"),
            })
        
        # 3. 傍晚光伏为零提醒
        evening_start = 17
        evening_solar = [s for i, s in enumerate(solar_preds) 
                        if predictions["solar"]["timestamps"][i].hour >= evening_start]
        if evening_solar and np.mean(evening_solar) < 10:
            alerts.append({
                "type": "solar_zero",
                "level": "warning",
                "message": "🌙 预计傍晚光伏出力接近零，建议保留储能电量至18:00释放以应对充电高峰",
                "time": "18:00",
            })
        
        # 4. 数据链路检查
        if predictions["charging"]["peak_value"] < 1 or predictions["solar"]["peak_value"] < 1:
            alerts.append({
                "type": "data_missing",
                "level": "danger",
                "message": "🚨 数据链路异常！模型输入数据可能中断，预测结果不可靠",
                "time": datetime.now().strftime("%H:%M"),
            })
        
        return alerts
    
    def generate_strategy(self, predictions):
        """生成储能调度策略"""
        strategies = []
        
        solar_preds = np.array(predictions["solar"]["predictions"])
        charging_preds = np.array(predictions["charging"]["predictions"])
        net_load = np.array(predictions["net_load"])
        times = predictions["solar"]["timestamps"]
        
        # 按小时聚合
        hourly_data = {}
        for i, t in enumerate(times):
            hour_key = t.strftime("%H:00")
            if hour_key not in hourly_data:
                hourly_data[hour_key] = {"solar": 0, "charging": 0, "net": 0, "count": 0}
            hourly_data[hour_key]["solar"] += solar_preds[i]
            hourly_data[hour_key]["charging"] += charging_preds[i]
            hourly_data[hour_key]["net"] += net_load[i]
            hourly_data[hour_key]["count"] += 1
        
        for hour_key, data in hourly_data.items():
            avg_solar = data["solar"] / data["count"]
            avg_net = data["net"] / data["count"]
            hour = int(hour_key.split(":")[0])
            
            # 确定电价时段
            period = get_current_price_period_static(hour)
            
            if avg_net < -30:  # 光伏盈余
                strategies.append({
                    "time": hour_key,
                    "action": "🔋 储能充电",
                    "power": f"{abs(avg_net):.0f}",
                    "reason": "光伏盈余时段，消纳绿电",
                    "period": period,
                })
            elif avg_net > 80 and period == "peak":  # 峰时高负荷
                strategies.append({
                    "time": hour_key,
                    "action": "⚡ 储能放电",
                    "power": f"{min(avg_net * 0.5, 100):.0f}",
                    "reason": "电价高峰+负荷激增，储能放电支撑",
                    "period": period,
                })
            elif period == "valley" and avg_solar < 5:
                strategies.append({
                    "time": hour_key,
                    "action": "🔌 电网充电",
                    "power": "50",
                    "reason": "谷时电价，从电网充电备用",
                    "period": period,
                })
        
        return strategies


def get_current_price_period_static(hour):
    """静态电价时段判断"""
    if 8 <= hour < 12 or 17 <= hour < 21:
        return "peak"
    elif 12 <= hour < 17 or 21 <= hour < 22:
        return "mid"
    else:
        return "valley"


# ==================== Gradio 应用适配层 ====================
# 将预测器包装为 Gradio 应用所需的统一接口
# 接口: .predict(forecast_hours, weather_data=None) -> dict with times, predictions, confidence_*

class SolarForecastServiceAdapter:
    """光伏预测服务 - 适配 Gradio 接口"""
    
    def __init__(self):
        self._predictor = None
    
    @property
    def predictor(self):
        if self._predictor is None:
            self._predictor = SolarPredictor()
        return self._predictor
    
    def predict(self, forecast_hours=4, weather_data=None):
        """统一预测接口
        
        Args:
            forecast_hours: 预测时长 (小时)
            weather_data: 天气数据字典
        
        Returns:
            dict with times, predictions, confidence_lower, confidence_upper
        """
        if weather_data is None:
            weather_data = {
                "current_radiation": 500,
                "current_cloudcover": 30,
                "temperature": 25,
                "rain": 0,
                "wind_speed": 3,
            }
        
        # 转换为15分钟步数
        n_steps = int(forecast_hours * 4)
        result = self.predictor.predict(weather_data, lookback=96)
        
        # 截取需要的步数
        n_steps = min(n_steps, len(result["predictions"]))
        
        return {
            "times": result["timestamps"][:n_steps],
            "predictions": result["predictions"][:n_steps],
            "confidence_lower": result["confidence_lower"][:n_steps],
            "confidence_upper": result["confidence_upper"][:n_steps],
        }


class ChargingForecastServiceAdapter:
    """充电负荷预测服务 - 适配 Gradio 接口"""
    
    def __init__(self):
        self._predictor = None
    
    @property
    def predictor(self):
        if self._predictor is None:
            self._predictor = ChargingLoadPredictor()
        return self._predictor
    
    def predict(self, forecast_hours=4, current_load=None):
        """统一预测接口
        
        Args:
            forecast_hours: 预测时长 (小时)
            current_load: 当前负荷 (kW)
        
        Returns:
            dict with times, predictions, confidence_lower, confidence_upper
        """
        n_steps = int(forecast_hours * 4)
        result = self.predictor.predict(current_load=current_load, lookback=96)
        
        n_steps = min(n_steps, len(result["predictions"]))
        
        return {
            "times": result["timestamps"][:n_steps],
            "predictions": result["predictions"][:n_steps],
            "confidence_lower": result["confidence_lower"][:n_steps],
            "confidence_upper": result["confidence_upper"][:n_steps],
        }


# Gradio 全局服务实例
solar_forecast_service = SolarForecastServiceAdapter()
charging_forecast_service = ChargingForecastServiceAdapter()


# 创建全局单例 (保留旧接口兼容)
_solar_predictor = None
_charging_predictor = None
_prediction_service = None
_alert_service = None


def get_solar_predictor():
    global _solar_predictor
    if _solar_predictor is None:
        _solar_predictor = SolarPredictor()
    return _solar_predictor


def get_charging_predictor():
    global _charging_predictor
    if _charging_predictor is None:
        _charging_predictor = ChargingLoadPredictor()
    return _charging_predictor


def get_prediction_service():
    global _prediction_service
    if _prediction_service is None:
        _prediction_service = PredictionService()
    return _prediction_service


def get_alert_service():
    global _alert_service
    if _alert_service is None:
        _alert_service = AlertService(get_prediction_service())
    return _alert_service
