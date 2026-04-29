"""验证 app.py 能否正常创建 Gradio 应用（不启动服务器）"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

# 先测试导入
from model_service import solar_forecast_service, charging_forecast_service
from weather_service import update_weather_ui, get_weather_data

print("[OK] 所有模块导入成功")

# 测试模型服务接口
weather_data = get_weather_data()
print(f"[OK] 天气数据获取: radiation={weather_data['current_radiation']:.0f}")

solar_result = solar_forecast_service.predict(forecast_hours=4, weather_data=weather_data)
print(f"[OK] 光伏预测: {len(solar_result['predictions'])} 点, 峰值={max(solar_result['predictions']):.1f}kW")

charging_result = charging_forecast_service.predict(forecast_hours=4)
print(f"[OK] 充电预测: {len(charging_result['predictions'])} 点, 峰值={max(charging_result['predictions']):.1f}kW")

# 测试天气UI更新
rad_plot, warn, df, info = update_weather_ui()
print("[OK] weather_service.update_weather_ui() 调用成功")

# 测试 Gradio 应用创建
print("\n正在创建 Gradio 应用...")
import importlib.util
spec = importlib.util.spec_from_file_location("test_app", os.path.join(os.path.dirname(__file__), "app.py"))
test_app_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(test_app_module)
create_app = test_app_module.create_app
app = create_app()
print(f"[OK] Gradio 应用创建成功! 名称: {app.title}")

# 检查 Blocks 结构
blocks = list(app.children)
print(f"[OK] 应用包含 {len(blocks)} 个顶层组件")

# 检查是否有 Tabs
tabs_count = sum(1 for b in blocks if b.__class__.__name__ == 'Tabs')
print(f"[OK] 应用包含 {tabs_count} 个 Tabs 组件")

print("\n" + "=" * 50)
print("  所有验证通过!")
print("=" * 50)