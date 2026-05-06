"""诊断模型加载问题"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Solar_Forecast"))

print("=" * 50)
print("1. 检查导入")
try:
    from NN import GeneratorWithFeatures, CNN_LSTM
    print("   GeneratorWithFeatures: OK")
    print("   CNN_LSTM: OK")
except Exception as e:
    print(f"   导入失败: {e}")

try:
    import torch
    print("   torch: OK")
except:
    print("   torch: 不可用")

print("\n2. 检查光伏模型路径")
pth = r"d:\AAA_proj_tongji\tj-proj-app\Solar_Forecast\best_pth\best_generator.pth"
print(f"   路径: {pth}")
print(f"   存在: {os.path.exists(pth)}")

if os.path.exists(pth):
    try:
        d = torch.load(pth, map_location='cpu', weights_only=True)
        keys = list(d.keys())
        print(f"   state_dict keys (前10): {keys[:10]}")
        print(f"   state_dict keys 总数: {len(keys)}")
        
        # 尝试用 GeneratorWithFeatures 加载
        try:
            m = GeneratorWithFeatures(input_size=5, hidden_size=128, num_layers=2, output_size=1, dropout=0.2, bidirectional=False, cnn_channels=[64,64], kernel_size=3)
            m.load_state_dict(d)
            print("   -> GeneratorWithFeatures 加载成功!")
        except Exception as e:
            print(f"   -> GeneratorWithFeatures 加载失败: {e}")
        
        # 尝试用 CNN_LSTM 加载
        try:
            m2 = CNN_LSTM(input_size=5, hidden_size=128, num_layers=2, output_size=1, dropout=0.2, bidirectional=False, cnn_channels=[64,64], kernel_size=3)
            m2.load_state_dict(d)
            print("   -> CNN_LSTM 加载成功!")
        except Exception as e:
            print(f"   -> CNN_LSTM 加载失败: {e}")
    except Exception as e:
        print(f"   加载pth失败: {e}")

print("\n3. 检查充电模型路径")
pth2 = r"d:\AAA_proj_tongji\tj-proj-app\Charging_Forecast\best_pth\final_best_hybrid_model.pth"
print(f"   路径: {pth2}")
print(f"   存在: {os.path.exists(pth2)}")

if os.path.exists(pth2):
    try:
        d2 = torch.load(pth2, map_location='cpu', weights_only=True)
        keys2 = list(d2.keys())
        print(f"   state_dict keys (前15): {keys2[:15]}")
        print(f"   state_dict keys 总数: {len(keys2)}")
        
        # 分析 key 结构
        top_keys = set()
        for k in keys2:
            top = k.split('.')[0]
            top_keys.add(top)
        print(f"   顶层模块: {sorted(top_keys)}")
    except Exception as e:
        print(f"   加载pth失败: {e}")

print("=" * 50)