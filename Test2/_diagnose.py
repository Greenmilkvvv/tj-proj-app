"""诊断模型加载"""
import sys, os

# 输出到文件
log_file = os.path.join(os.path.dirname(__file__), "_diag.txt")
with open(log_file, "w", encoding="utf-8") as f:
    f.write("===== 诊断开始 =====\n")
    
    try:
        # 1. 导入
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Solar_Forecast"))
        from NN import GeneratorWithFeatures
        import torch
        f.write("1. 导入成功\n")
        
        # 2. 光伏模型
        solar_pth = r"d:\AAA_proj_tongji\tj-proj-app\Solar_Forecast\best_pth\best_generator.pth"
        f.write(f"2. 光伏pth存在: {os.path.exists(solar_pth)}\n")
        if os.path.exists(solar_pth):
            sd = torch.load(solar_pth, map_location="cpu", weights_only=True)
            ks = list(sd.keys())
            f.write(f"   keys数: {len(ks)}, 前3: {ks[:3]}\n")
            try:
                m = GeneratorWithFeatures(5, 128, 2, 1, 0.2, False, [64,64], 3)
                m.load_state_dict(sd)
                f.write("   GeneratorWithFeatures 加载OK\n")
            except Exception as e:
                f.write(f"   GeneratorWithFeatures 失败: {e}\n")
                # 尝试 CNN_LSTM
                try:
                    from NN import CNN_LSTM
                    m2 = CNN_LSTM(5, 128, 2, 1, 0.2, False, [64,64], 3)
                    m2.load_state_dict(sd)
                    f.write("   CNN_LSTM 加载OK\n")
                except Exception as e2:
                    f.write(f"   CNN_LSTM 也失败: {e2}\n")
        
        # 3. 充电模型
        charge_pth = r"d:\AAA_proj_tongji\tj-proj-app\Charging_Forecast\best_pth\final_best_hybrid_model.pth"
        f.write(f"3. 充电pth存在: {os.path.exists(charge_pth)}\n")
        if os.path.exists(charge_pth):
            sd2 = torch.load(charge_pth, map_location="cpu", weights_only=True)
            ks2 = list(sd2.keys())
            f.write(f"   keys数: {len(ks2)}, 前3: {ks2[:3]}\n")
            f.write(f"   keys全部: {ks2}\n")
            
    except Exception as e:
        import traceback
        f.write(f"异常: {e}\n{traceback.format_exc()}\n")
    
    f.write("===== 诊断结束 =====\n")

print("done")