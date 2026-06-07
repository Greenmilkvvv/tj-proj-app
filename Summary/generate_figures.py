"""
生成 Test8 说明书所需的所有小图
================================
调用 Test8 中的 data_service / weather_service / prediction_service 函数，
将图表保存为 .jpg 到 Summary/figures/ 目录下。
"""
import os
import sys

# 确保可以 import Test8 的模块
TEST8_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Test8")
sys.path.insert(0, TEST8_DIR)

FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

import plotly.graph_objects as go
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


def save_fig(fig, name):
    """保存 Plotly Figure 为 .jpg"""
    if fig is None:
        print(f"  [SKIP] {name} — fig is None")
        return
    path = os.path.join(FIGURES_DIR, name)
    try:
        fig.write_image(path, format="jpg", scale=2, width=1200, height=None)
        print(f"  [OK] {name}")
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")


# ============================================================
# 1. 联合预测图表（需要先跑预测）
# ============================================================
def generate_prediction_charts():
    """生成联合预测的主图和摘要图"""
    from data_service import create_prediction_chart, create_summary_chart
    from prediction_service import run_prediction

    print("\n[1] 联合预测图表...")

    # 运行一次预测 (24h, 默认参数)
    try:
        weather = {"radiation": 400}
        result = run_prediction(96, weather, current_price=0.63, current_load=875.0)
    except Exception as e:
        print(f"  [WARN] run_prediction 失败: {e}")
        # 构造一个 fallback 图表
        fig = go.Figure()
        fig.add_annotation(x=0.5, y=0.5, text="联合预测图表（需后端运行）",
                           showarrow=False, font=dict(size=14, color="#888"))
        fig.update_layout(height=550, template="plotly_white")
        save_fig(fig, "fig01_joint_prediction.jpg")
        fig2 = go.Figure()
        fig2.add_annotation(x=0.5, y=0.5, text="联合预测摘要图（需后端运行）",
                            showarrow=False, font=dict(size=14, color="#888"))
        fig2.update_layout(height=350, template="plotly_white")
        save_fig(fig2, "fig02_joint_summary.jpg")
        return

    # 主预测图
    fig1 = create_prediction_chart(result)
    save_fig(fig1, "fig01_joint_prediction.jpg")

    # 摘要图
    fig2 = create_summary_chart(result)
    save_fig(fig2, "fig02_joint_summary.jpg")


# ============================================================
# 2. 数据探索图表
# ============================================================
def generate_data_exploration_charts():
    """生成数据探索 Tab 的三张图"""
    from data_service import (
        plot_aggregated_load,
        get_correlation_chart,
        get_hourly_profile_chart,
    )

    print("\n[2] 数据探索图表...")

    # 从数据集推断日期范围
    from data_service import load_builtin_data
    _, charging_df = load_builtin_data()
    if charging_df is not None:
        time_col = "timestamp" if "timestamp" in charging_df.columns else "datetime"
        dmin = str(charging_df[time_col].min())[:10]
        dmax = str(charging_df[time_col].max())[:10]
    else:
        dmin, dmax = "2026-01-01", "2026-01-30"

    # 聚合日负荷曲线
    fig3 = plot_aggregated_load(dmin, dmax)
    save_fig(fig3, "fig03_aggregated_load.jpg")

    # Pearson 相关系数
    fig4 = get_correlation_chart()
    save_fig(fig4, "fig04_correlation.jpg")

    # 小时级负荷画像
    fig5 = get_hourly_profile_chart()
    save_fig(fig5, "fig05_hourly_profile.jpg")


# ============================================================
# 3. 误差分析图表
# ============================================================
def generate_error_analysis_charts():
    """生成误差分析 Tab 的六张图"""
    from data_service import (
        run_backtest_charging,
        run_backtest_solar,
        build_charging_error_distribution_chart,
        build_solar_error_distribution_chart,
        build_charging_error_by_hour_chart,
        build_solar_error_by_hour_chart,
    )

    print("\n[3] 误差分析图表...")

    # 充电回测
    fig6, summary6 = run_backtest_charging()
    save_fig(fig6, "fig06_charging_backtest.jpg")

    # 光伏回测
    fig7, summary7 = run_backtest_solar()
    save_fig(fig7, "fig07_solar_backtest.jpg")

    # 充电残差分布
    fig8 = build_charging_error_distribution_chart()
    save_fig(fig8, "fig08_charging_error_dist.jpg")

    # 光伏残差分布
    fig9 = build_solar_error_distribution_chart()
    save_fig(fig9, "fig09_solar_error_dist.jpg")

    # 充电按小时误差
    fig10 = build_charging_error_by_hour_chart()
    save_fig(fig10, "fig10_charging_error_hourly.jpg")

    # 光伏按小时误差
    fig11 = build_solar_error_by_hour_chart()
    save_fig(fig11, "fig11_solar_error_hourly.jpg")


# ============================================================
# 4. 气象数据图表
# ============================================================
def generate_weather_chart():
    """生成气象 Tab 的辐照度趋势图"""
    from weather_service import fetch_weather_data, build_radiation_chart

    print("\n[4] 气象数据图表...")

    try:
        wd = fetch_weather_data()
        fig12 = build_radiation_chart(wd)
    except Exception as e:
        print(f"  [WARN] 气象数据获取失败: {e}")
        fig12 = go.Figure()
        fig12.add_annotation(x=0.5, y=0.5, text="气象趋势图\n（需网络连接获取实时数据）",
                             showarrow=False, font=dict(size=14, color="#888"))
        fig12.update_layout(height=350, template="plotly_white")

    save_fig(fig12, "fig12_radiation_trend.jpg")


# ============================================================
# 主流程
# ============================================================
def main():
    print("=" * 60)
    print("Test8 说明书图片生成器")
    print(f"输出目录: {FIGURES_DIR}")
    print("=" * 60)

    generate_prediction_charts()
    generate_data_exploration_charts()
    generate_error_analysis_charts()
    generate_weather_chart()

    print("\n" + "=" * 60)
    print("所有图片生成完毕！")
    print(f"文件位于: {FIGURES_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()