/* ============================================================
   光储充智能预测系统 — ECharts 图表管理
   ============================================================ */

var Charts = (function() {
  'use strict';

  var mainChart = null;
  var weatherChart = null;
  var priceChart = null;

  /* ---------- 获取 ECharts 主题色 ---------- */
  function isDark() {
    return document.documentElement.getAttribute('data-theme') === 'dark';
  }

  function getColors() {
    return isDark() ? {
      solar: '#FBBF24',
      load: '#60A5FA',
      band: 'rgba(96,165,250,0.12)',
      green: '#34D399',
      text: '#94A3B8',
      axis: '#334155',
      split: '#1E293B',
    } : {
      solar: '#F59E0B',
      load: '#3B82F6',
      band: 'rgba(59,130,246,0.12)',
      green: '#10B981',
      text: '#64748B',
      axis: '#E2E8F0',
      split: '#F1F5F9',
    };
  }

  /* ---------- 初始化图表实例 ---------- */
  function initAll() {
    var mainEl = document.getElementById('main-chart');
    var weatherEl = document.getElementById('weather-chart');
    var priceEl = document.getElementById('price-chart');

    if (mainEl && !mainChart) {
      mainChart = echarts.init(mainEl);
    }
    if (weatherEl && !weatherChart) {
      weatherChart = echarts.init(weatherEl);
    }
    if (priceEl && !priceChart) {
      priceChart = echarts.init(priceEl);
    }
  }

  /* ---------- 更新主预测图表 ---------- */
  function updateMainChart(prediction, view) {
    if (!mainChart) initAll();
    if (!mainChart) return;

    view = view || 'combined';
    var colors = getColors();
    var t = prediction.times || [];
    var solar = prediction.solar || [];
    var loadMean = prediction.load_mean || [];
    var loadLower = prediction.load_lower || [];
    var loadUpper = prediction.load_upper || [];

    var series = [];
    var yAxis = [];

    if (view === 'combined' || view === 'solar') {
      series.push({
        name: '光伏出力 (kW)',
        type: 'line',
        data: solar,
        smooth: true,
        symbol: 'none',
        lineStyle: { color: colors.solar, width: 2.5 },
        areaStyle: {
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: colors.solar + '40' },
            { offset: 1, color: colors.solar + '05' },
          ]),
        },
      });
    }

    if (view === 'combined' || view === 'load') {
      series.push({
        name: '充电负荷 (kW)',
        type: 'line',
        data: loadMean,
        smooth: true,
        symbol: 'none',
        lineStyle: { color: colors.load, width: 2.5 },
        areaStyle: {
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: colors.load + '40' },
            { offset: 1, color: colors.load + '05' },
          ]),
        },
      });

      /* 置信区间 */
      if (loadLower.length && loadUpper.length) {
        series.push({
          name: '置信区间',
          type: 'line',
          data: loadUpper,
          smooth: true,
          symbol: 'none',
          lineStyle: { color: 'transparent', width: 0 },
          areaStyle: { color: 'transparent' },
          stack: 'confidence',
          silent: true,
        });
        /* 下限用透明线叠加上去形成 band */
        series.push({
          name: '置信区间底',
          type: 'line',
          data: loadLower,
          smooth: true,
          symbol: 'none',
          lineStyle: { color: 'transparent', width: 0 },
          areaStyle: { color: colors.band },
          stack: 'confidence',
          silent: true,
        });
      }
    }

    var option = {
      tooltip: {
        trigger: 'axis',
        backgroundColor: isDark() ? '#1E293B' : '#FFF',
        borderColor: isDark() ? '#334155' : '#E2E8F0',
        textStyle: { color: isDark() ? '#F1F5F9' : '#1E293B', fontSize: 13 },
        formatter: function(params) {
          var s = '<b>' + params[0].axisValue + '</b><br/>';
          params.forEach(function(p) {
            if (p.seriesName.indexOf('置信') < 0) {
              s += '<span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:' + p.color + ';margin-right:6px;"></span>';
              s += p.seriesName + ': <b>' + (p.value != null ? p.value.toFixed(2) : '--') + ' kW</b><br/>';
            }
          });
          return s;
        },
      },
      legend: {
        bottom: 0,
        textStyle: { color: colors.text, fontSize: 12 },
        data: view === 'combined'
          ? ['光伏出力 (kW)', '充电负荷 (kW)']
          : (view === 'solar' ? ['光伏出力 (kW)'] : ['充电负荷 (kW)']),
      },
      grid: { left: 50, right: 30, top: 20, bottom: 40 },
      xAxis: {
        type: 'category',
        data: t,
        axisLine: { lineStyle: { color: colors.axis } },
        axisLabel: { color: colors.text, fontSize: 11 },
        splitLine: { show: false },
      },
      yAxis: {
        type: 'value',
        name: '功率 (kW)',
        nameTextStyle: { color: colors.text, fontSize: 11 },
        axisLabel: { color: colors.text, fontSize: 11 },
        splitLine: { lineStyle: { color: colors.split, type: 'dashed' } },
      },
      series: series,
    };

    mainChart.setOption(option, true);
  }

  /* ---------- 更新气象图表 ---------- */
  function updateWeatherChart(weather) {
    if (!weatherChart) initAll();
    if (!weatherChart) return;

    var colors = getColors();
    var dt = weather.day_times || [];
    var rads = weather.day_radiations || [];
    var clouds = weather.day_cloudcovers || [];

    var option = {
      tooltip: {
        trigger: 'axis',
        backgroundColor: isDark() ? '#1E293B' : '#FFF',
        borderColor: isDark() ? '#334155' : '#E2E8F0',
        textStyle: { color: isDark() ? '#F1F5F9' : '#1E293B', fontSize: 13 },
      },
      legend: {
        bottom: 0,
        textStyle: { color: colors.text, fontSize: 12 },
        data: ['辐射 (W/m²)', '云量 (%)'],
      },
      grid: { left: 50, right: 60, top: 20, bottom: 40 },
      xAxis: {
        type: 'category',
        data: dt,
        axisLine: { lineStyle: { color: colors.axis } },
        axisLabel: { color: colors.text, fontSize: 11 },
      },
      yAxis: [
        {
          type: 'value',
          name: '辐射 (W/m²)',
          nameTextStyle: { color: colors.text, fontSize: 11 },
          axisLabel: { color: colors.text, fontSize: 11 },
          splitLine: { lineStyle: { color: colors.split, type: 'dashed' } },
        },
        {
          type: 'value',
          name: '云量 (%)',
          nameTextStyle: { color: colors.text, fontSize: 11 },
          axisLabel: { color: colors.text, fontSize: 11 },
          splitLine: { show: false },
          max: 100,
        },
      ],
      series: [
        {
          name: '辐射 (W/m²)',
          type: 'bar',
          data: rads,
          itemStyle: {
            color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
              { offset: 0, color: '#F59E0B' },
              { offset: 1, color: '#D97706' },
            ]),
            borderRadius: [4, 4, 0, 0],
          },
        },
        {
          name: '云量 (%)',
          type: 'line',
          data: clouds,
          smooth: true,
          symbol: 'circle',
          symbolSize: 4,
          yAxisIndex: 1,
          lineStyle: { color: colors.text, width: 2 },
          itemStyle: { color: colors.text },
        },
      ],
    };

    weatherChart.setOption(option, true);
  }

  /* ---------- 更新电价图表 ---------- */
  function updatePriceChart(priceCurve) {
    if (!priceChart) initAll();
    if (!priceChart) return;

    var colors = getColors();
    var labels = priceCurve.labels || [];
    var values = priceCurve.values || [];

    var coloredData = values.map(function(v) {
      if (v > 0.8) return { value: v, itemStyle: { color: '#EF4444' } };
      if (v > 0.4) return { value: v, itemStyle: { color: '#F59E0B' } };
      return { value: v, itemStyle: { color: '#10B981' } };
    });

    var option = {
      tooltip: {
        trigger: 'axis',
        formatter: function(params) {
          return params[0].axisValue + '<br/>电价: <b>' + params[0].value + ' 元/kWh</b>';
        },
        backgroundColor: isDark() ? '#1E293B' : '#FFF',
        borderColor: isDark() ? '#334155' : '#E2E8F0',
        textStyle: { color: isDark() ? '#F1F5F9' : '#1E293B', fontSize: 13 },
      },
      grid: { left: 10, right: 15, top: 10, bottom: 10 },
      xAxis: {
        type: 'category',
        data: labels,
        show: false,
      },
      yAxis: {
        type: 'value',
        show: false,
        min: 0,
        max: 1,
      },
      series: [{
        type: 'bar',
        data: coloredData,
        barWidth: '80%',
      }],
    };

    priceChart.setOption(option, true);
  }

  /* ---------- 响应窗口大小变化 ---------- */
  function resize() {
    if (mainChart) mainChart.resize();
    if (weatherChart) weatherChart.resize();
    if (priceChart) priceChart.resize();
  }

  /* ---------- 主题切换时重绘 ---------- */
  function refreshTheme() {
    if (mainChart && mainChart.getOption()) {
      var opt = mainChart.getOption();
      mainChart.setOption(opt, true);
    }
    if (weatherChart && weatherChart.getOption()) {
      var wopt = weatherChart.getOption();
      weatherChart.setOption(wopt, true);
    }
    if (priceChart && priceChart.getOption()) {
      var popt = priceChart.getOption();
      priceChart.setOption(popt, true);
    }
  }

  /* 注册 resize 监听 */
  window.addEventListener('resize', resize);

  /* 暴露公共方法 */
  return {
    initAll: initAll,
    updateMainChart: updateMainChart,
    updateWeatherChart: updateWeatherChart,
    updatePriceChart: updatePriceChart,
    resize: resize,
    refreshTheme: refreshTheme,
    isDark: isDark,
  };
})();