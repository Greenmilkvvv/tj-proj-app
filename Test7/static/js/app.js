/**
 * 光储充智能预测 Demo v4 — 前端主逻辑
 * ==================================================
 * 基于 Test5 的 5-Tab 界面设计，纯 HTML+CSS+JS，
 * 集成 Plotly 图表库，通过 REST API 与 BentoML 后端通信。
 */

(function () {
  'use strict';

  // ============================================================
  // 常量
  // ============================================================
  // BentoML 服务地址 (开发时 localhost:3000, 部署后可修改)
  const API_BASE = window.location.origin + '/api';

  // 温度区间分级颜色
  const TEMP_COLORS = {
    cold: '#2196f3',
    mild: '#4caf50',
    warm: '#ff9800',
    hot: '#f44336',
  };

  function tempClass(temp) {
    if (temp == null) return 'mild';
    if (temp <= 5) return 'cold';
    if (temp <= 18) return 'mild';
    if (temp <= 30) return 'warm';
    return 'hot';
  }

  function tempColor(temp) {
    return TEMP_COLORS[tempClass(temp)] || TEMP_COLORS.mild;
  }

  // ============================================================
  // DOM 工具
  // ============================================================
  function $(sel, parent) {
    return (parent || document).querySelector(sel);
  }
  function $$(sel, parent) {
    return Array.from((parent || document).querySelectorAll(sel));
  }

  // ============================================================
  // 状态管理
  // ============================================================
  let currentTab = 'tab-weather';
  let lastWeatherData = null;
  let lastPredictionResult = null;

  // ============================================================
  // 工具函数
  // ============================================================

  /** 设置状态指示器 */
  function setStatus(status, text) {
    const dot = $('#status-dot');
    const txt = $('#status-text');
    dot.className = 'status-dot';
    if (status === 'loading') dot.classList.add('loading');
    if (status === 'error') dot.classList.add('error');
    if (txt) txt.textContent = text || status;
  }

  /** 显示 loading 状态 */
  function showLoading(el) {
    if (typeof el === 'string') el = $(el);
    if (el) el.innerHTML = '<p class="loading-text"><span class="spinner"></span>加载中...</p>';
  }

  /** 显示错误 */
  function showError(el, msg) {
    if (typeof el === 'string') el = $(el);
    if (el) el.innerHTML = `<p class="error-text">❌ ${msg}</p>`;
  }

  /** 通用 API 调用 (BentoML API 需要 POST + JSON body) */
  async function apiCall(endpoint, params = {}) {
    const url = API_BASE + '/' + endpoint;
    const resp = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });
    return resp.json();
  }

  /** 发送 GET 请求 (用于不需要 POST 的场景，如静态资源) */
  async function apiGet(endpoint, params = {}) {
    const url = new URL(API_BASE + '/' + endpoint);
    Object.entries(params).forEach(([k, v]) => {
      if (v !== undefined && v !== null && v !== '') url.searchParams.set(k, v);
    });
    const resp = await fetch(url.toString());
    return resp.json();
  }

  /** 渲染 Plotly 图表到容器 */
  function renderPlotly(containerId, chartData) {
    const el = document.getElementById(containerId);
    if (!el) return;
    if (!chartData) {
      el.innerHTML = '<p class="loading-text">图表数据不可用</p>';
      return;
    }
    // chartData 可能是 plotly JSON (来自 Python PlotlyJSONEncoder)
    if (chartData.data && chartData.layout) {
      Plotly.newPlot(el, chartData.data, chartData.layout, {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ['lasso2d', 'select2d'],
      });
      return;
    }
    // chartData 也可能是包含 layout 的完整 figure
    if (chartData.layout) {
      Plotly.newPlot(el, chartData.data || [], chartData.layout, {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ['lasso2d', 'select2d'],
      });
      return;
    }
    el.innerHTML = '<p class="loading-text">图表数据格式错误</p>';
  }

  /** 将 ISO 字符串转为 HH:MM */
  function toHHMM(isoStr) {
    if (!isoStr) return '--:--';
    const m = isoStr.match(/T(\d{2}:\d{2})/);
    return m ? m[1] : isoStr.slice(-8, -3) || isoStr;
  }

  // ============================================================
  // Tab 切换
  // ============================================================
  function initTabs() {
    const nav = $('#tab-nav');
    if (!nav) return;
    nav.addEventListener('click', function (e) {
      const btn = e.target.closest('.tab-btn');
      if (!btn) return;
      const tabId = btn.dataset.tab;
      if (!tabId) return;

      // 更新按钮状态
      $$('.tab-btn', nav).forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      // 切换面板
      $$('.tab-panel').forEach(p => p.classList.remove('active'));
      const panel = document.getElementById(tabId);
      if (panel) panel.classList.add('active');

      currentTab = tabId;

      // 切换时自动加载对应数据
      onTabActivated(tabId);
    });
  }

  /** Tab 切换后的自动加载逻辑 */
  function onTabActivated(tabId) {
    switch (tabId) {
      case 'tab-weather':
        loadWeather();
        break;
      case 'tab-explore':
        loadDataOverview();
        loadDatePicker();
        break;
      case 'tab-model':
        loadModelInfo();
        break;
      // predict 和 error 手动触发，不自动加载
    }
  }

  // ============================================================
  // Tab 1: 实时气象
  // ============================================================
  async function loadWeather() {
    setStatus('loading', '加载气象...');
    const summaryEl = $('#weather-summary');
    showLoading(summaryEl);

    try {
      const data = await apiCall('api_weather');
      if (!data.success) {
        showError(summaryEl, data.error || '气象数据加载失败');
        setStatus('error', '气象失败');
        return;
      }
      lastWeatherData = data.weather;
      renderWeatherSummary(data.weather, data.summary_html);
      renderRadiationChart(data.weather);
      renderForecastTable(data.weather);
      setStatus('ok', '气象已更新');
    } catch (e) {
      showError(summaryEl, '网络请求失败: ' + e.message);
      setStatus('error', '连接失败');
    }
  }

  function renderWeatherSummary(weather, summaryHtml) {
    const el = $('#weather-summary');
    if (!el) return;

    // 优先使用服务端生成的 HTML
    if (summaryHtml) {
      el.innerHTML = summaryHtml;
      return;
    }

    // fallback: 前端构建气象卡片
    if (!weather) {
      el.innerHTML = '<p class="loading-text">气象数据不可用</p>';
      return;
    }

    const w = weather;
    const temp = w.temperature != null ? w.temperature : (w.air_temperature != null ? w.air_temperature : '--');
    const rad = w.radiation != null ? w.radiation : (w.shortwave_radiation != null ? w.shortwave_radiation : '--');
    const humid = w.humidity != null ? w.humidity : (w.relative_humidity != null ? w.relative_humidity : '--');
    const ws = w.wind_speed != null ? w.wind_speed : (w.wind != null ? w.wind : '--');

    const tColor = tempColor(parseFloat(temp));
    el.innerHTML = `
      <div class="weather-grid">
        <div class="weather-item">
          <div class="wi-icon">🌡️</div>
          <div class="wi-value" style="color:${tColor}">${temp}°C</div>
          <div class="wi-label">气温</div>
        </div>
        <div class="weather-item">
          <div class="wi-icon">☀️</div>
          <div class="wi-value">${rad} W/m²</div>
          <div class="wi-label">辐照度</div>
        </div>
        <div class="weather-item">
          <div class="wi-icon">💧</div>
          <div class="wi-value">${humid}%</div>
          <div class="wi-label">相对湿度</div>
        </div>
        <div class="weather-item">
          <div class="wi-icon">💨</div>
          <div class="wi-value">${ws} m/s</div>
          <div class="wi-label">风速</div>
        </div>
      </div>
    `;
  }

  function renderRadiationChart(weather) {
    const el = $('#chart-radiation');
    if (!el || !weather) return;

    const rad = weather.radiation != null ? weather.radiation : weather.shortwave_radiation;
    const temp = weather.temperature != null ? weather.temperature : weather.air_temperature;
    const time = weather.time || weather.datetime || new Date().toISOString();
    const timeStr = toHHMM(time);

    const trace1 = {
      x: [timeStr],
      y: [rad || 0],
      type: 'bar',
      name: '辐照度 (W/m²)',
      marker: { color: '#ff9800' },
      yaxis: 'y',
    };
    const trace2 = {
      x: [timeStr],
      y: [temp || 0],
      type: 'scatter',
      mode: 'lines+markers',
      name: '气温 (°C)',
      marker: { color: tempColor(parseFloat(temp)), size: 10 },
      yaxis: 'y2',
    };

    const layout = {
      title: '当前辐照度 & 气温',
      yaxis: { title: '辐照度 (W/m²)', side: 'left' },
      yaxis2: { title: '气温 (°C)', side: 'right', overlaying: 'y' },
      margin: { l: 50, r: 50, t: 40, b: 40 },
      template: 'plotly_white',
      height: 300,
      legend: { orientation: 'h', y: 1.12 },
    };

    Plotly.newPlot(el, [trace1, trace2], layout, { responsive: true });
  }

  function renderForecastTable(weather) {
    const el = $('#weather-forecast-table');
    if (!el || !weather) {
      if (el) el.innerHTML = '<p class="loading-text">暂无预报数据</p>';
      return;
    }

    // 尝试从 weather 中获取预报数据
    const forecast = weather.forecast || weather.hourly || [];
    if (!Array.isArray(forecast) || forecast.length === 0) {
      el.innerHTML = '<p class="loading-text">暂无未来预报数据</p>';
      return;
    }

    let rows = '';
    forecast.slice(0, 12).forEach(f => {
      const t = f.time || f.datetime || '';
      const temp = f.temperature != null ? f.temperature : '--';
      const rad = f.radiation != null ? f.radiation : '--';
      const ws = f.wind_speed != null ? f.wind_speed : '--';
      rows += `<tr>
        <td>${toHHMM(t)}</td>
        <td>${temp}°C</td>
        <td>${rad} W/m²</td>
        <td>${ws} m/s</td>
      </tr>`;
    });

    el.innerHTML = `
      <table class="forecast-table">
        <thead>
          <tr><th>时间</th><th>气温</th><th>辐照度</th><th>风速</th></tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    `;
  }

  // ============================================================
  // Tab 2: 联合预测
  // ============================================================
  function initPredictButton() {
    const btn = $('#btn-predict');
    if (!btn) return;
    btn.addEventListener('click', runPrediction);
  }

  async function runPrediction() {
    const modeEl = $('#pred-mode');
    const priceEl = $('#pred-price');
    const loadEl = $('#pred-load');
    if (!modeEl || !priceEl || !loadEl) return;

    const nSteps = parseInt(modeEl.value) || 4;
    const price = parseFloat(priceEl.value) || 0.63;
    const load = parseFloat(loadEl.value) || 875;

    const btn = $('#btn-predict');
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span>预测中...';
    setStatus('loading', '预测中...');

    const chartEl = $('#chart-prediction');
    showLoading(chartEl);

    try {
      const data = await apiCall('api_predict', {
        model: 'combined',
        n_steps: nSteps,
        price: price,
        load: load,
      });

      if (!data.success) {
        showError(chartEl, data.error || '预测失败');
        setStatus('error', '预测失败');
        btn.disabled = false;
        btn.innerHTML = '🚀 开始预测';
        return;
      }

      lastPredictionResult = data;
      updateMetricCards(data);
      renderPredictionChart(data);
      renderStrategy(data);
      updateModelStatusBar(data);
      setStatus('ok', '预测完成');
    } catch (e) {
      showError(chartEl, '网络请求失败: ' + e.message);
      setStatus('error', '网络错误');
    } finally {
      btn.disabled = false;
      btn.innerHTML = '🚀 开始预测';
    }
  }

  function updateMetricCards(data) {
    $('#mc-green').textContent = data.green_ratio != null ? data.green_ratio.toFixed(1) + '%' : '--';
    $('#mc-solar').textContent = data.total_solar != null ? data.total_solar.toFixed(1) : '--';
    $('#mc-load').textContent = data.total_load != null ? data.total_load.toFixed(1) : '--';
    $('#mc-peak').textContent = data.solar_peak != null ? data.solar_peak.toFixed(1) : '--';
  }

  function renderPredictionChart(data) {
    const el = $('#chart-prediction');
    if (!el) return;
    if (!data.times || !data.solar || !data.load_mean) {
      el.innerHTML = '<p class="loading-text">预测数据不足</p>';
      return;
    }

    const times = data.times;
    const solar = data.solar;
    const loadMean = data.load_mean;
    const loadLower = data.load_lower || loadMean.map(v => v * 0.95);
    const loadUpper = data.load_upper || loadMean.map(v => v * 1.05);

    // 净负荷
    const net = solar.map((s, i) => s - loadMean[i]);

    const trace1 = {
      x: times,
      y: solar,
      type: 'scatter',
      mode: 'lines',
      name: '光伏出力 (kW)',
      line: { color: '#ff9800', width: 2 },
      fill: 'tozeroy',
      fillcolor: 'rgba(255,152,0,0.15)',
      xaxis: 'x',
      yaxis: 'y',
    };

    const trace2 = {
      x: times,
      y: loadMean,
      type: 'scatter',
      mode: 'lines',
      name: '充电负荷 (kW)',
      line: { color: '#2196f3', width: 2 },
      xaxis: 'x',
      yaxis: 'y',
    };

    // 置信区间
    const fillX = times.concat(times.slice().reverse());
    const fillY = loadUpper.concat(loadLower.slice().reverse());
    const trace3 = {
      x: fillX,
      y: fillY,
      type: 'scatter',
      fill: 'toself',
      fillcolor: 'rgba(33,150,243,0.15)',
      line: { color: 'rgba(33,150,243,0.1)', width: 0 },
      name: '负荷区间',
      xaxis: 'x',
      yaxis: 'y',
      showlegend: true,
    };

    const colors = net.map(v => (v >= 0 ? '#4caf50' : '#f44336'));
    const trace4 = {
      x: times,
      y: net,
      type: 'bar',
      name: '净负荷',
      marker: { color: colors },
      opacity: 0.8,
      xaxis: 'x2',
      yaxis: 'y2',
    };

    const trace5 = {
      x: [times[0], times[times.length - 1]],
      y: [0, 0],
      type: 'scatter',
      mode: 'lines',
      line: { dash: 'dash', color: '#666', width: 1 },
      name: '零点线',
      xaxis: 'x2',
      yaxis: 'y2',
      showlegend: false,
      hoverinfo: 'none',
    };

    const layout = {
      grid: { rows: 2, columns: 1, pattern: 'independent', roworder: 'top to bottom' },
      title: '光伏出力预测 + 充电负荷预测',
      legend: { orientation: 'h', y: 1.15, x: 1, xanchor: 'right' },
      height: 550,
      margin: { l: 50, r: 40, t: 60, b: 40 },
      template: 'plotly_white',
      xaxis: { title: '时间' },
      yaxis: { title: 'kW' },
      xaxis2: { title: '时间' },
      yaxis2: { title: '净负荷 (kW)' },
    };

    Plotly.newPlot(el, [trace1, trace2, trace3, trace4, trace5], layout, {
      responsive: true,
      displayModeBar: true,
      modeBarButtonsToRemove: ['lasso2d', 'select2d'],
    });
  }

  function renderStrategy(data) {
    const el = $('#strategy-content');
    if (!el) return;
    if (data.strategy) {
      // 简单 Markdown 渲染
      const html = data.strategy
        .replace(/### (.*)/g, '<h3>$1</h3>')
        .replace(/## (.*)/g, '<h3>$1</h3>')
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/^- (.*)$/gm, '<li>$1</li>')
        .replace(/(\n<li>.*<\/li>\n?)+/g, '<ul>$&</ul>')
        .replace(/\n\n/g, '<br><br>')
        .replace(/\n/g, '<br>');
      el.innerHTML = html;
      return;
    }

    // fallback: 前端生成简单建议
    const green = data.green_ratio || 0;
    let html = `<p><strong>绿电替代率</strong>: ${green.toFixed(1)}%</p>`;
    if (green >= 80) {
      html += '<p>🟢 绿电充足，建议优先使用光伏充电，储能系统保持满充状态。</p>';
    } else if (green >= 50) {
      html += '<p>🟡 绿电中等，可在高辐照时段充电，适当使用电网补充。</p>';
    } else {
      html += '<p>🔴 绿电不足，建议利用谷电时段充电，减少峰电用电。</p>';
    }
    el.innerHTML = html;
  }

  function updateModelStatusBar(data) {
    const el = $('#model-status-bar');
    if (!el) return;
    const status = data.model_status || {};
    const solarOk = status.solar_ok ? '✅' : '⚠️ (模拟)';
    const chargingOk = status.charging_ok ? '✅' : '⚠️ (模拟)';
    el.innerHTML = `
      <div class="section-card" style="margin-top:12px;font-size:13px;padding:10px 16px;">
        光伏模型: ${solarOk} &nbsp;|&nbsp; 充电模型: ${chargingOk}
        &nbsp;|&nbsp; 预测步数: ${(data.times || []).length} 步
      </div>
    `;
  }

  // ============================================================
  // Tab 3: 数据探索
  // ============================================================
  async function loadDataOverview() {
    const el = $('#data-overview');
    if (!el) return;
    showLoading(el);
    try {
      const data = await apiCall('api_data_overview');
      if (data.success && data.overview_html) {
        el.innerHTML = data.overview_html;
      } else {
        showError(el, data.error || '无法加载概览');
      }
    } catch (e) {
      showError(el, '网络请求失败: ' + e.message);
    }
  }

  async function loadDatePicker() {
    const el = $('#date-picker');
    if (!el) return;
    try {
      const data = await apiCall('api_data_overview');
      if (data.success && data.dates) {
        el.innerHTML = data.dates
          .map(d => `<option value="${d}">${d}</option>`)
          .join('');
      }
    } catch (e) {
      // ignore
    }
  }

  function initDataExplorationButtons() {
    // 日负荷曲线
    const btnLoad = $('#btn-load-curve');
    if (btnLoad) {
      btnLoad.addEventListener('click', async () => {
        const sel = $('#date-picker');
        const selected = sel ? Array.from(sel.selectedOptions).map(o => o.value) : [];
        const chartEl = $('#chart-daily-load');
        showLoading(chartEl);
        try {
          const data = await apiCall('api_daily_load', { dates: selected });
          if (data.success && data.chart) {
            renderPlotly('chart-daily-load', data.chart);
          } else {
            showError(chartEl, data.error || '加载失败');
          }
        } catch (e) {
          showError(chartEl, '网络错误: ' + e.message);
        }
      });
    }

    // 相关性
    const btnCorr = $('#btn-correlation');
    if (btnCorr) {
      btnCorr.addEventListener('click', async () => {
        const chartEl = $('#chart-correlation');
        showLoading(chartEl);
        try {
          const data = await apiCall('api_correlation');
          if (data.success) renderPlotly('chart-correlation', data.chart);
          else showError(chartEl, data.error || '加载失败');
        } catch (e) {
          showError(chartEl, '网络错误');
        }
      });
    }

    // 24h 剖面
    const btnProf = $('#btn-hourly-profile');
    if (btnProf) {
      btnProf.addEventListener('click', async () => {
        const chartEl = $('#chart-hourly-profile');
        showLoading(chartEl);
        try {
          const data = await apiCall('api_hourly_profile');
          if (data.success) renderPlotly('chart-hourly-profile', data.chart);
          else showError(chartEl, data.error || '加载失败');
        } catch (e) {
          showError(chartEl, '网络错误');
        }
      });
    }
  }

  // ============================================================
  // Tab 4: 误差分析
  // ============================================================
  function initErrorButtons() {
    const apiMap = {
      'btn-backtest-charging': { endpoint: 'api_backtest_charging', chart: 'chart-backtest-charging', summary: 'backtest-charging-summary' },
      'btn-backtest-solar': { endpoint: 'api_backtest_solar', chart: 'chart-backtest-solar', summary: 'backtest-solar-summary' },
      'btn-charging-error-dist': { endpoint: 'api_charging_error_dist', chart: 'chart-charging-error-dist' },
      'btn-solar-error-dist': { endpoint: 'api_solar_error_dist', chart: 'chart-solar-error-dist' },
      'btn-charging-error-hourly': { endpoint: 'api_charging_error_hourly', chart: 'chart-charging-error-hourly' },
      'btn-solar-error-hourly': { endpoint: 'api_solar_error_hourly', chart: 'chart-solar-error-hourly' },
    };

    Object.entries(apiMap).forEach(([btnId, cfg]) => {
      const btn = $('#' + btnId);
      if (!btn) return;
      btn.addEventListener('click', async () => {
        const chartEl = $('#' + cfg.chart);
        if (chartEl) showLoading(chartEl);
        try {
          const data = await apiCall(cfg.endpoint);
          if (data.success) {
            if (cfg.chart) renderPlotly(cfg.chart, data.chart);
            if (cfg.summary && data.summary) {
              const sumEl = $('#' + cfg.summary);
              if (sumEl) {
                // 简单 Markdown -> HTML
                const html = data.summary
                  .replace(/### (.*)/g, '<h3>$1</h3>')
                  .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                  .replace(/\|(.*)\|/g, (m) => {
                    if (m.includes('---')) return '';
                    return m;
                  })
                  .replace(/\n\n/g, '<br>')
                  .replace(/\n/g, '<br>');
                sumEl.innerHTML = html;
              }
            }
          } else {
            if (chartEl) showError(chartEl, data.error || '回测失败');
          }
        } catch (e) {
          if (chartEl) showError(chartEl, '网络错误: ' + e.message);
        }
      });
    });
  }

  // ============================================================
  // Tab 5: 模型信息
  // ============================================================
  async function loadModelInfo() {
    const el = $('#model-info-content');
    if (!el) return;
    showLoading(el);
    try {
      const data = await apiCall('api_model_info');
      if (data.success && data.info) {
        // 简单 Markdown -> HTML
        let html = data.info
          .replace(/### (.*)/g, '<h3>$1</h3>')
          .replace(/#### (.*)/g, '<h4>$1</h4>')
          .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
          .replace(/^- (.*)$/gm, '<li>$1</li>')
          .replace(/(\n<li>.*<\/li>\n?)+/g, '<ul>$&</ul>')
          .replace(/\n\n/g, '<br><br>')
          .replace(/\n/g, '<br>');
        el.innerHTML = html;
      } else {
        showError(el, data.error || '模型信息不可用');
      }
    } catch (e) {
      showError(el, '网络请求失败: ' + e.message);
    }
  }

  // ============================================================
  // 暗色模式切换
  // ============================================================
  function initThemeToggle() {
    const btn = $('#theme-toggle');
    if (!btn) return;

    // 从 localStorage 读取偏好
    const saved = localStorage.getItem('theme');
    if (saved === 'dark') {
      document.body.classList.add('dark');
      btn.textContent = '☀️';
    }

    btn.addEventListener('click', () => {
      const isDark = document.body.classList.toggle('dark');
      localStorage.setItem('theme', isDark ? 'dark' : 'light');
      btn.textContent = isDark ? '☀️' : '🌙';
    });
  }

  // ============================================================
  // 初始化入口
  // ============================================================
  function init() {
    initTabs();
    initThemeToggle();
    initPredictButton();
    initDataExplorationButtons();
    initErrorButtons();

    // 启动时加载气象数据
    loadWeather();
  }

  // 等待 DOM 就绪
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();