/* ============================================================ */
/* 光储充智能预测系统 — 前端应用 (纯 JS + Plotly)              */
/* ============================================================ */

// ──────────────────────────────────────────────
// 全局状态
// ──────────────────────────────────────────────
const STATE = {
  theme: 'dark',
  currentTab: 'tab-predict',
  predictResult: null,
  weatherData: null,
  dataOverview: null,
  availableDates: [],
  backtestType: null, // 'charging' | 'solar'
  strategyData: null,
  // 缓存服务端图表原始 JSON，用于主题切换时重绘
  serverCharts: {},
};

// API 基础路径 (BentoML REST API)
const API_BASE = '/api';

// ──────────────────────────────────────────────
// DOM Ready
// ──────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  initTabs();
  initClock();
  initPredictTab();
  initDataTab();
  initErrorTab();
  initWeatherTab();
  initStrategyTab();

  // 默认加载
  setTimeout(loadWeather, 400);
});

// ──────────────────────────────────────────────
// 通用工具函数
// ──────────────────────────────────────────────
function $(id) {
  const el = document.getElementById(id);
  if (!el) console.warn(`[app] Element not found: #${id}`);
  return el;
}

function apiCall(endpoint, params = {}) {
  const url = API_BASE + '/' + endpoint;
  return fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  }).then(r => {
    if (!r.ok) throw new Error(`HTTP ${r.status} ${r.statusText}`);
    return r.json();
  });
}

/**
 * 简易 Markdown → HTML 转换
 * 处理后端返回的 Markdown 文本（###, | 表格, **粗体**, - 列表）
 */
function markdownToHTML(md) {
  if (!md) return '';
  let html = md;
  // 标题
  html = html.replace(/^#### (.+)$/gm, '<h4>$1</h4>');
  html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
  // 粗体
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  // 块引用
  html = html.replace(/^> (.+)$/gm, '<blockquote>$1</blockquote>');
  // 水平线
  html = html.replace(/^---$/gm, '<hr>');
  // 表格：将连续的 | 行包裹为 <table>
  html = html.replace(/((?:^\|.+\|[\r\n]?)+)/gm, function(match) {
    const lines = match.trim().split('\n');
    let tableHtml = '<table class="md-table">';
    lines.forEach((line, i) => {
      const cells = line.split('|').filter(c => c.trim() !== '');
      const tag = (i === 0) ? 'th' : 'td';
      // 跳过分隔行
      if (line.match(/^\|[\s\-:|]+\|$/)) return;
      tableHtml += '<tr>';
      cells.forEach(cell => {
        tableHtml += `<${tag}>${cell.trim()}</${tag}>`;
      });
      tableHtml += '</tr>';
    });
    tableHtml += '</table>';
    return tableHtml;
  });
  // 无序列表
  html = html.replace(/^- (.+)$/gm, '<li>$1</li>');
  html = html.replace(/((?:<li>.+<\/li>\n?)+)/g, '<ul>$1</ul>');
  // 段落：将非标签的连续文本包裹
  html = html.replace(/\n\n/g, '<br><br>');
  return html;
}

function setFooter(msg, ok = true) {
  const el = $('footer-api-status');
  const msgEl = $('footer-msg');
  if (!el || !msgEl) return;
  if (ok) {
    el.innerHTML = '<span class="status-dot green"></span> API: 正常';
    el.style.color = 'var(--success)';
  } else {
    el.innerHTML = '<span class="status-dot red"></span> API: 异常';
    el.style.color = 'var(--danger)';
  }
  msgEl.textContent = msg || '';
}

function showError(boxId, msg) {
  const box = $(boxId);
  if (!box) return;
  box.textContent = msg;
  box.classList.remove('hidden');
}

function hideError(boxId) {
  const el = $(boxId);
  if (el) el.classList.add('hidden');
}

function disableBtn(id) {
  const el = $(id);
  if (!el) return;
  el.disabled = true;
  el.textContent = '处理中…';
}

function enableBtn(id, text) {
  const el = $(id);
  if (!el) return;
  el.disabled = false;
  el.textContent = text;
}

/**
 * 安全设置 textContent
 */
function safeText(id, text) {
  const el = $(id);
  if (el) el.textContent = text;
}

/**
 * 安全设置 innerHTML
 */
function safeHTML(id, html) {
  const el = $(id);
  if (el) el.innerHTML = html;
}

/**
 * 构建与 plotly 主题匹配的 layout 基础属性
 */
function plotlyLayout(title, xlabel, ylabel, extra = {}) {
  const isDark = STATE.theme === 'dark';
  return {
    title: title,
    xaxis: {
      title: xlabel,
      gridcolor: isDark ? '#334155' : '#e2e8f0',
      zerolinecolor: isDark ? '#475569' : '#cbd5e1',
      tickfont: { color: isDark ? '#94a3b8' : '#475569' },
    },
    yaxis: {
      title: ylabel,
      gridcolor: isDark ? '#334155' : '#e2e8f0',
      zerolinecolor: isDark ? '#475569' : '#cbd5e1',
      tickfont: { color: isDark ? '#94a3b8' : '#475569' },
    },
    plot_bgcolor: isDark ? '#1e293b' : '#ffffff',
    paper_bgcolor: isDark ? '#1e293b' : '#ffffff',
    font: { color: isDark ? '#e2e8f0' : '#1e293b' },
    margin: { l: 60, r: 30, t: 40, b: 50 },
    legend: { orientation: 'h', y: 1.12 },
    ...extra,
  };
}

// ──────────────────────────────────────────────
// 主题
// ──────────────────────────────────────────────
function initTheme() {
  const select = $('theme-select');
  if (!select) return;
  select.value = STATE.theme;
  document.documentElement.setAttribute('data-theme', STATE.theme);

  select.addEventListener('change', () => {
    STATE.theme = select.value;
    document.documentElement.setAttribute('data-theme', STATE.theme);
    redrawAllCharts();
  });

  const mobileBtn = $('theme-toggle-mobile');
  if (!mobileBtn) return;
  mobileBtn.addEventListener('click', () => {
    STATE.theme = STATE.theme === 'dark' ? 'light' : 'dark';
    select.value = STATE.theme;
    document.documentElement.setAttribute('data-theme', STATE.theme);
    redrawAllCharts();
  });
}

function redrawAllCharts() {
  if (STATE.predictResult) {
    renderSolarOnlyChart(STATE.predictResult);
    renderChargingOnlyChart(STATE.predictResult);
    renderJointChart(STATE.predictResult);
  }

  // 重绘当前活动 tab 中来自服务器的图表
  if (STATE.currentTab === 'tab-data') loadDataTabCharts();
  if (STATE.currentTab === 'tab-error') loadErrorTabCharts();
  if (STATE.currentTab === 'tab-weather') loadWeather();
  if (STATE.currentTab === 'tab-strategy') renderStrategyDisplay();
}

// ──────────────────────────────────────────────
// 时钟
// ──────────────────────────────────────────────
function initClock() {
  function tick() {
    const now = new Date();
    const el = $('header-time');
    if (el) el.textContent = now.toLocaleString('zh-CN');
  }
  tick();
  setInterval(tick, 1000);
}

// ──────────────────────────────────────────────
// Tab 切换
// ──────────────────────────────────────────────
function initTabs() {
  const buttons = document.querySelectorAll('.tab-btn[data-tab]');
  buttons.forEach(btn => {
    btn.addEventListener('click', () => {
      const tabId = btn.getAttribute('data-tab');
      switchTab(tabId);
    });
  });
}

function switchTab(tabId) {
  STATE.currentTab = tabId;
  // 更新按钮状态
  document.querySelectorAll('.tab-btn[data-tab]').forEach(b => b.classList.remove('active'));
  const activeBtn = document.querySelector(`.tab-btn[data-tab="${tabId}"]`);
  if (activeBtn) activeBtn.classList.add('active');
  // 更新内容显示
  document.querySelectorAll('.tab-content').forEach(s => s.classList.remove('active'));
  const target = $(tabId);
  if (target) target.classList.add('active');
  // 按需加载图表
  setTimeout(() => {
    if (tabId === 'tab-data') loadDataTabCharts();
    if (tabId === 'tab-error') loadErrorTabCharts();
    if (tabId === 'tab-weather') Plotly.Plots.resize('weather-chart');
  }, 100);
}

// ──────────────────────────────────────────────
// Tab 1: 充放电预测
// ──────────────────────────────────────────────
function initPredictTab() {
  // 日期选择器默认值
  const dp = $('date-picker');
  if (dp) {
    const today = new Date();
    dp.value = today.toISOString().slice(0, 10);
  }

  const btnPredict = $('btn-predict');
  if (btnPredict) btnPredict.addEventListener('click', runPrediction);
}

async function runPrediction() {
  disableBtn('btn-predict');
  hideError('predict-error');
  const solarDiv = $('chart-solar-only');
  const chargingDiv = $('chart-charging-only');
  const jointDiv = $('chart-joint');
  if (solarDiv) solarDiv.innerHTML = '<div class="spinner"></div>';
  if (chargingDiv) chargingDiv.innerHTML = '<div class="spinner"></div>';
  if (jointDiv) jointDiv.innerHTML = '<div class="spinner"></div>';

  const modelSel = $('model-select');
  const stepsSel = $('steps-select');
  const priceInput = $('price-input');
  const loadInput = $('load-input');
  if (!modelSel || !stepsSel || !priceInput || !loadInput) {
    showError('predict-error', '表单元素缺失，请刷新页面');
    enableBtn('btn-predict', '执行预测');
    return;
  }
  const model = modelSel.value;
  const nSteps = parseInt(stepsSel.value);
  const price = parseFloat(priceInput.value);
  const load = parseFloat(loadInput.value);

  try {
    const data = await apiCall('api_predict', {
      model,
      n_steps: nSteps,
      price,
      load,
    });

    if (!data.success) {
      showError('predict-error', data.error || '预测失败');
      if (solarDiv) solarDiv.textContent = '预测失败';
      if (chargingDiv) chargingDiv.textContent = '预测失败';
      if (jointDiv) jointDiv.textContent = '预测失败';
      resetKPIs();
      setFooter(data.error || '预测失败', false);
    } else {
      STATE.predictResult = data;
      updateKPIs(data);
      renderSolarOnlyChart(data);
      renderChargingOnlyChart(data);
      renderJointChart(data);
      setFooter('预测完成', true);
      const hs = $('header-status');
      if (hs) hs.innerHTML = '预测就绪';
    }
  } catch (e) {
    console.error(e);
    showError('predict-error', e.message);
    if (solarDiv) solarDiv.textContent = '请求失败';
    if (chargingDiv) chargingDiv.textContent = '请求失败';
    if (jointDiv) jointDiv.textContent = '请求失败';
    resetKPIs();
    setFooter(e.message, false);
  } finally {
    enableBtn('btn-predict', '执行预测');
  }
}

// ──────────────────────────────────────────────
// 预测图：光伏单独 (Row1 左)
// ──────────────────────────────────────────────
function renderSolarOnlyChart(data) {
  const div = $('chart-solar-only');
  if (!div) return;
  div.innerHTML = '';
  Plotly.purge('chart-solar-only');
  const traces = [
    {
      x: data.times,
      y: data.solar,
      type: 'scatter',
      mode: 'lines+markers',
      name: '光伏出力 (kW)',
      line: { color: '#ff9800', width: 1.5 },
      marker: { size: 3 },
      fill: 'tozeroy',
      fillcolor: 'rgba(255,152,0,0.15)',
    },
  ];
  const layout = plotlyLayout('光伏出力预测', '时间', 'kW', { height: 350 });
  Plotly.newPlot('chart-solar-only', traces, layout, { responsive: true, displayModeBar: false });
}

// ──────────────────────────────────────────────
// 预测图：充电单独 (Row1 右)
// ──────────────────────────────────────────────
function renderChargingOnlyChart(data) {
  const div = $('chart-charging-only');
  if (!div) return;
  div.innerHTML = '';
  Plotly.purge('chart-charging-only');
  const fillColor = STATE.theme === 'dark' ? 'rgba(59,130,246,0.2)' : 'rgba(37,99,235,0.15)';
  const traces = [
    // 不确定性区间（先绘制，在下方）
    {
      x: data.times.concat(data.times.slice().reverse()),
      y: data.load_upper.concat(data.load_lower.slice().reverse()),
      type: 'scatter',
      fill: 'toself',
      fillcolor: fillColor,
      line: { color: 'rgba(33,150,243,0.1)', width: 0 },
      name: '负荷区间',
      hoverinfo: 'skip',
    },
    // 主线
    {
      x: data.times,
      y: data.load_mean,
      type: 'scatter',
      mode: 'lines+markers',
      name: '充电负荷 (kW)',
      line: { color: '#2196f3', width: 2 },
      marker: { size: 4 },
    },
  ];
  const layout = plotlyLayout('充电负荷预测', '时间', 'kW', { height: 350 });
  Plotly.newPlot('chart-charging-only', traces, layout, { responsive: true, displayModeBar: false });
}

// ──────────────────────────────────────────────
// 预测图：联合分布 (Row2) — 上半光伏+充电，下半净负荷条形图
// ──────────────────────────────────────────────
function renderJointChart(data) {
  const div = $('chart-joint');
  if (!div) return;
  div.innerHTML = '';
  Plotly.purge('chart-joint');

  const net = data.solar.map((s, i) => s - data.load_mean[i]);
  const barColors = net.map(v => v >= 0 ? '#4caf50' : '#f44336');

  const traces = [
    // Row 1: 光伏出力 (col=1, row=1)
    {
      x: data.times,
      y: data.solar,
      type: 'scatter',
      mode: 'lines',
      name: '光伏出力 (kW)',
      line: { color: '#ff9800', width: 2 },
      fill: 'tozeroy',
      fillcolor: 'rgba(255,152,0,0.15)',
      xaxis: 'x',
      yaxis: 'y',
    },
    // Row 1: 充电负荷 (col=1, row=1)
    {
      x: data.times,
      y: data.load_mean,
      type: 'scatter',
      mode: 'lines',
      name: '充电负荷 (kW)',
      line: { color: '#2196f3', width: 2 },
      xaxis: 'x',
      yaxis: 'y',
    },
    // Row 1: 不确定性区间
    {
      x: data.times.concat(data.times.slice().reverse()),
      y: data.load_upper.concat(data.load_lower.slice().reverse()),
      type: 'scatter',
      fill: 'toself',
      fillcolor: 'rgba(33,150,243,0.15)',
      line: { color: 'rgba(33,150,243,0.1)', width: 0 },
      name: '负荷区间',
      hoverinfo: 'skip',
      xaxis: 'x',
      yaxis: 'y',
    },
    // Row 2: 净负荷条形图
    {
      x: data.times,
      y: net,
      type: 'bar',
      name: '净负荷',
      marker: { color: barColors, opacity: 0.8 },
      xaxis: 'x2',
      yaxis: 'y2',
    },
  ];

  const isDark = STATE.theme === 'dark';
  const layout = {
    title: '联合预测分析',
    grid: { rows: 2, columns: 1, pattern: 'independent', roworder: 'top to bottom' },
    xaxis: {
      title: null,
      domain: [0, 1],
      anchor: 'y',
      gridcolor: isDark ? '#334155' : '#e2e8f0',
      zerolinecolor: isDark ? '#475569' : '#cbd5e1',
      tickfont: { color: isDark ? '#94a3b8' : '#475569' },
    },
    yaxis: {
      title: 'kW',
      domain: [0.37, 1],
      anchor: 'x',
      gridcolor: isDark ? '#334155' : '#e2e8f0',
      zerolinecolor: isDark ? '#475569' : '#cbd5e1',
      tickfont: { color: isDark ? '#94a3b8' : '#475569' },
    },
    xaxis2: {
      title: '时间',
      domain: [0, 1],
      anchor: 'y2',
      matches: 'x',
      showticklabels: true,
      gridcolor: isDark ? '#334155' : '#e2e8f0',
      zerolinecolor: isDark ? '#475569' : '#cbd5e1',
      tickfont: { color: isDark ? '#94a3b8' : '#475569' },
    },
    yaxis2: {
      title: 'kW',
      domain: [0, 0.35],
      anchor: 'x2',
      gridcolor: isDark ? '#334155' : '#e2e8f0',
      zerolinecolor: isDark ? '#475569' : '#cbd5e1',
      tickfont: { color: isDark ? '#94a3b8' : '#475569' },
    },
    height: 550,
    plot_bgcolor: isDark ? '#1e293b' : '#ffffff',
    paper_bgcolor: isDark ? '#1e293b' : '#ffffff',
    font: { color: isDark ? '#e2e8f0' : '#1e293b' },
    legend: { orientation: 'h', yanchor: 'bottom', y: 1.02, xanchor: 'right', x: 1 },
    hovermode: 'x unified',
    margin: { l: 60, r: 30, t: 40, b: 50 },
    // 水平零线（净负荷 y=0）
    shapes: [
      {
        type: 'line',
        x0: 0,
        y0: 0,
        x1: 1,
        y1: 0,
        xref: 'paper',
        yref: 'y2',
        line: { dash: 'dash', color: isDark ? '#64748b' : '#9ca3af', width: 1 },
      },
    ],
  };

  Plotly.newPlot('chart-joint', traces, layout, { responsive: true, displayModeBar: false });
}

function updateKPIs(data) {
  safeText('kpi-total-solar-value', data.total_solar ? data.total_solar.toFixed(1) : '--');
  safeText('kpi-total-load-value', data.total_load ? data.total_load.toFixed(1) : '--');
  safeText('kpi-solar-peak-value', data.solar_peak ? data.solar_peak.toFixed(1) : '--');
}

function resetKPIs() {
  safeText('kpi-total-solar-value', '--');
  safeText('kpi-total-load-value', '--');
  safeText('kpi-solar-peak-value', '--');
}

function renderPredictChart1(data) {
  const div = $('chart-predict-1');
  if (!div) return;
  div.innerHTML = '';  // 确保清除 spinner（Plotly.purge 在无图表时不清理 DOM）
  Plotly.purge('chart-predict-1');
  const traces = [
    {
      x: data.times,
      y: data.solar,
      type: 'scatter',
      mode: 'lines+markers',
      name: '光伏发电 (kW)',
      line: { color: '#f59e0b', width: 2 },
      marker: { size: 4 },
    },
    {
      x: data.times,
      y: data.load_mean,
      type: 'scatter',
      mode: 'lines+markers',
      name: '充电负荷 (kW)',
      line: { color: '#3b82f6', width: 2 },
      marker: { size: 4 },
    },
  ];
  const layout = plotlyLayout('光伏 & 充电负荷预测', '时间', '功率 (kW)');
  Plotly.newPlot('chart-predict-1', traces, layout, { responsive: true, displayModeBar: false });
}

function renderPredictChart2(data) {
  const div = $('chart-predict-2');
  if (!div) return;
  div.innerHTML = '';  // 确保清除 spinner
  Plotly.purge('chart-predict-2');
  const fillColor = STATE.theme === 'dark' ? 'rgba(59,130,246,0.2)' : 'rgba(37,99,235,0.15)';
  const traces = [
    {
      x: data.times.concat(data.times.slice().reverse()),
      y: data.load_upper.concat(data.load_lower.slice().reverse()),
      type: 'scatter',
      fill: 'toself',
      fillcolor: fillColor,
      line: { color: 'rgba(59,130,246,0)' },
      name: '不确定性区间 (MC Dropout)',
      hoverinfo: 'skip',
    },
    {
      x: data.times,
      y: data.load_mean,
      type: 'scatter',
      mode: 'lines+markers',
      name: '充电负荷均值',
      line: { color: '#3b82f6', width: 2.5 },
      marker: { size: 5 },
    },
  ];
  const layout = plotlyLayout('充电负荷预测 (含 MC Dropout 不确定性)', '时间', '功率 (kW)');
  Plotly.newPlot('chart-predict-2', traces, layout, { responsive: true, displayModeBar: false });
}

function renderStrategyChart(data) {
  const div = $('chart-strategy');
  if (!div) return;
  div.innerHTML = '';  // 确保清除 spinner
  Plotly.purge('chart-strategy');
  if (!data.strategy || !data.strategy.length) {
    div.textContent = '无调度策略数据';
    return;
  }
  const times = data.strategy.map(s => s.time);
  const solarVals = data.strategy.map(s => s.solar || 0);
  const loadVals = data.strategy.map(s => s.load || 0);
  const decisions = data.strategy.map(s => s.action);
  const decisionLabels = data.strategy.map(s => {
    let lbl = s.time + '<br>';
    lbl += `光伏: ${s.solar?.toFixed(1) || 0} kW<br>`;
    lbl += `负荷: ${s.load?.toFixed(1) || 0} kW<br>`;
    lbl += `电价: ${s.price} 元<br>`;
    lbl += `建议: ${s.action}`;
    return lbl;
  });

  const actionColors = decisions.map(d => {
    if (d.includes('储能') || d.includes('存')) return '#22c55e';
    if (d.includes('放电') || d.includes('释')) return '#ef4444';
    if (d.includes('电网') || d.includes('购电')) return '#f59e0b';
    return '#94a3b8';
  });

  const traces = [
    {
      x: times,
      y: solarVals,
      type: 'bar',
      name: '光伏',
      marker: { color: '#fbbf24', opacity: 0.8 },
      text: decisionLabels,
      hoverinfo: 'text',
    },
    {
      x: times,
      y: loadVals,
      type: 'bar',
      name: '充电',
      marker: { color: '#3b82f6', opacity: 0.8 },
      text: decisionLabels,
      hoverinfo: 'text',
    },
  ];

  const maxVal = Math.max(...solarVals, ...loadVals) || 1;
  const layout = plotlyLayout('充放电调度策略', '时间', '功率 (kW)', {
    barmode: 'group',
    shapes: times.map((t, i) => ({
      type: 'rect',
      x0: i - 0.4,
      x1: i + 0.4,
      y0: maxVal * 1.05,
      y1: maxVal * 1.12,
      fillcolor: actionColors[i],
      opacity: 0.7,
      line: { width: 0 },
    })),
  });

  // 添加颜色说明标注
  traces.push({
    x: [null],
    y: [null],
    type: 'scatter',
    mode: 'markers',
    marker: { size: 0 },
    name: '· 储能 | · 放电 | · 购电 | · 待机',
  });

  Plotly.newPlot('chart-strategy', traces, layout, { responsive: true, displayModeBar: false });
}

// ──────────────────────────────────────────────
// Tab 2: 数据探索
// ──────────────────────────────────────────────
let _dataOverviewLoaded = false;

/**
 * 安全绑定事件：元素存在时才 addEventListener
 */
function safeOn(id, event, handler) {
  const el = $(id);
  if (el) el.addEventListener(event, handler);
}

async function initDataTab() {
  const dailyModeRadios = document.querySelectorAll('input[name="daily-mode"]');

  // 模式切换
  dailyModeRadios.forEach(radio => {
    radio.addEventListener('change', () => {
      const mode = document.querySelector('input[name="daily-mode"]:checked')?.value;
      if (mode === 'aggregated') {
        const rg = $('daily-range-group');
        const sg = $('daily-single-group');
        if (rg) rg.style.display = 'flex';
        if (sg) sg.style.display = 'none';
        loadAggregatedLoadChart();
      } else {
        const rg = $('daily-range-group');
        const sg = $('daily-single-group');
        if (rg) rg.style.display = 'none';
        if (sg) sg.style.display = 'flex';
        loadSingleDayLoadChart();
      }
    });
  });

  // 聚合模式：快捷按钮 + 日期变更自动刷新
  safeOn('btn-week', 'click', () => setDateRange('week'));
  safeOn('btn-month', 'click', () => setDateRange('month'));
  safeOn('btn-last-7-days', 'click', () => setDateRange('7d'));
  safeOn('btn-last-30-days', 'click', () => setDateRange('30d'));

  let dateChangeTimer = null;
  ['daily-start-date', 'daily-end-date'].forEach(id => {
    safeOn(id, 'change', () => {
      clearTimeout(dateChangeTimer);
      dateChangeTimer = setTimeout(loadAggregatedLoadChart, 400);
    });
  });

  // 单日模式：快捷按钮 + 手动日期变更
  safeOn('btn-daily-yesterday', 'click', () => setSingleDate(-1));
  safeOn('btn-daily-today', 'click', () => setSingleDate(0));
  safeOn('btn-daily-prev', 'click', () => shiftSingleDate(-1));
  safeOn('btn-daily-next', 'click', () => shiftSingleDate(1));

  safeOn('daily-single-date', 'change', () => {
    clearTimeout(dateChangeTimer);
    dateChangeTimer = setTimeout(loadSingleDayLoadChart, 400);
  });

  // 页面初始化：从后端获取数据默认范围
  if (!_dataOverviewLoaded) {
    await loadDataOverview();
    _dataOverviewLoaded = true;
  }

  // 默认用聚合均值模式，初始加载近7天
  const startVal = $('daily-start-date')?.value;
  const endVal = $('daily-end-date')?.value;
  if (startVal && endVal) {
    loadAggregatedLoadChart();
  } else {
    setDateRange('7d');
  }
}

async function loadDataOverview() {
  try {
    const data = await apiCall('api_data_overview');
    if (data.success && data.date_range) {
      const dr = data.date_range;
      // 填充聚合日期范围输入框
      const startInput = $('daily-start-date');
      const endInput = $('daily-end-date');
      const singleDate = $('daily-single-date');
      if (startInput) startInput.value = dr.min;
      if (endInput) endInput.value = dr.max;
      if (singleDate) singleDate.value = dr.max;
      // 显示日期范围提示
      const hint = $('daily-date-range-hint');
      if (hint) {
        hint.textContent = `数据范围: ${dr.min} ~ ${dr.max} (共 ${dr.total_days} 天)`;
      }
    }
  } catch (e) {
    console.warn('loadDataOverview failed:', e);
    // 降级：用今天日期作为默认
    setDateRange('7d');
  }
}

function setDateRange(preset) {
  const startInput = $('daily-start-date');
  const endInput = $('daily-end-date');
  if (!endInput || !startInput) return;

  const today = new Date();
  today.setHours(0, 0, 0, 0);
  let start = new Date(today);
  let end = new Date(today);

  switch (preset) {
    case 'week':
      // 本周一 ~ 今天
      const dayOfWeek = today.getDay();
      const mondayOffset = dayOfWeek === 0 ? -6 : 1 - dayOfWeek;
      start = new Date(today);
      start.setDate(today.getDate() + mondayOffset);
      break;
    case 'month':
      // 本月1日 ~ 今天
      start = new Date(today.getFullYear(), today.getMonth(), 1);
      break;
    case '30d':
      start.setDate(today.getDate() - 29);
      break;
    case '7d':
    default:
      start.setDate(today.getDate() - 6);
      break;
  }

  startInput.value = start.toISOString().slice(0, 10);
  endInput.value = end.toISOString().slice(0, 10);

  loadAggregatedLoadChart();
}

function setSingleDate(offsetDays) {
  const input = $('daily-single-date');
  if (!input) return;
  const d = new Date();
  d.setHours(0, 0, 0, 0);
  d.setDate(d.getDate() + offsetDays);
  input.value = d.toISOString().slice(0, 10);
  loadSingleDayLoadChart();
}

function shiftSingleDate(step) {
  const input = $('daily-single-date');
  if (!input || !input.value) return;
  const d = new Date(input.value + 'T00:00:00');
  d.setDate(d.getDate() + step);
  input.value = d.toISOString().slice(0, 10);
  loadSingleDayLoadChart();
}

function loadDataTabCharts() {
  // 优先从缓存重绘，否则首次加载
  if (STATE.serverCharts['chart-correlation']) {
    renderChartFromServer('chart-correlation', STATE.serverCharts['chart-correlation']);
  } else if (!document.getElementById('chart-correlation')?.dataset?.loaded) {
    loadCorrelationChart();
  }
  if (STATE.serverCharts['chart-hourly-profile']) {
    renderChartFromServer('chart-hourly-profile', STATE.serverCharts['chart-hourly-profile']);
  } else if (!document.getElementById('chart-hourly-profile')?.dataset?.loaded) {
    loadHourlyProfileChart();
  }
  if (STATE.serverCharts['chart-daily-load']) {
    renderChartFromServer('chart-daily-load', STATE.serverCharts['chart-daily-load']);
  }
}

// ─── 聚合均值模式 ──────────────────────────
async function loadAggregatedLoadChart() {
  const startInput = $('daily-start-date');
  const endInput = $('daily-end-date');
  if (!startInput || !endInput) return;
  const dateStart = startInput.value;
  const dateEnd = endInput.value;
  if (!dateStart || !dateEnd) {
    safeText('chart-daily-load', '请选择起始日期和结束日期');
    return;
  }
  if (dateStart > dateEnd) {
    safeText('chart-daily-load', '起始日期不能晚于结束日期');
    return;
  }
  safeHTML('chart-daily-load', '<div class="spinner"></div>');
  try {
    const data = await apiCall('api_aggregated_load', { date_start: dateStart, date_end: dateEnd });
    if (data.chart) {
      renderChartFromServer('chart-daily-load', data.chart, true);
    } else {
      safeText('chart-daily-load', data.error || '加载失败');
    }
  } catch (e) {
    console.error(e);
    safeText('chart-daily-load', '加载失败: ' + e.message);
  }
}

// ─── 单日聚焦模式 ──────────────────────────
async function loadSingleDayLoadChart() {
  const input = $('daily-single-date');
  if (!input || !input.value) {
    safeText('chart-daily-load', '请选择一个日期');
    return;
  }
  safeHTML('chart-daily-load', '<div class="spinner"></div>');
  try {
    const data = await apiCall('api_single_day_load', { date: input.value });
    if (data.chart) {
      renderChartFromServer('chart-daily-load', data.chart, true);
    } else {
      safeText('chart-daily-load', data.error || '加载失败');
    }
  } catch (e) {
    console.error(e);
    safeText('chart-daily-load', '加载失败: ' + e.message);
  }
}

async function loadCorrelationChart() {
  safeHTML('chart-correlation', '<div class="spinner"></div>');
  try {
    const data = await apiCall('api_correlation');
    if (data.success && data.chart) {
      renderChartFromServer('chart-correlation', data.chart, true);
      const el = document.getElementById('chart-correlation');
      if (el) el.dataset.loaded = '1';
    } else if (data.chart) {
      renderChartFromServer('chart-correlation', data.chart, true);  // fallback 错误图表
      const el = document.getElementById('chart-correlation');
      if (el) el.dataset.loaded = '1';
    } else {
      safeText('chart-correlation', data.error || '无数据');
    }
  } catch (e) {
    console.error(e);
    safeText('chart-correlation', '加载失败');
  }
}

async function loadHourlyProfileChart() {
  safeHTML('chart-hourly-profile', '<div class="spinner"></div>');
  try {
    const data = await apiCall('api_hourly_profile');
    if (data.success && data.chart) {
      renderChartFromServer('chart-hourly-profile', data.chart, true);
      const el = document.getElementById('chart-hourly-profile');
      if (el) el.dataset.loaded = '1';
    } else if (data.chart) {
      renderChartFromServer('chart-hourly-profile', data.chart, true);  // fallback 错误图表
      const el = document.getElementById('chart-hourly-profile');
      if (el) el.dataset.loaded = '1';
    } else {
      safeText('chart-hourly-profile', data.error || '无数据');
    }
  } catch (e) {
    console.error(e);
    safeText('chart-hourly-profile', '加载失败');
  }
}

function adaptServerChartTheme(layout) {
  // 全面适配服务器返回的 plotly 图表到当前主题
  const isDark = STATE.theme === 'dark';
  const bg = isDark ? '#1e293b' : '#ffffff';
  const fg = isDark ? '#e2e8f0' : '#1e293b';
  const grid = isDark ? '#334155' : '#e2e8f0';
  const zero = isDark ? '#475569' : '#cbd5e1';
  const tick = isDark ? '#94a3b8' : '#475569';

  layout.paper_bgcolor = bg;
  layout.plot_bgcolor = bg;
  if (!layout.font) layout.font = {};
  layout.font.color = fg;
  // 标题颜色
  if (!layout.title) layout.title = {};
  if (typeof layout.title === 'string') layout.title = { text: layout.title };
  layout.title.font = layout.title.font || {};
  layout.title.font.color = fg;

  // 统一 margin，消除服务器图表与客户端图表的定位偏差
  layout.margin = layout.margin || {};
  layout.margin.l = layout.margin.l || 60;
  layout.margin.r = layout.margin.r || 30;
  layout.margin.t = layout.margin.t || 40;
  layout.margin.b = layout.margin.b || 50;

  // 递归处理所有轴
  function fixAxis(ax) {
    if (!ax) return;
    ax.gridcolor = grid;
    ax.zerolinecolor = zero;
    if (!ax.tickfont) ax.tickfont = {};
    ax.tickfont.color = tick;
    if (!ax.title) ax.title = {};
    if (typeof ax.title === 'string') ax.title = { text: ax.title };
    if (ax.title.font) ax.title.font.color = fg;
    // 处理子图标题
    if (!ax._subplotTitleFixed && ax.title && ax.title.text) {
      ax._subplotTitleFixed = true;
    }
    // 处理多轴 (xaxis, xaxis2...)
    if (ax.title && ax.title.font) ax.title.font.color = fg;
  }

  // 遍历所有可能的轴名
  for (let i = 1; i <= 10; i++) {
    const suffix = i === 1 ? '' : i;
    ['xaxis', 'yaxis'].forEach(prefix => {
      const key = prefix + suffix;
      if (layout[key]) fixAxis(layout[key]);
    });
  }

  // 图例
  if (layout.legend) {
    if (!layout.legend.font) layout.legend.font = {};
    layout.legend.font.color = fg;
  }

  // 移除服务器强制的 template，让前端完全控制
  delete layout.template;

  return layout;
}

function renderChartFromServer(divId, chartData, cache = false) {
  // chartData 是 Plotly figure JSON
  const div = $(divId);
  if (!div) return;
  if (cache) STATE.serverCharts[divId] = chartData;
  div.innerHTML = '';  // 确保清除 spinner
  Plotly.purge(divId); // 清除已有图表
  if (chartData.data && chartData.layout) {
    const layout = adaptServerChartTheme(JSON.parse(JSON.stringify(chartData.layout)));
    Plotly.newPlot(div, chartData.data, layout, { responsive: true, displayModeBar: false });
  } else {
    div.textContent = '无图表数据';
  }
}

// ──────────────────────────────────────────────
// Tab 3: 误差分析
// ──────────────────────────────────────────────
function initErrorTab() {
  safeOn('btn-backtest-charging', 'click', () => runBacktest('charging'));
  safeOn('btn-backtest-solar', 'click', () => runBacktest('solar'));
}

function loadErrorTabCharts() {
  // 占位 - 由按钮触发
}

async function runBacktest(type) {
  STATE.backtestType = type;
  const chartDiv = $('chart-backtest');
  const errorDiv = $('chart-error-dist');
  const hourlyDiv = $('chart-error-hourly');
  const summaryDiv = $('error-summary');

  if (chartDiv) chartDiv.innerHTML = '<div class="spinner"></div>';
  if (errorDiv) errorDiv.innerHTML = '<div class="spinner"></div>';
  if (hourlyDiv) hourlyDiv.innerHTML = '<div class="spinner"></div>';
  if (summaryDiv) summaryDiv.innerHTML = '<p>加载中…</p>';

  const endpoint = type === 'charging' ? 'api_backtest_charging' : 'api_backtest_solar';
  const errDistEp = type === 'charging' ? 'api_charging_error_dist' : 'api_solar_error_dist';
  const errHourlyEp = type === 'charging' ? 'api_charging_error_hourly' : 'api_solar_error_hourly';

  try {
    const [btData, distData, hourlyData] = await Promise.all([
      apiCall(endpoint),
      apiCall(errDistEp),
      apiCall(errHourlyEp),
    ]);

    // 渲染回测主图表
    if (btData.success && btData.chart) {
      renderChartFromServer('chart-backtest', btData.chart, true);
      if (summaryDiv) {
        summaryDiv.innerHTML = btData.summary ? markdownToHTML(btData.summary) : '<p>无误差指标数据</p>';
      }
    } else if (btData.chart) {
      // fallback 错误图表
      renderChartFromServer('chart-backtest', btData.chart, true);
      if (summaryDiv) summaryDiv.innerHTML = `<p class="error">${btData.error || '回测失败'}</p>`;
    } else {
      if (chartDiv) { chartDiv.innerHTML = ''; chartDiv.textContent = btData.error || '回测失败'; }
      if (errorDiv) errorDiv.innerHTML = '';
      if (hourlyDiv) hourlyDiv.innerHTML = '';
      if (summaryDiv) summaryDiv.innerHTML = `<p class="error">${btData.error || '回测失败'}</p>`;
    }

    // 渲染误差分布图
    if (distData.success && distData.chart) {
      renderChartFromServer('chart-error-dist', distData.chart, true);
    } else if (distData.chart) {
      // fallback 错误图表
      renderChartFromServer('chart-error-dist', distData.chart, true);
    } else if (distData.error && errorDiv) {
      errorDiv.innerHTML = '';
      errorDiv.textContent = distData.error;
    }

    // 渲染逐时误差图
    if (hourlyData.success && hourlyData.chart) {
      renderChartFromServer('chart-error-hourly', hourlyData.chart, true);
    } else if (hourlyData.chart) {
      // fallback 错误图表
      renderChartFromServer('chart-error-hourly', hourlyData.chart, true);
    } else if (hourlyData.error && hourlyDiv) {
      hourlyDiv.innerHTML = '';
      hourlyDiv.textContent = hourlyData.error;
    }
    setFooter(`${type === 'charging' ? '充电' : '光伏'}回测完成`, true);
  } catch (e) {
    console.error(e);
    if (chartDiv) chartDiv.textContent = '回测失败';
    if (errorDiv) errorDiv.textContent = '请求失败';
    if (hourlyDiv) hourlyDiv.textContent = '请求失败';
    setFooter(e.message, false);
  }
}

// ──────────────────────────────────────────────
// Tab 4: 气象数据
// ──────────────────────────────────────────────
function initWeatherTab() {
  safeOn('btn-refresh-weather', 'click', loadWeather);
}

async function loadWeather() {
  const display = $('weather-display');
  if (!display) return;
  display.innerHTML = '<div class="spinner"></div><p>加载中…</p>';
  try {
    const data = await apiCall('api_weather');
    if (data.success) {
      STATE.weatherData = data.weather;
      renderWeather(data);
      setFooter('气象数据加载完成', true);
    } else {
      display.innerHTML = `<p class="error">${data.error || '加载失败'}</p>`;
      setFooter(data.error || '气象数据加载失败', false);
    }
  } catch (e) {
    console.error(e);
    display.innerHTML = `<p class="error">${e.message}</p>`;
    setFooter(e.message, false);
  }
}

function renderWeather(data) {
  const summaryHtml = data.summary_html || '';
  const w = data.weather || {};

  // 预报 HTML 表格（后端已渲染为 HTML）
  let html = summaryHtml;
  if (w.forecast_html) {
    html += '<hr><h3>未来 3h 预报</h3>' + w.forecast_html;
  }
  if (w.warning_msg) {
    html = w.warning_msg + html;
  }

  const display = $('weather-display');
  if (display) display.innerHTML = html;

  // 渲染辐照度趋势图
  const chartDiv = $('weather-chart');
  if (data.chart && chartDiv) {
    chartDiv.innerHTML = '';  // 确保清除 spinner
    Plotly.purge('weather-chart');
    const layout = adaptServerChartTheme(JSON.parse(JSON.stringify(data.chart.layout || {})));
    Plotly.newPlot('weather-chart', data.chart.data, layout, { responsive: true, displayModeBar: false });
  } else if (chartDiv) {
    chartDiv.innerHTML = '';
  }
}

// ──────────────────────────────────────────────
// Tab 5: 策略建议
// ──────────────────────────────────────────────
function initStrategyTab() {
  safeOn('btn-generate-strategy', 'click', generateStrategy);
}

async function generateStrategy() {
  const display = $('strategy-display');
  if (!display) return;

  // 检查是否有预测结果
  if (!STATE.predictResult || !STATE.predictResult.strategy) {
    display.innerHTML = '<p style="color: var(--warning);">尚未执行预测，请先在「联合预测」Tab 中执行预测。</p>';
    return;
  }

  const strategy = STATE.predictResult.strategy;
  if (!strategy || strategy.length === 0) {
    display.innerHTML = '<p style="color: var(--warning);">策略数据为空，请重新执行预测。</p>';
    return;
  }

  renderStrategyTable(strategy);
  setFooter('策略建议已生成', true);
}

function renderStrategyTable(strategy) {
  const display = $('strategy-display');
  if (!display) return;
  if (!strategy || strategy.length === 0) {
    display.innerHTML = '<p>无策略数据</p>';
    return;
  }

  let html = '<table class="strategy-table"><thead><tr>';
  html += '<th>时间</th><th>光伏 (kW)</th><th>负荷 (kW)</th><th>净负荷 (kW)</th><th>调度动作</th>';
  html += '</tr></thead><tbody>';

  strategy.forEach(row => {
    const actionClass = row.action.includes('充电') ? 'action-charge'
      : row.action.includes('放电') ? 'action-discharge'
      : 'action-idle';
    html += '<tr>';
    html += `<td>${row.time}</td>`;
    html += `<td>${row.solar.toFixed(1)}</td>`;
    html += `<td>${row.load.toFixed(1)}</td>`;
    html += `<td>${row.net.toFixed(1)}</td>`;
    html += `<td class="${actionClass}">${row.action}</td>`;
    html += '</tr>';
  });

  html += '</tbody></table>';
  display.innerHTML = html;
}

function renderStrategyDisplay() {
  // 重绘策略数据（主题切换时）
  if (STATE.predictResult && STATE.predictResult.strategy) {
    renderStrategyTable(STATE.predictResult.strategy);
  }
}