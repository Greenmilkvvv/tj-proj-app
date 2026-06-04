/* ============================================================
   光储充智能预测系统 — 应用主逻辑
   ============================================================ */

var App = (function() {
  'use strict';

  /* ---------- 配置 ---------- */
  var API_BASE = '/api/';
  var REFRESH_INTERVAL = 30000; // 30s 自动刷新气象
  var currentView = 'combined';
  var latestResult = null;

  /* ---------- 工具函数 ---------- */
  function $_(id) { return document.getElementById(id); }

  function showLoading() {
    $_('loading-overlay').classList.remove('hidden');
  }
  function hideLoading() {
    $_('loading-overlay').classList.add('hidden');
  }

  function setStatus(online) {
    var dot = $_('connection-status');
    var text = $_('status-text');
    if (online) {
      dot.className = 'status-dot online';
      text.textContent = '在线';
    } else {
      dot.className = 'status-dot offline';
      text.textContent = '离线';
    }
  }

  function updateClock() {
    var now = new Date();
    var s = now.getHours().toString().padStart(2, '0')
      + ':' + now.getMinutes().toString().padStart(2, '0')
      + ':' + now.getSeconds().toString().padStart(2, '0');
    var el = $_('current-time');
    if (el) el.textContent = s;
  }

  /* ---------- API 请求封装 ---------- */
  function apiCall(endpoint, data) {
    data = data || {};
    return fetch(API_BASE + endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    })
      .then(function(r) {
        if (!r.ok) throw new Error('HTTP ' + r.status);
        return r.json();
      })
      .then(function(d) {
        setStatus(true);
        return d;
      })
      .catch(function(err) {
        console.error('API error:', err);
        setStatus(false);
        throw err;
      });
  }

  /* ---------- 健康检查 ---------- */
  function checkHealth() {
    return apiCall('health_check')
      .then(function(d) {
        updateModelStatus(d);
        return d;
      })
      .catch(function() {
        setStatus(false);
      });
  }

  /* ---------- 气象数据 ---------- */
  function fetchWeather() {
    return apiCall('weather')
      .then(function(d) {
        updateWeatherUI(d);
        Charts.updateWeatherChart(d);
        setWeatherBadge('实时');
        return d;
      })
      .catch(function() {
        setWeatherBadge('离线');
      });
  }

  /* ---------- 执行预测 ---------- */
  function runPrediction() {
    var stepBtns = document.querySelectorAll('#step-group .btn-option');
    var nSteps = 96;
    stepBtns.forEach(function(b) {
      if (b.classList.contains('active')) nSteps = parseInt(b.dataset.steps);
    });

    var price = parseFloat($_('price-input').value) || 0.8;
    var load = parseFloat($_('load-input').value) || 875;

    showLoading();
    setStatus(true);

    return apiCall('predict', {
      n_steps: nSteps,
      current_price: price,
      current_load: load,
    })
      .then(function(d) {
        latestResult = d;
        updatePredictionUI(d);
        Charts.updateMainChart(d.prediction, currentView);
        Charts.updatePriceChart(d.price_curve);
        hideLoading();
        return d;
      })
      .catch(function(err) {
        hideLoading();
        setStatus(false);
        console.error('预测失败:', err);
      });
  }

  /* ---------- UI 更新函数 ---------- */
  function updateWeatherUI(d) {
    if (!d) return;

    var temp = d.temperature;
    if (temp !== undefined && temp !== null) {
    $_('weather-temp').textContent = temp.toFixed(1) + '°C';
    }
    var desc = d.description || '';
    $_('weather-desc').textContent = desc || '--';
    $_('weather-rad').textContent = (d.current_radiation != null ? Math.round(d.current_radiation) : '--') + ' W/m²';
    $_('weather-cloud').textContent = (d.current_cloudcover != null ? Math.round(d.current_cloudcover) : '--') + '%';
    $_('weather-rain').textContent = (d.current_rain != null ? d.current_rain.toFixed(1) : '--') + ' mm';

    /* 天气图标 */
    var icon = $_('weather-icon');
    if (icon) {
      var radiation = d.current_radiation || 0;
      if (radiation > 600) icon.innerHTML = '&#x2600;&#xFE0F;';
      else if (radiation > 200) icon.innerHTML = '&#x26C5;';
      else icon.innerHTML = '&#x2601;&#xFE0F;';
    }

    /* 恶劣天气预警 */
    var alertEl = $_('weather-warning');
    if (alertEl) {
      var cloud = d.current_cloudcover || 0;
      var rain = d.current_rain || 0;
      if (cloud > 80 || rain > 8) {
        alertEl.classList.remove('hidden');
      } else {
        alertEl.classList.add('hidden');
      }
    }
  }

  function setWeatherBadge(text) {
    var b = $_('weather-badge');
    if (b) b.textContent = text;
  }

  function updatePredictionUI(d) {
    var p = d.prediction;
    var s = d.strategy;

    /* KPI 卡片 */
    var totalSolar = p.solar ? p.solar.reduce(function(a, b) { return a + b; }, 0) * 0.25 : 0; // 15min per step -> h
    var totalLoad = p.load_mean ? p.load_mean.reduce(function(a, b) { return a + b; }, 0) * 0.25 : 0;
    var greenRate = totalLoad > 0 ? Math.min(100, (totalSolar / totalLoad * 100)) : 0;
    var peakSolar = p.solar ? Math.max.apply(null, p.solar) : 0;

    $_('kpi-solar').innerHTML = totalSolar.toFixed(1) + ' <small>kWh</small>';
    $_('kpi-load').innerHTML = totalLoad.toFixed(1) + ' <small>kWh</small>';
    $_('kpi-green').innerHTML = greenRate.toFixed(1) + ' <small>%</small>';
    $_('kpi-peak').innerHTML = peakSolar.toFixed(1) + ' <small>kW</small>';

    /* 策略建议 */
    var sc = $_('strategy-content');
    if (sc && s) {
      var html = '';
      if (s.recommendations && s.recommendations.length) {
        s.recommendations.forEach(function(r) {
          html += '<div class="strategy-item"><i class="fas fa-check-circle surplus-tag"></i>' + r + '</div>';
        });
      }
      if (s.warnings && s.warnings.length) {
        s.warnings.forEach(function(w) {
          html += '<div class="strategy-item"><i class="fas fa-exclamation-triangle deficit-tag"></i>' + w + '</div>';
        });
      }
      if (!html) html = '<p class="placeholder-text">暂无策略建议</p>';
      sc.innerHTML = html;
    }

    /* 能量平衡 */
    var bc = $_('balance-content');
    if (bc && p) {
      var surplus = p.surplus_periods || [];
      var deficit = p.deficit_periods || [];
      var html = '';
      if (surplus.length) {
        html += '<div class="balance-item"><i class="fas fa-sun surplus-tag"></i><b>盈余时段 (' + surplus.length + '):</b> ' + surplus.slice(0, 3).join(', ') + (surplus.length > 3 ? '...' : '') + '</div>';
      }
      if (deficit.length) {
        html += '<div class="balance-item"><i class="fas fa-bolt deficit-tag"></i><b>缺口时段 (' + deficit.length + '):</b> ' + deficit.slice(0, 3).join(', ') + (deficit.length > 3 ? '...' : '') + '</div>';
      }
      if (!html) html = '<p class="placeholder-text">能量平衡数据暂无</p>';
      bc.innerHTML = html;
    }
  }

  function updateModelStatus(d) {
    if (!d) return;
    var svModel = d.solar_model_loaded !== undefined
      ? (d.solar_model_loaded ? 'ok' : 'fail')
      : (d.models && d.models.solar_forecast ? 'ok' : 'fail');
    var cvModel = d.charging_model_loaded !== undefined
      ? (d.charging_model_loaded ? 'ok' : 'fail')
      : (d.models && d.models.charging_forecast ? 'ok' : 'fail');
    var wvStatus = d.weather_available !== undefined
      ? (d.weather_available ? 'ok' : 'fail')
      : (d.weather_ok ? 'ok' : 'fail');

    setTag($_('model-solar-status'), svModel);
    setTag($_('model-charging-status'), cvModel);
    setTag($_('model-weather-status'), wvStatus);

    function setTag(el, cls) {
      if (!el) return;
      el.className = 'status-tag ' + cls;
      el.textContent = cls === 'ok' ? '就绪' : '未加载';
    }
  }

  /* ---------- 主题变更时刷新图表颜色 ---------- */
  function onThemeChange() {
    Charts.refreshTheme();
    if (latestResult) {
      Charts.updateMainChart(latestResult.prediction, currentView);
      Charts.updatePriceChart(latestResult.price_curve);
    }
  }

  /* ---------- 事件绑定 ---------- */
  function bindEvents() {
    /* 预测按钮 */
    var predictBtn = $_('predict-btn');
    if (predictBtn) {
      predictBtn.addEventListener('click', runPrediction);
    }

    /* 步长选择 */
    var stepGroup = $_('step-group');
    if (stepGroup) {
      stepGroup.addEventListener('click', function(e) {
        if (e.target.classList.contains('btn-option')) {
          stepGroup.querySelectorAll('.btn-option').forEach(function(b) {
            b.classList.remove('active');
          });
          e.target.classList.add('active');
        }
      });
    }

    /* 图表视图切换 */
    var chartTabs = document.querySelector('.chart-tabs');
    if (chartTabs) {
      chartTabs.addEventListener('click', function(e) {
        if (e.target.classList.contains('tab-btn')) {
          chartTabs.querySelectorAll('.tab-btn').forEach(function(b) {
            b.classList.remove('active');
          });
          e.target.classList.add('active');
          currentView = e.target.dataset.view;
          if (latestResult) {
            Charts.updateMainChart(latestResult.prediction, currentView);
          }
        }
      });
    }

    /* 刷新按钮 */
    var refreshBtn = $_('refresh-btn');
    if (refreshBtn) {
      refreshBtn.addEventListener('click', function() {
        fetchWeather();
        checkHealth();
      });
    }

    /* 主题切换监听 */
    var themeBtn = $_('theme-toggle');
    if (themeBtn) {
      themeBtn.addEventListener('click', function() {
        /* 等 DOM 属性更新后再刷新图表 */
        setTimeout(onThemeChange, 100);
      });
    }

    /* 监听系统主题变化 */
    if (window.matchMedia) {
      window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', function() {
        setTimeout(onThemeChange, 100);
      });
    }
  }

  /* ---------- 初始化 ---------- */
  function init() {
    bindEvents();
    Charts.initAll();

    /* 时钟更新 */
    updateClock();
    setInterval(updateClock, 10000);

    /* 启动检查 */
    checkHealth();
    fetchWeather();

    /* 定时刷新气象 */
    setInterval(fetchWeather, REFRESH_INTERVAL);

    /* 自动运行一次预测 */
    setTimeout(runPrediction, 800);
  }

  /* DOM ready */
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  /* 暴露公共方法 */
  return {
    runPrediction: runPrediction,
    fetchWeather: fetchWeather,
    checkHealth: checkHealth,
  };
})();