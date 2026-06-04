(function() {
  'use strict';

  var STORAGE_KEY = 'tj-dashboard-theme';
  var DARK_CLASS = 'dark';
  var LIGHT_CLASS = 'light';

  /* 从 localStorage 读取或跟随系统 */
  function getPreferredTheme() {
    var stored = localStorage.getItem(STORAGE_KEY);
    if (stored === DARK_CLASS || stored === LIGHT_CLASS) return stored;
    if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) return DARK_CLASS;
    return LIGHT_CLASS;
  }

  function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    var btn = document.getElementById('theme-toggle');
    if (btn) {
      var icon = btn.querySelector('i');
      if (icon) {
        icon.className = theme === DARK_CLASS ? 'fas fa-sun' : 'fas fa-moon';
      }
    }
  }

  function toggleTheme() {
    var current = document.documentElement.getAttribute('data-theme') || getPreferredTheme();
    var next = current === DARK_CLASS ? LIGHT_CLASS : DARK_CLASS;
    localStorage.setItem(STORAGE_KEY, next);
    applyTheme(next);
  }

  /* 初始化 */
  var theme = getPreferredTheme();
  applyTheme(theme);

  /* 绑定按钮 */
  document.addEventListener('DOMContentLoaded', function() {
    var btn = document.getElementById('theme-toggle');
    if (btn) btn.addEventListener('click', toggleTheme);
  });
})();