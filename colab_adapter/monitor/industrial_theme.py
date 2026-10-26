"""Industrial-inspired dual-theme shell for the existing read-only Colab monitor.

Visual reference: Wincasoft, "Industrial Turbine Monitoring Dashboard UI"
https://dribbble.com/shots/27176071-Industrial-Turbine-Monitoring-Dashboard-UI

The original composition is interpreted using first-party CSS shapes, not copied
artwork. This module intentionally changes presentation only: the monitoring
endpoints, IDs, browser token handling, data semantics and CSP are preserved.
"""
from __future__ import annotations

STYLE = r"""
    /* COLAB CONTROL / original industrial-inspired design tokens */
    :root, :root[data-theme="light"] {
      color-scheme: light;
      --bg: #eaf2f3;
      --card-bg: #f5f9fa;
      --card-header: #e8f1f3;
      --border: #d8e6e9;
      --border-light: #b7cfd4;
      --text: #36515b;
      --text-bright: #051921;
      --text-dim: #647e88;
      --accent: #335fb8;
      --accent-hover: #215a6e;
      --green: #127a68;
      --yellow: #a96e12;
      --red: #bc414e;
      --purple: #6a58b9;
      --shell: #fcfefe;
      --glass: rgba(244, 250, 251, 0.94);
      --hero-from: #318baa;
      --hero-to: #12495d;
      --hero-ink: #effbff;
      --shadow: 0 21px 65px rgba(3, 31, 43, 0.11);
      --focus: #335fb8;
      --font-sans: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      --font-mono: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
    }
    :root[data-theme="dark"] {
      color-scheme: dark;
      --bg: #071922;
      --card-bg: #122a34;
      --card-header: #183541;
      --border: #294752;
      --border-light: #3d626e;
      --text: #bbd2da;
      --text-bright: #f1f9fc;
      --text-dim: #91aeb8;
      --accent: #77a9f7;
      --accent-hover: #a7c9fb;
      --green: #68d6ba;
      --yellow: #f4c16b;
      --red: #fb8c91;
      --purple: #beabff;
      --shell: #0a202a;
      --glass: rgba(19, 44, 54, 0.94);
      --hero-from: #215b72;
      --hero-to: #0b2e41;
      --hero-ink: #edfaff;
      --shadow: 0 24px 85px rgba(0, 0, 0, 0.3);
      --focus: #a0c5ff;
    }
    @media (prefers-color-scheme: dark) {
      :root:not([data-theme]) {
        color-scheme: dark;
        --bg: #071922; --card-bg: #122a34; --card-header: #183541;
        --border: #294752; --border-light: #3d626e;
        --text: #bbd2da; --text-bright: #f1f9fc; --text-dim: #91aeb8;
        --accent: #77a9f7; --accent-hover: #a7c9fb;
        --green: #68d6ba; --yellow: #f4c16b; --red: #fb8c91; --purple: #beabff;
        --shell: #0a202a; --glass: rgba(19,44,54,.94);
        --hero-from: #215b72; --hero-to: #0b2e41; --hero-ink: #edfaff;
        --shadow: 0 24px 85px rgba(0,0,0,.3); --focus: #a0c5ff;
      }
    }

    html { scroll-behavior: smooth; }
    body {
      min-height: 100vh;
      padding: clamp(12px, 2.4vw, 34px);
      background:
        radial-gradient(ellipse at 4% 0%, rgba(99,190,213,.36), transparent 30%),
        linear-gradient(155deg, #1f6171 0%, #173f52 32%, var(--bg) 70%);
      font-family: var(--font-sans);
      line-height: 1.55;
      color: var(--text);
    }
    .container {
      max-width: 1550px;
      padding: clamp(18px, 2.6vw, 42px);
      background: var(--shell);
      border: 1px solid var(--border);
      border-radius: clamp(20px, 2.8vw, 36px);
      box-shadow: var(--shadow);
      min-height: calc(100vh - 68px);
      overflow: clip;
    }
    header {
      border-bottom: 0;
      align-items: center;
      gap: 16px 24px;
      padding: 0 0 16px;
      margin-bottom: 8px;
    }
    .header-brand h1 {
      font-size: clamp(1.25rem, 1.7vw, 1.7rem);
      font-weight: 850;
      letter-spacing: -.055em;
      line-height: 1.15;
    }
    .brand-accent { color: #2c6c83; }
    :root[data-theme="dark"] .brand-accent { color: #86c4d7; }
    .header-brand .subtitle {
      margin-top: 5px;
      font-size: .76rem;
      font-weight: 550;
      letter-spacing: .055em;
    }
    .header-actions { gap: 7px; }
    .header-meta { font-size: .74rem; }
    .header-actions .filter-input {
      min-height: 38px;
      border-radius: 12px;
      max-width: 145px !important;
    }
    .btn {
      min-height: 38px;
      padding: .5rem .9rem;
      border-radius: 12px;
      border-color: var(--border);
      font-family: var(--font-sans);
      background: var(--card-bg);
      color: var(--text-bright);
      font-weight: 620;
      box-shadow: none;
    }
    .btn-sm { min-height: 36px; padding: .43rem .75rem; }
    .btn:hover, .btn-secondary:hover {
      background: var(--card-header);
      color: var(--text-bright);
      border-color: var(--border-light);
    }
    .btn-primary, .btn-success {
      background: #215a6e;
      color: #fff;
      border-color: #215a6e;
    }
    .btn-primary:hover, .btn-success:hover { background: #2c6c83; color: #fff; }
    .btn:focus-visible, a:focus-visible, input:focus-visible, select:focus-visible {
      outline: 3px solid var(--focus);
      outline-offset: 3px;
    }
    .theme-toggle {
      white-space: nowrap;
      background: var(--card-bg);
      border-color: var(--border-light);
    }
    .section-nav {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      padding: 9px 0 24px;
    }
    .section-nav a {
      text-decoration: none;
      color: var(--text-dim);
      background: transparent;
      border: 1px solid transparent;
      padding: 8px 14px;
      font-size: .82rem;
      font-weight: 680;
      border-radius: 999px;
      transition: background .18s, color .18s;
    }
    .section-nav a:hover, .section-nav a:focus-visible {
      background: var(--card-header);
      color: var(--text-bright);
    }
    .section-nav a:first-child {
      color: #fff;
      background: #215a6e;
    }
    .section { margin-bottom: clamp(22px, 2.8vw, 42px); scroll-margin-top: 18px; }
    .section-header { margin-bottom: 14px; }
    .section-header h2 {
      font-size: clamp(1.05rem, 1.3vw, 1.35rem);
      font-weight: 760;
      letter-spacing: -.035em;
      color: var(--text-bright);
    }
    .section-desc { display: block; margin-top: 3px; font-size: .8rem; }
    .card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 19px;
      padding: clamp(15px, 1.5vw, 24px);
      box-shadow: 0 2px 9px rgba(7,36,46,.015);
    }
    .card-header { align-items: flex-start; }
    .card-title-group h3, .card-header h3 {
      font-size: 1rem;
      color: var(--text-bright);
      font-weight: 750;
      letter-spacing: -.025em;
    }
    .badge { border-radius: 999px; letter-spacing: .045em; padding: 5px 10px; }
    .badge-info {
      background: rgba(51,95,184,.13);
      color: var(--accent);
      border-color: rgba(51,95,184,.19);
    }
    .badge-neutral {
      background: var(--card-header);
      border-color: var(--border);
      color: var(--text-dim);
    }
    .badge-success {
      background: rgba(18,122,104,.1);
      color: var(--green);
      border-color: rgba(18,122,104,.26);
    }
    .badge-warning {
      background: rgba(203,140,32,.11);
      color: var(--yellow);
      border-color: rgba(203,140,32,.27);
    }
    .badge-danger {
      background: rgba(209,71,82,.10);
      color: var(--red);
      border-color: rgba(209,71,82,.27);
    }
    .banner { border-radius: 14px; }
    .banner-danger { color: var(--red); }
    .banner-warning { color: var(--yellow); }

    /* Editorial industrial hero: entirely decorative, no fabricated runtime data. */
    .overview-hero {
      display: grid;
      grid-template-columns: minmax(0, 1.12fr) minmax(0, 1fr);
      gap: 14px;
      margin-bottom: 15px;
    }
    .hero-visual {
      min-width: 0;
      min-height: clamp(340px, 32vw, 455px);
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      position: relative;
      isolation: isolate;
      overflow: hidden;
      border-radius: 24px;
      padding: clamp(22px, 2.4vw, 36px);
      color: var(--hero-ink);
      background:
        radial-gradient(circle at 76% 36%, rgba(225,246,255,.32) 0, transparent 28%),
        radial-gradient(circle at 16% 94%, rgba(125,220,250,.40), transparent 46%),
        linear-gradient(135deg, var(--hero-from), var(--hero-to));
    }
    .hero-visual::before {
      content: "";
      position: absolute;
      inset: 0;
      z-index: -1;
      opacity: .20;
      background-image: linear-gradient(rgba(255,255,255,.3) 1px, transparent 1px),
                        linear-gradient(90deg, rgba(255,255,255,.3) 1px, transparent 1px);
      background-size: 44px 44px;
      mask-image: linear-gradient(105deg, transparent 10%, black 94%);
    }
    .hero-visual::after {
      content: "";
      pointer-events: none;
      position: absolute;
      width: 450px; height: 450px;
      border: 1px solid rgba(255,255,255,.22);
      border-radius: 50%;
      right: -160px; top: -140px;
      box-shadow: 0 0 0 42px rgba(255,255,255,.035), 0 0 0 90px rgba(255,255,255,.025);
    }
    .hero-topline {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      position: relative;
      z-index: 2;
    }
    .hero-label {
      display: inline-block;
      padding: 7px 12px;
      border: 1px solid rgba(255,255,255,.34);
      border-radius: 999px;
      backdrop-filter: blur(8px);
      background: rgba(0,20,39,.09);
      letter-spacing: .14em;
      font-size: .65rem;
      font-weight: 800;
    }
    .hero-topline .hero-number { font-family: var(--font-mono); font-size: .73rem; opacity: .77; }
    .hero-art {
      position: absolute;
      width: clamp(205px, 26vw, 330px);
      aspect-ratio: 1;
      right: 1.8%;
      top: 12%;
      display: grid;
      place-items: center;
      pointer-events: none;
      opacity: .85;
    }
    .hero-ring {
      position: absolute;
      border: 1px solid rgba(230,250,255,.6);
      border-radius: 50%;
      width: 84%; height: 84%;
      box-shadow: 0 0 35px rgba(136,224,248,.22), inset 0 0 33px rgba(136,224,248,.2);
    }
    .hero-ring::before, .hero-ring::after {
      content: "";
      position: absolute;
      inset: 10%;
      border-radius: 50%;
      border: 1px dashed rgba(228,250,255,.57);
    }
    .hero-ring::after { inset: 24%; border-style: solid; opacity: .65; }
    .hero-core {
      width: 37%; height: 37%;
      border-radius: 50%;
      display: grid;
      place-items: center;
      border: 1px solid rgba(241,252,255,.67);
      background: radial-gradient(circle at 30% 24%, rgba(235,252,255,.42), rgba(4,57,84,.51));
      box-shadow: 0 0 0 12px rgba(228,247,252,.06), 0 0 40px rgba(210,245,255,.25);
      font-family: var(--font-mono);
      font-size: clamp(1rem, 2vw, 1.45rem);
      letter-spacing: .17em;
      font-weight: 650;
    }
    .hero-copy { position: relative; z-index: 2; max-width: 57%; }
    .hero-eyebrow {
      color: rgba(233,250,255,.82);
      letter-spacing: .18em;
      font-size: .64rem;
      font-weight: 800;
      margin-bottom: 11px;
    }
    .hero-copy h3 {
      color: #fff;
      font-size: clamp(2.25rem, 4.3vw, 4.65rem);
      font-weight: 800;
      letter-spacing: -.073em;
      line-height: 1.02;
      max-width: 540px;
    }
    .hero-copy p:last-child {
      margin-top: 16px;
      max-width: 310px;
      font-size: .84rem;
      line-height: 1.55;
      color: rgba(241,250,255,.9);
    }
    .hero-chart { display: flex; min-width: 0; }
    .hero-chart .chart-card {
      width: 100%;
      min-height: clamp(340px, 32vw, 455px);
      border-radius: 24px;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      background: var(--card-bg);
    }
    .hero-chart .chart-container {
      min-height: 260px;
      display: grid;
      place-items: center;
      width: 100%;
      padding-top: 14px;
    }
    .hero-chart .chart-svg { width: 100%; max-width: 100%; }

    .kpi-grid {
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 13px;
      margin-bottom: 0;
    }
    .kpi-card {
      min-width: 0;
      min-height: 154px;
      padding: 20px 18px;
      position: relative;
      display: flex;
      flex-direction: column;
      justify-content: center;
      overflow: hidden;
    }
    .kpi-card::before {
      content: "";
      position: absolute;
      top: 19px; right: 19px;
      width: 8px; height: 8px;
      border: 2px solid #5d97b3;
      border-radius: 50%;
      opacity: .85;
    }
    .kpi-label { font-size: .71rem; letter-spacing: .08em; line-height: 1.35; margin-bottom: 16px; }
    .kpi-value {
      font-family: var(--font-sans);
      font-size: clamp(1.28rem, 2.15vw, 2rem);
      font-weight: 800;
      letter-spacing: -.05em;
      margin-bottom: 7px;
      overflow-wrap: anywhere;
    }
    .kpi-meta { font-size: .73rem; }
    .filter-bar {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 12px;
      margin-bottom: 13px;
    }
    .filter-group input, .filter-group select, .filter-input {
      color: var(--text-bright);
      background: var(--shell);
      border: 1px solid var(--border-light);
      border-radius: 10px;
      min-height: 37px;
    }
    .table-card { padding: 8px 14px; }
    .table-responsive { overflow-x: auto; border-radius: 13px; }
    .data-table { width: 100%; border-collapse: separate; border-spacing: 0; }
    .data-table thead th {
      background: var(--card-header);
      color: var(--text-dim);
      font-size: .67rem;
      letter-spacing: .07em;
      font-weight: 760;
      white-space: nowrap;
      padding: 15px 12px;
      text-transform: uppercase;
    }
    .data-table thead th:first-child { border-radius: 10px 0 0 10px; }
    .data-table thead th:last-child { border-radius: 0 10px 10px 0; }
    .data-table tbody tr:hover { background: var(--card-header); }
    .data-table tbody td {
      padding: 15px 12px;
      font-size: .79rem;
      border-bottom-color: var(--border);
      color: var(--text);
    }
    .data-table td strong { color: var(--text-bright); }
    .inspector-meta-card { background: var(--card-header); }
    .inspector-card { border-radius: 18px; }
    .log-console {
      border-radius: 13px;
      background: #061823;
      border: 1px solid #325462;
      color: #a8e4cb;
    }
    .sha-tag, .sha-badge {
      background: var(--card-header);
      color: var(--text);
      border-radius: 8px;
    }
    .empty-panel {
      border: 1px dashed var(--border-light);
      background: var(--card-bg);
      border-radius: 20px;
    }
    footer { font-size: .76rem; border-color: var(--border); }

    /* Legacy charts use presentation attrs; theme them without changing data geometry. */
    .chart-svg [stroke="#58a6ff"] { stroke: var(--accent); }
    .chart-svg [stroke="#3fb950"] { stroke: var(--green); }
    .chart-svg [stroke="#30363d"], .chart-svg [stroke="#21262d"] { stroke: var(--border-light); }
    .chart-svg [fill="#8b949e"] { fill: var(--text-dim); }
    .chart-svg [fill="#c9d1d9"] { fill: var(--text-bright); }

    @media (max-width: 1240px) {
      .kpi-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
      .overview-hero { grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); }
      .hero-copy { max-width: 65%; }
    }
    @media (max-width: 900px) {
      .overview-hero { grid-template-columns: 1fr; }
      .hero-visual { min-height: 355px; }
      .hero-art { right: 5%; top: 0%; width: 300px; }
      .hero-chart .chart-card { min-height: 330px; }
      .kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .header-actions { width: 100%; }
    }
    @media (max-width: 600px) {
      body { padding: 8px; }
      .container { padding: 15px; min-height: calc(100vh - 16px); border-radius: 23px; }
      header { flex-direction: column; align-items: stretch; }
      .header-actions { justify-content: flex-start; gap: 6px; }
      .header-meta { width: 100%; }
      .header-actions .filter-input { max-width: 100% !important; flex: 1; }
      .section-nav { flex-wrap: nowrap; overflow-x: auto; padding: 6px 0 20px; }
      .section-nav a { white-space: nowrap; }
      .hero-visual { min-height: 340px; padding: 20px; }
      .hero-art { width: 260px; right: -55px; top: 18%; opacity: .52; }
      .hero-copy { max-width: 86%; }
      .hero-copy h3 { font-size: 2.8rem; }
      .hero-chart .chart-card { min-height: 300px; }
      .kpi-grid { grid-template-columns: 1fr 1fr; gap: 8px; }
      .kpi-card { min-height: 143px; padding: 13px; }
      .kpi-card::before { top: 12px; right: 12px; }
      .kpi-label { margin-bottom: 12px; font-size: .64rem; }
      .kpi-value { font-size: 1.35rem; }
      .kpi-meta { font-size: .67rem; }
      .inspector-grid { grid-template-columns: 1fr; }
      .data-table { min-width: 620px; }
    }
    @media (prefers-reduced-motion: reduce) {
      html { scroll-behavior: auto; }
      *, *::before, *::after { transition-duration: .01ms !important; animation-duration: .01ms !important; }
    }
"""

HERO = r"""
      <div class="overview-hero">
        <div class="hero-visual" aria-label="Remote GPU workspace monitoring overview">
          <div class="hero-topline">
            <span class="hero-label">COLAB / WORKSPACE</span>
            <span class="hero-number">MONITOR — 02</span>
          </div>
          <div class="hero-art" aria-hidden="true">
            <div class="hero-ring"></div>
            <div class="hero-core">GPU</div>
          </div>
          <div class="hero-copy">
            <p class="hero-eyebrow">RESEARCH COMPUTE / OBSERVABILITY</p>
            <h3>Compute,<br>in full view.</h3>
            <p>Track remote GPU leases, research runs and verified artifacts from one read-only workspace.</p>
          </div>
        </div>
        <div class="hero-chart">
          __COMPUTE_CHART__
        </div>
      </div>
"""

SCRIPT = r"""
    /* Presentation only: never writes server-side state or reads token values. */
    const MONITOR_THEME_KEY = 'colab-monitor-color-theme';
    function initialMonitorTheme() {
      try {
        const stored = window.localStorage.getItem(MONITOR_THEME_KEY);
        if (stored === 'light' || stored === 'dark') return stored;
      } catch (_) { /* private-mode storage restrictions */ }
      try {
        return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
      } catch (_) { return 'light'; }
    }
    function applyMonitorTheme(theme, persist) {
      const resolved = theme === 'dark' ? 'dark' : 'light';
      document.documentElement.setAttribute('data-theme', resolved);
      const toggle = document.getElementById('theme-toggle');
      if (toggle) {
        toggle.textContent = resolved === 'dark' ? '☀ Light mode' : '☾ Dark mode';
        toggle.setAttribute('aria-label', resolved === 'dark' ? 'Switch to light theme' : 'Switch to dark theme');
        toggle.setAttribute('aria-pressed', resolved === 'dark' ? 'true' : 'false');
      }
      if (persist) {
        try { window.localStorage.setItem(MONITOR_THEME_KEY, resolved); }
        catch (_) { /* presentation still works without persistence */ }
      }
    }
    function initMonitorTheme() {
      applyMonitorTheme(initialMonitorTheme(), false);
      const toggle = document.getElementById('theme-toggle');
      if (toggle) {
        toggle.addEventListener('click', function () {
          const current = document.documentElement.getAttribute('data-theme');
          applyMonitorTheme(current === 'dark' ? 'light' : 'dark', true);
        });
      }
    }
    initMonitorTheme();
"""


def apply_industrial_theme(source: str) -> str:
    """Compose the industrial dashboard without touching backend API or element IDs."""
    css_anchor = "  </style>"
    header_anchor = "        <h1>Colab Monitor</h1>"
    actions_anchor = '        <button id="btn-refresh"'
    banner_anchor = "    <!-- Global Banners -->"
    chart_start_anchor = "      <!-- Compute Balance History Chart -->"
    chart_end_anchor = "    </section>"
    grid_anchor = '      <div class="kpi-grid">'
    script_anchor = "    // State management"
    expected = [css_anchor, header_anchor, actions_anchor, banner_anchor,
                chart_start_anchor, grid_anchor, script_anchor]
    for marker in expected:
        if source.count(marker) != 1:
            raise RuntimeError("Dashboard markup drift: expected one " + marker)
    chart_start = source.index(chart_start_anchor)
    chart_end = source.index(chart_end_anchor, chart_start)
    chart = source[chart_start:chart_end]
    if chart.count('id="compute-chart-container"') != 1:
        raise RuntimeError("Dashboard chart markup changed")
    source = source[:chart_start] + source[chart_end:]
    source = source.replace(grid_anchor, HERO.replace("__COMPUTE_CHART__", chart) + "\n" + grid_anchor, 1)
    source = source.replace(header_anchor,
                            '<h1>COLAB <span class="brand-accent">/ CONTROL</span></h1>', 1)
    source = source.replace(
        '        <p class="subtitle">V2 Remote GPU Workspaces & Durable Artifacts</p>',
        '        <p class="subtitle">REMOTE GPU OBSERVABILITY &nbsp; / &nbsp; V2</p>', 1)
    source = source.replace(actions_anchor,
                            '        <button id="theme-toggle" type="button" class="btn btn-sm theme-toggle" aria-label="Switch color theme" aria-pressed="false">☾ Dark mode</button>\n'
                            + actions_anchor, 1)
    source = source.replace(banner_anchor,
                            '''    <nav class="section-nav" aria-label="Monitor sections">
      <a href="#section-compute">Overview</a>
      <a href="#section-sessions">GPU sessions</a>
      <a href="#section-runs">Training runs</a>
      <a href="#section-inspector">Run inspector</a>
    </nav>
\n''' + banner_anchor, 1)
    source = source.replace(css_anchor, STYLE + "\n" + css_anchor, 1)
    source = source.replace(script_anchor, SCRIPT + "\n" + script_anchor, 1)
    source = source.replace('  <title>Colab Experiment &amp; Compute Monitor</title>',
                            '  <title>Colab / Control — Research Compute Monitor</title>', 1)
    if source.count('id="compute-chart-container"') != 1:
        raise RuntimeError("Dashboard chart must remain unique")
    return source
