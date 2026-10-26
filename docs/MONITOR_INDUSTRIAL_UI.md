# Colab / Control — Industrial dashboard visual system

Reference: https://dribbble.com/shots/27176071-Industrial-Turbine-Monitoring-Dashboard-UI

An original adaptation of the Wincasoft industrial-monitoring composition to GPU research. This implementation does not copy the turbine image or third-party source artwork.

## Visual intent

- Deep-teal exterior frame; spacious light/dark inner shell.
- Large abstract GPU instrument hero beside the actual compute-unit history chart.
- Modular five-KPI strip; session tables, run explorer and inspector retained.
- Fine blue/teal chart styling, gentle rounded cards, and compact section navigation.
- Reference colors: #D7E6E8, #051921, #2C6C83, #215A6E, #335FB8.

## Behavior and preservation

The theme and layout are in colab_adapter/monitor/industrial_theme.py. The existing DASHBOARD_HTML is composed through apply_industrial_theme. No monitoring endpoint or payload changed.

Every original DOM ID, API request, bearer-token flow, CSP nonce, error banner, chart, managed session toggle, run inspector, logs, Drive publication status and artifact integrity state remains in place. Decorative GPU rings contain no fabricated account metrics.

The header toggle respects the system color preference initially, saves a non-sensitive theme choice under the localStorage key colab-monitor-color-theme, updates aria-pressed/aria-label, and works even if persistence is blocked. It does not store credentials.

The layout responds to large, medium and narrow viewports. Narrow screens keep complex data tables scrollable inside their card. Reduced-motion preferences are respected.

## Verification

Offline tests: python3 -m unittest tests.test_industrial_monitor_theme -v.

Real Chromium tests (isolated CI only): COLAB_REQUIRE_BROWSER_TESTS=1 python3 -m unittest tests.test_industrial_theme_browser -v.

Browser checks use synthetic local API fixtures, not Google Colab or Drive. The workflow uploads a short-lived artifact named monitor-industrial-themes containing:

- monitor-light-desktop.png
- monitor-dark-desktop.png
- monitor-light-mobile.png

Review these rendered images before release. The tests check the actual theme switch, reload persistence, CSS difference, local KPIs, mobile overflow, and absence of stop/delete controls.

This UI change provides no cloud permission, billing guarantee or additional provider validation. Retain LIVE_SUCCESS_PATH_VERIFIED_WITH_LIMITATIONS until separate evidence warrants a changed status.
