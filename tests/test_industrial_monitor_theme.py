"""Regression contracts for the industrial light/dark monitor redesign.

Presentation-only: no Colab allocation, Drive write or live API access.
"""
from __future__ import annotations

import re
import unittest
from html.parser import HTMLParser

from colab_adapter.monitor.ui import (
    DASHBOARD_HTML,
    get_dashboard_css,
    get_dashboard_js,
    render_dashboard_html,
)


class _IdCollector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = []
        self.inline_handlers = []

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key == "id":
                self.ids.append(value)
            if key.lower().startswith("on") and key.lower() not in ("once",):
                self.inline_handlers.append((tag, key))


class IndustrialThemeContracts(unittest.TestCase):
    def test_both_themes_use_reference_palette_and_accessible_contrast_variables(self):
        css = get_dashboard_css()
        self.assertIn(':root[data-theme="light"]', css)
        self.assertIn(':root[data-theme="dark"]', css)
        for color in ("#d8e6e9", "#051921", "#2c6c83", "#215a6e", "#335fb8"):
            self.assertIn(color, css.lower())
        self.assertIn("prefers-reduced-motion", css)
        self.assertIn("max-width: 600px", css)

    def test_original_monitor_data_ids_and_inspector_are_preserved(self):
        parser = _IdCollector()
        parser.feed(DASHBOARD_HTML)
        all_ids = set(parser.ids)
        for component_id in (
            "val-cu", "val-rate", "val-assignments", "val-runs",
            "val-collector-health", "compute-chart-container",
            "sessions-tbody", "runs-tbody", "artifacts-tbody",
            "section-compute", "section-sessions", "section-runs",
            "section-inspector", "run-metrics-chart-container",
            "log-console", "auth-token-input", "btn-set-token",
            "btn-export-csv", "btn-refresh", "health-badge",
            "filter-project", "filter-status",
        ):
            self.assertIn(component_id, all_ids, component_id)
        self.assertEqual(len(parser.ids), len(all_ids), "Duplicate HTML id")
        self.assertFalse(parser.inline_handlers, "No inline event handlers")

    def test_theme_toggle_uses_event_listener_and_non_sensitive_preference_storage(self):
        js = get_dashboard_js()
        self.assertIn("initMonitorTheme", js)
        self.assertIn("theme-toggle", js)
        self.assertIn("localStorage.setItem(MONITOR_THEME_KEY, resolved)", js)
        self.assertIn("prefers-color-scheme: dark", js)
        self.assertIn("aria-pressed", js)
        self.assertNotIn("innerHTML", js)
        self.assertIn("function refreshAll()", js)
        self.assertIn("function renderSessionsTable", js)
        self.assertEqual(DASHBOARD_HTML.count('id="theme-toggle"'), 1)

    def test_hero_does_not_fabricate_account_telemetry(self):
        hero = DASHBOARD_HTML.split('<div class="overview-hero">', 1)[1].split(
            '<div class="kpi-grid">', 1
        )[0]
        self.assertIn("Remote GPU workspace monitoring", hero)
        self.assertEqual(hero.count('id="compute-chart-container"'), 1)
        self.assertNotIn("186.69", hero)
        self.assertNotIn("active_assignments\": 0", hero)

    def test_csp_nonce_integrity_and_no_external_assets(self):
        html = render_dashboard_html(csp_nonce="nonceA123")
        self.assertEqual(html.count('nonce="nonceA123"'), 2)
        self.assertNotIn("__CSP_NONCE__", html)
        self.assertIn("<style", html)
        self.assertIn("<script", html)
        self.assertNotIn("<script src=", html)
        self.assertEqual(DASHBOARD_HTML.count("<script "), 1)
        self.assertEqual(DASHBOARD_HTML.count("<style "), 1)
        self.assertIn("MONITOR_THEME_KEY", html)

    def test_read_only_paths_and_browser_token_contract_preserved(self):
        js = get_dashboard_js()
        self.assertIn("sessionStorage.getItem('colab_monitor_token')", js)
        self.assertIn("sessionStorage.setItem('colab_monitor_token', token)", js)
        for endpoint in (
            "/api/v1/health", "/api/v1/summary",
            "/api/v1/compute/samples?limit=240", "/api/v1/sessions",
            "/api/v1/runs",
        ):
            self.assertIn(endpoint, js)
        self.assertNotIn("fetch('/api/v1/alloc", js)
        self.assertNotIn("fetch('/api/v1/stop", js)


if __name__ == "__main__":
    unittest.main()
