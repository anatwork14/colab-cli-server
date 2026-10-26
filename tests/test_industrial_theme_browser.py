"""Real-Chromium visual smoke checks for industrial light/dark dashboard.

Runs only in the browser-security CI job. All API responses are isolated,
synthetic loopback fixtures: no Google contact, GPU, Drive or OAuth.
"""
from __future__ import annotations

import http.server
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import unittest

from colab_adapter.monitor.ui import render_dashboard_html


_FIXTURES = {
    "/api/v1/health": {
        "status": "HEALTHY", "is_stale": False, "uptime_seconds": 600,
        "sample_count": 2, "collector_backoff_level": 0,
        "last_collected_at": "2026-10-26T00:00:00Z"
    },
    "/api/v1/summary": {
        "current_balance_cu": 22.5, "current_rate_cu_per_hour": 0,
        "active_assignments": 0, "managed_running_runs": 0,
        "total_runs": 0, "is_stale": False
    },
    "/api/v1/compute/samples": [],
    "/api/v1/sessions": [],
    "/api/v1/runs": [],
}
_JS = r"""
const { chromium } = require('playwright-core');
(async () => {
  const port = process.argv[1];
  const out = process.argv[2];
  const browser = await chromium.launch({ headless: true, args: ['--no-sandbox'] });
  const page = await browser.newPage({viewport:{width:1440,height:1000}, colorScheme:'light'});
  const errors = [];
  page.on('pageerror', e => errors.push(String(e)));
  await page.goto('http://127.0.0.1:' + port + '/', {waitUntil:'domcontentloaded'});
  await page.waitForFunction(() => document.querySelector('#val-cu').textContent === '22.50 CU');
  const theme = () => page.evaluate(() => ({
    theme: document.documentElement.getAttribute('data-theme'),
    background: getComputedStyle(document.querySelector('.container')).backgroundColor
  }));
  const light = await theme();
  if (light.theme !== 'light') throw new Error('Initial light theme unavailable');
  if ((await page.locator('#val-assignments').innerText()).trim() !== '0')
    throw new Error('Actual zero must render as 0');
  if (await page.locator('#compute-chart-container').count() !== 1)
    throw new Error('The compute chart must be unique');
  await page.screenshot({path:out+'/monitor-light-desktop.png', fullPage:true});
  await page.locator('#theme-toggle').click();
  const dark = await theme();
  if (dark.theme !== 'dark' || dark.background === light.background)
    throw new Error('Theme toggle failed to change shell colors');
  const stored = await page.evaluate(() => localStorage.getItem('colab-monitor-color-theme'));
  if (stored !== 'dark') throw new Error('Dark theme preference not persisted');
  await page.screenshot({path:out+'/monitor-dark-desktop.png', fullPage:true});
  await page.reload();
  await page.waitForFunction(() => document.documentElement.getAttribute('data-theme') === 'dark');
  await page.locator('#theme-toggle').click();
  await page.setViewportSize({width:390,height:844});
  await page.screenshot({path:out+'/monitor-light-mobile.png', fullPage:true});
  const widths = await page.evaluate(() => ({viewport:innerWidth, body:document.body.scrollWidth}));
  if (widths.body > widths.viewport + 2)
    throw new Error('Mobile page overflows viewport: '+JSON.stringify(widths));
  if (errors.length) throw new Error('Browser page errors: '+errors.join('; '));
  if (await page.locator('button').filter({hasText:/^stop|^delete/i}).count())
    throw new Error('Unexpected mutating monitor control');
  console.log('INDUSTRIAL_THEME_BROWSER_PASS light/dark/mobile, persisted toggle, synthetic metrics, no mutation');
  await browser.close();
})().catch(err => { console.error(err.stack || String(err)); process.exit(1); });
"""


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        import urllib.parse
        path = urllib.parse.urlsplit(self.path).path
        if path == "/":
            data = render_dashboard_html(csp_nonce="test-nonce").encode("utf-8")
            typ = "text/html; charset=utf-8"
        elif path in _FIXTURES:
            data = json.dumps(_FIXTURES[path]).encode("utf-8")
            typ = "application/json"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", typ)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@unittest.skipUnless(os.getenv("COLAB_REQUIRE_BROWSER_TESTS") == "1",
                     "Chromium visual check runs in dedicated browser CI")
class IndustrialThemeBrowserTests(unittest.TestCase):
    def test_light_dark_mobile_rendered_chromium(self):
        node = shutil.which("node")
        self.assertIsNotNone(node, "Node is required for browser CI")
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            screenshots = Path("artifacts").resolve()
            screenshots.mkdir(exist_ok=True)
            result = subprocess.run(
                [node, "-e", _JS, str(server.server_port), str(screenshots)],
                capture_output=True, text=True, timeout=95,
            )
            self.assertEqual(result.returncode, 0, result.stdout + "\n" + result.stderr)
            self.assertIn("INDUSTRIAL_THEME_BROWSER_PASS", result.stdout)
            for name in ("monitor-light-desktop.png", "monitor-dark-desktop.png",
                         "monitor-light-mobile.png"):
                self.assertGreater((screenshots / name).stat().st_size, 1000)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)


if __name__ == "__main__":
    unittest.main()
