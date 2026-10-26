"""Dashboard UI contract for M8 defects: null shown as 0, Drive status guessed as PENDING, and the
unbounded session table.

Pure JS helpers are delimited by `// BEGIN pure-helpers` / `// END pure-helpers` in ui.py. The
node-based tests evaluate that exact block (skipped when node is not installed). Structural tests
check the rest of the embedded script. The real-Chromium suite is in
test_browser_xss_and_cross_process_qa.py and is not run here.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import unittest

from colab_adapter.monitor.ui import DASHBOARD_HTML, get_dashboard_js

NODE = shutil.which("node")

_HELPER_RE = re.compile(r"// BEGIN pure-helpers\n(.*?)// END pure-helpers", re.DOTALL)

_NODE_HARNESS = r"""
const src = process.argv[1];
const cases = JSON.parse(process.argv[2]);
const store = {};
const els = {};
for (const id of cases.ids) { els[id] = { textContent: '--' }; }
globalThis.document = { getElementById: (id) => els[id] || null };
const h = new Function(src + '\nreturn {kpiAssignmentsText, kpiRateText, kpiBalanceText, kpiCountText, planSessionRows, applySummaryKpis, SESSION_ROW_CAP};')();
const sessions = [];
for (let i = 0; i < 120; i++) sessions.push({ session_id: 'live-' + i, state: 'READY' });
for (let i = 0; i < 202; i++) sessions.push({ session_id: 'dead-' + i, state: 'TERMINATED' });
h.applySummaryKpis({
  current_balance_cu: null, current_rate_cu_per_hour: null, active_assignments: null,
  managed_running_runs: 1, total_runs: null, is_stale: true,
});
const out = {
  assignNull: h.kpiAssignmentsText(null),
  assignUndef: h.kpiAssignmentsText(undefined),
  assignGarbage: h.kpiAssignmentsText('x'),
  assignZero: h.kpiAssignmentsText(0),
  assignTwo: h.kpiAssignmentsText(2),
  rateNull: h.kpiRateText(null),
  rateValue: h.kpiRateText(1.9),
  balanceNull: h.kpiBalanceText(null),
  balanceValue: h.kpiBalanceText(84.5),
  cap: h.SESSION_ROW_CAP,
  collapsed: h.planSessionRows(sessions, false, h.SESSION_ROW_CAP),
  expanded: h.planSessionRows(sessions, true, h.SESSION_ROW_CAP),
  mixedOrder: h.planSessionRows([{ state: 'TERMINATED', session_id: 't' }, { state: 'READY', session_id: 'r' }], true, 50).rows.map(s => s.session_id),
  onlyTerminated: h.planSessionRows([{ state: 'TERMINATED', session_id: 't' }], false, 50),
  dom: Object.fromEntries(Object.entries(els).map(([k, v]) => [k, v.textContent])),
};
console.log(JSON.stringify(out));
"""


def _helper_src() -> str:
    match = _HELPER_RE.search(get_dashboard_js())
    if match is None:
        raise AssertionError("pure-helpers block missing from embedded dashboard script")
    return match.group(1)


@unittest.skipIf(NODE is None, "node is not installed; pure helper evaluation skipped")
class PureHelperBehaviourTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ids = ["val-cu", "meta-cu", "val-rate", "val-assignments", "val-runs"]
        proc = subprocess.run(
            [NODE, "-e", _NODE_HARNESS, _helper_src(), json.dumps({"ids": ids})],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        )
        if proc.returncode != 0:
            raise AssertionError(f"node harness failed: {proc.stderr}")
        cls.out = json.loads(proc.stdout)

    def test_null_active_assignments_renders_unavailable_not_zero(self):
        self.assertEqual(self.out["assignNull"], "Unavailable")
        self.assertEqual(self.out["assignUndef"], "Unavailable")
        self.assertEqual(self.out["assignGarbage"], "Unavailable")

    def test_real_zero_and_counts_still_render_as_numbers(self):
        self.assertEqual(self.out["assignZero"], "0")
        self.assertEqual(self.out["assignTwo"], "2")

    def test_null_rate_and_balance_render_unavailable(self):
        self.assertEqual(self.out["rateNull"], "Unavailable")
        self.assertEqual(self.out["rateValue"], "1.90 CU/hr")
        self.assertEqual(self.out["balanceNull"], "Unavailable")
        self.assertEqual(self.out["balanceValue"], "84.50 CU")

    def test_summary_kpis_set_textcontent_unavailable_for_nulls(self):
        dom = self.out["dom"]
        self.assertEqual(dom["val-cu"], "Unavailable")
        self.assertEqual(dom["val-rate"], "Unavailable")
        self.assertEqual(dom["val-assignments"], "Unavailable")
        self.assertEqual(dom["val-runs"], "1 / Unavailable")

    def test_session_rows_cap_and_collapse_terminated(self):
        self.assertEqual(self.out["cap"], 50)
        collapsed = self.out["collapsed"]
        self.assertEqual(len(collapsed["rows"]), 50)
        self.assertEqual(collapsed["moreNotShown"], 70)  # 120 live, only 50 shown
        self.assertEqual(collapsed["terminatedCount"], 202)
        self.assertTrue(all(r["state"] == "READY" for r in collapsed["rows"]))

    def test_expanded_still_lists_live_first_and_respects_cap(self):
        expanded = self.out["expanded"]
        self.assertEqual(len(expanded["rows"]), 50)
        self.assertEqual(expanded["moreNotShown"], 322 - 50)
        self.assertTrue(all(r["state"] == "READY" for r in expanded["rows"][:50]))

    def test_live_sessions_precede_terminated_when_expanded(self):
        self.assertEqual(self.out["mixedOrder"], ["r", "t"])

    def test_only_terminated_collapsed_shows_no_rows(self):
        only = self.out["onlyTerminated"]
        self.assertEqual(only["rows"], [])
        self.assertEqual(only["moreNotShown"], 0)
        self.assertEqual(only["terminatedCount"], 1)


class DashboardStructureTests(unittest.TestCase):
    def setUp(self):
        self.js = get_dashboard_js()

    def test_no_null_to_zero_fallbacks_on_kpis(self):
        self.assertIn("valAssign.textContent = kpiAssignmentsText(", self.js)
        self.assertNotIn("active_assignments != null ? String", self.js)
        self.assertNotIn("'0.00 CU/hr'", self.js)
        self.assertNotIn("valRate.textContent = (s.current_rate_cu_per_hour", self.js)

    def test_compute_chart_does_not_plot_unknown_rate_as_zero(self):
        self.assertNotIn("(s.rate_cu_per_hour != null ? Number(s.rate_cu_per_hour) : 0)", self.js)
        self.assertNotIn("valid[i].rate_cu_per_hour != null ? Number(valid[i].rate_cu_per_hour) : 0", self.js)

    def test_drive_status_defaults_to_not_recorded_not_pending(self):
        self.assertIn("Drive Status: NOT RECORDED", DASHBOARD_HTML)
        self.assertNotIn("Drive Status: PENDING", DASHBOARD_HTML)
        self.assertIn("String(r.drive_publication_status || 'NOT RECORDED')", self.js)
        self.assertNotIn("String(r.drive_publication_status || 'PENDING')", self.js)
        self.assertNotIn("'PENDING' : (artifactsList && artifactsList.length ? 'PENDING' : 'DISABLED')", self.js)

    def test_session_toggle_built_with_dom_api_and_listener(self):
        self.assertIn("document.createElement('button')", self.js)
        self.assertIn("btn.addEventListener('click'", self.js)
        self.assertIn("`Show ${", self.js)
        self.assertIn("terminated`", self.js)
        self.assertIn("`Hide ${", self.js)

    def test_session_overflow_line_is_explicit(self):
        self.assertIn("more not shown", self.js)
        self.assertIn("planSessionRows(", self.js)

    def test_no_inline_handlers_or_html_sinks_in_script(self):
        self.assertNotIn("innerHTML", self.js)
        self.assertNotIn("outerHTML", self.js)
        self.assertNotIn("insertAdjacentHTML", self.js)
        self.assertIsNone(re.search(r"setAttribute\(\s*['\"]on", self.js))
        self.assertIsNone(re.search(r"\.on(click|change|submit|input)\s*=", self.js))

    def test_pure_helper_block_is_delimited_once(self):
        self.assertEqual(self.js.count("// BEGIN pure-helpers"), 1)
        self.assertEqual(self.js.count("// END pure-helpers"), 1)


if __name__ == "__main__":
    unittest.main()
