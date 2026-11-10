"""Living observation plan validation. Offline-only; never allocates GPU or fabricates data."""
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "research/observations/qwen35-4b-base-eagle3-plan.v1.json"
NOTE = ROOT / "research/observations/2026-11-10-qwen35-4b-base-vs-eagle3-memlens-mmlongbench.md"


class LivingQwen4BObservationPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = json.loads(PLAN.read_text(encoding="utf-8"))
        cls.note = NOTE.read_text(encoding="utf-8")

    def test_plan_is_observation_location_but_not_an_observed_result(self):
        p = self.plan
        self.assertEqual(p["schema_version"], 1)
        self.assertEqual(p["plan_revision"], "0.1")
        self.assertEqual(p["status"], "PLANNED_NOT_EXECUTED")
        self.assertTrue(p["mutable_observation"])
        self.assertFalse(p["experimental_results_available"])
        self.assertFalse(p["training"])
        self.assertFalse(p["finetuning"])
        self.assertIn("PLANNED_NOT_EXECUTED", self.note)
        self.assertIn("NO GPU RUN", self.note)
        self.assertTrue(all(value is None for value in p["measured"].values()))

    def test_exactly_two_arms_and_one_posttrained_4b_target(self):
        p = self.plan
        self.assertEqual(p["model"]["repo_id"], "Qwen/Qwen3.5-4B")
        self.assertIsNone(p["model"]["revision"])
        self.assertTrue(p["model"]["post_trained"])
        self.assertEqual([(a["id"],a["mode"]) for a in p["arms"]],
                         [("A0","normal"),("A1","speculative")])
        self.assertIsNone(p["arms"][0]["draft"])
        self.assertEqual(p["arms"][1]["draft_repo_id"],
                         "yuyijiong/Qwen3.5-4B-Eagle3")
        self.assertIsNone(p["arms"][1]["draft_revision"])
        self.assertEqual(p["arms"][1]["multimodal_compatibility"],
                         "BLOCKED_UNQUALIFIED")
        self.assertEqual(p["engine"]["version"], "0.29.0")
        self.assertFalse(p["engine"]["prefix_caching"])

    def test_only_authentic_author_benchmark_data(self):
        self.assertEqual([(d["id"],d["publisher"],d["questions"])
                          for d in self.plan["datasets"]],
                         [("memlens32k","xiyuRenBill/MEMLENS",789),
                          ("mmlongdoc","yubo2333/MMLongBench-Doc",1091)])
        self.assertTrue(self.plan["quality"]["original_unmodified_inputs"])
        self.assertFalse(self.plan["quality"]["synthetic_data"])
        self.assertEqual(self.plan["quality"]["planned_arm_question_evaluations"],
                         (789+1091)*2)
        self.assertFalse(self.plan["quality"]["ignore_eos"])
        self.assertFalse(self.plan["quality"]["official_judges_qualified"])

    def test_exhaustive_ccu_grid_counts_match_formulas(self):
        p = self.plan["performance"]
        lengths = [128,512,1024,2048,4096,8192,16384]
        ccus = [1,2,4,8,16,32]
        self.assertEqual(p["input_token_targets"], lengths)
        self.assertEqual(p["ccu"], ccus)
        attempts = sum(max(16,4*c) for _ in lengths for c in ccus)
        warmups = sum(c for _ in lengths for c in ccus)
        pairs = len(self.plan["arms"])*len(self.plan["datasets"])
        self.assertEqual(p["cells_per_dataset_arm"], len(lengths)*len(ccus))
        self.assertEqual(p["theoretical_total_cells"], len(lengths)*len(ccus)*pairs)
        self.assertEqual(p["measured_requests_per_dataset_arm"], attempts)
        self.assertEqual(p["warmup_requests_per_dataset_arm"], warmups)
        self.assertEqual(p["theoretical_total_measured_requests"], attempts*pairs)
        self.assertEqual(p["theoretical_total_warmup_requests"], warmups*pairs)
        self.assertEqual((attempts,warmups),(1904,441))

    def test_quality_and_fixed_output_benchmark_are_separate(self):
        p = self.plan
        self.assertIs(p["quality"]["ignore_eos"], False)
        self.assertEqual(p["quality"]["max_output_tokens_initial"],256)
        self.assertEqual(p["performance"]["output_tokens_target"],256)
        self.assertEqual(p["performance"]["ignore_eos"],
                         "REQUIRE_RUNTIME_PROOF_CHAT_STREAM")
        self.assertEqual(p["performance"]["eligible_inputs"],
                         "ORIGINAL_AUTHENTIC_NATURALLY_MATCHING_TOKEN_BUCKETS")
        self.assertTrue(p["performance"]["no_truncation_or_padding"])
        self.assertEqual(p["performance"]["missing_bucket_status"],
                         "INSUFFICIENT_AUTHENTIC_INPUTS")
        self.assertTrue(p["performance"]["matched_trace_between_arms"])

    def test_metric_groups_and_readiness_guards_are_explicit(self):
        groups=self.plan["metrics"]
        self.assertEqual(set(groups),{"official_quality","hallucination","latency",
               "throughput","memory","eagle3","reliability","paired"})
        self.assertTrue(all(isinstance(values,list) and values for values in groups.values()))
        gate=self.plan["readiness"]
        self.assertEqual(gate["eagle3_multimodal"],"BLOCKED_UNQUALIFIED")
        self.assertEqual(gate["ccu32_warmup"],"RUNNER_CAP_10_MUST_FIX")
        self.assertEqual(gate["gpu_authorized"],False)
        self.assertEqual(gate["execution"],"NOT_RUN")
        for phrase in ("full 42-cell","INSUFFICIENT_AUTHENTIC_INPUTS",
                       "no numerical performance","EAGLE3", "false-support"):
            self.assertIn(phrase.lower(),self.note.lower())


if __name__=="__main__":
    unittest.main()
