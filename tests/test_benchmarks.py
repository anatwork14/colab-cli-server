"""Offline/negative unit checks for real-data adapters. No released dataset is generated."""
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from research_ops.benchmarks import datasets, messages, runner, agent, serve


class BenchmarkDatasetTests(unittest.TestCase):
    def test_rejects_path_traversal_and_urls(self):
        for value in ("../a.pdf", "/private.pdf", "doc/../a.pdf", "doc\\x.pdf", "",
                      "https://example.invalid/image.jpg"):
            with self.subTest(value=value):
                with self.assertRaises(datasets.DatasetIntegrityError):
                    datasets.safe_relative(value)

    def test_evidence_pages_are_parsed_safely_without_eval(self):
        self.assertEqual(datasets.evidence_pages({"evidence_pages":"[3, 5]"}),[3,5])
        for invalid in ("__import__('os').system('false')", "{3:5}", "[-1]", "not a list"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(datasets.DatasetIntegrityError):
                    datasets.evidence_pages({"evidence_pages":invalid})

    def test_real_dataset_file_absent_blocks(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"missing.json"
            with self.assertRaises(FileNotFoundError):
                datasets.load_records("memlens",path)

    def test_unpinned_denominator_fails_closed(self):
        with patch("research_ops.benchmarks.datasets.load_records",return_value=[]):
            with self.assertRaises(datasets.DatasetIntegrityError):
                datasets.audit("memlens","/tmp/not-real","/tmp",expected_count=None)


class PromptIntegrityTests(unittest.TestCase):
    def test_session_roles_and_image_placeholder_order(self):
        row={
            "question":"What was recorded?","question_date":"2026-01-01",
            "haystack_sessions":[{"date":"2025-12-31","session":[
                {"role":"user","content":"Before <image> after",
                 "images":[{"file":"image.jpg"}]},
                {"role":"assistant","content":"Something recorded"}]}],
        }
        with patch("research_ops.benchmarks.messages.locate_file",return_value=Path("/tmp/img.jpg")):
            with patch("research_ops.benchmarks.messages._image_data",
                       return_value=({"type":"image_url","image_url":{"url":"data:image/jpeg;base64,AA=="}},1)):
                prompt,meta=messages.memlens_messages(row,"/tmp",max_images=1)
        self.assertEqual([p["role"] for p in prompt],["system","user","assistant","user"])
        self.assertTrue(prompt[1]["content"][0]["text"].startswith("=== Session 1"))
        self.assertEqual([b["type"] for b in prompt[1]["content"]],
                         ["text","image_url","text"])
        self.assertEqual(meta["images_sent"],1)
        self.assertFalse(meta["prompt_truncation"])

    def test_unmatched_image_placeholder_blocks(self):
        with self.assertRaises(datasets.DatasetIntegrityError):
            messages._images_from_turn("image <image>", [], "/tmp",100)

    def test_pdf_full_page_limit_cannot_truncate(self):
        class FakeDoc:
            def __enter__(self):return self
            def __exit__(self,*a):return False
            def __len__(self):return 12
            def __getitem__(self,i):raise AssertionError("Should not render truncated pages")
        fake=SimpleNamespace(open=lambda path:FakeDoc())
        row={"doc_id":"paper.pdf","question":"Read the tables","answer":"Not answerable"}
        with patch.dict("sys.modules",{"fitz":fake}):
            with patch("research_ops.benchmarks.messages.locate_file",
                       return_value=Path("/tmp/paper.pdf")):
                with self.assertRaises(datasets.DatasetIntegrityError):
                    messages.doc_messages(row,"/tmp",max_pages=10)


class StreamingAndScoringTests(unittest.TestCase):
    def test_restricts_serving_to_loopback(self):
        for url in ("https://evil.example/v1","http://8.8.8.8:8000",
                    "file:///etc/passwd","http://127.0.0.1:8000/secret"):
            with self.subTest(url=url):
                with self.assertRaises(ValueError):
                    runner._endpoint(url)
        self.assertEqual(runner._endpoint("http://127.0.0.1:8000/v1"),
                         "http://127.0.0.1:8000")

    def test_sse_stream_and_token_usage_tpot_proxy(self):
        chunks=[
            {"choices":[{"delta":{"role":"assistant"}}]},
            {"choices":[{"delta":{"content":"correct "}}]},
            {"choices":[{"delta":{"content":"answer"},"finish_reason":"stop"}]},
            {"choices":[],"usage":{"prompt_tokens":42,"completion_tokens":3}},
        ]
        data=("\n".join("data: "+json.dumps(x)+"\n" for x in chunks)+
              "\ndata: [DONE]\n").encode()
        with patch("research_ops.benchmarks.runner.urlopen",return_value=io.BytesIO(data)):
            r=runner.streaming_completion("http://localhost:8000","my-model",
                      [{"role":"user","content":"real question"}],max_tokens=128,timeout_s=3)
        self.assertEqual(r["prediction"],"correct answer")
        self.assertEqual(r["prompt_tokens"],42)
        self.assertEqual(r["completion_tokens"],3)
        self.assertIsInstance(r["tpot_ms_estimate"],float)
        self.assertTrue(str(r["ITL_ms"]).startswith("NOT_MEASURED"))

    def test_missing_predictions_are_counted_not_removed(self):
        rows=[{"question_type":"information_extraction","answer":"A"},
              {"question_type":"answer_refusal","answer":"Insufficient information"}]
        receipts=[{"status":"SUCCESS","prediction":"A","ttft_ms":10.0,
                   "e2e_ms":20.0,"tpot_ms_estimate":3.0},None]
        x=runner.summarize("memlens",rows,receipts,1.0,subset=True)
        self.assertEqual(x["denominator"],2)
        self.assertEqual(x["failures"],1)
        self.assertEqual(x["proxy_exact_match_including_failures"],0.5)
        self.assertIsNone(x["official_accuracy"])
        self.assertIsNone(x["official_hallucination_rate"])

    def test_no_gpu_or_cloud_in_agent_template_plan(self):
        campaign=Path("research/benchmarks/campaign.example.json")
        result=agent.campaign_plan(campaign)
        self.assertEqual(result["status"],"BLOCKED_BEFORE_GPU")
        self.assertTrue(result["no_cloud_action"])
        self.assertFalse(result["run_ready"])


class ServerLaunchGuards(unittest.TestCase):
    def test_cannot_launch_without_explicit_execute(self):
        args=SimpleNamespace(execute=False,run_id="notapproved")
        with self.assertRaises(ValueError):
            serve.start(args)

    def test_untrusted_pid_never_stopped(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/"run-001"
            root.mkdir()
            (root/"server_process.json").write_text(
                json.dumps({"run_id":"run-001","pid":99999999,"start_ticks":1}))
            args=SimpleNamespace(output_root=folder,run_id="run-001",execute=True)
            answer=serve.stop(args)
            self.assertFalse(answer["stopped"])


if __name__=="__main__":
    unittest.main()
