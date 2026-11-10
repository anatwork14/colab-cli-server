"""Technical lead regressions for source/experiment/port ownership guards."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from research_ops.benchmarks import runner, serve


class LiveServerSafetyTests(unittest.TestCase):
    def test_status_must_match_frozen_port_before_contacting_vllm(self):
        with TemporaryDirectory() as directory:
            root=Path(directory)/"serve-port-test"
            root.mkdir()
            frozen={
                "pid":8000123,"start_ticks":100,"serving_model":"Qwen3.5",
                "argv_without_tokens":["vllm","serve","/content/model",
                                       "--host","127.0.0.1","--port","8000"],
            }
            (root/"server_process.json").write_text(json.dumps(frozen))
            args=SimpleNamespace(output_root=directory,run_id=root.name,port=9000)
            with patch.object(serve,"_owned",return_value=True), patch.object(
                    serve,"server_check",side_effect=AssertionError("must not query unrelated port")):
                status=serve.status(args)
            self.assertEqual(status["status"],"FROZEN_PORT_MISMATCH")
            self.assertEqual(status["frozen_port"],8000)

    def test_status_refuses_non_local_frozen_host(self):
        with TemporaryDirectory() as directory:
            root=Path(directory)/"serve-port-test"
            root.mkdir()
            bad={
                "pid":8000123,"start_ticks":100,"serving_model":"Qwen3.5",
                "argv_without_tokens":["vllm","serve","/content/model",
                                       "--host","0.0.0.0","--port","8000"],
            }
            (root/"server_process.json").write_text(json.dumps(bad))
            args=SimpleNamespace(output_root=directory,run_id=root.name,port=8000)
            with patch.object(serve,"_owned",return_value=True):
                self.assertEqual(serve.status(args)["status"],"INVALID_FROZEN_SERVER_RECEIPT")

    def test_stop_refuses_wrong_process_group(self):
        with TemporaryDirectory() as directory:
            root=Path(directory)/"serve-stop-test"
            root.mkdir()
            (root/"server_process.json").write_text(json.dumps({
                "pid":80801,"start_ticks":41,
            }))
            args=SimpleNamespace(output_root=directory,run_id=root.name,execute=True)
            with patch.object(serve,"_owned",return_value=True), patch.object(
                serve.os,"getpgid",return_value=80802), patch.object(
                serve.os,"getpgrp",return_value=444), patch.object(
                serve.os,"killpg",side_effect=AssertionError("should never kill")):
                with self.assertRaises(RuntimeError):
                    serve.stop(args)

    def test_experiment_requires_authentic_original_checkpoint_revision(self):
        pin="e"*40
        self.assertEqual(serve._assert_model_origin({
            "role":"model","source":{"type":"huggingface",
             "repo_id":"Qwen/Qwen3.5-4B","revision":pin}},
            "Qwen/Qwen3.5-4B")["revision"],pin)
        cache={"role":"model","tree_sha256":"d"*64,
               "source":{"type":"archive",
               "path":"/content/weights.tar","sha256":"f"*64,
               "origin_repo_id":"Qwen/Qwen3.5-4B","origin_revision":pin,
               "origin_tree_sha256":"d"*64}}
        self.assertEqual(serve._assert_model_origin(cache,"Qwen/Qwen3.5-4B")["revision"],pin)
        for src in (
            {"type":"huggingface","repo_id":"wrong/weights","revision":pin},
            {"type":"directory","path":"/content/cache","tree_sha256":"a"*64},
            {"type":"archive","origin_repo_id":"Qwen/Qwen3.5-4B","origin_revision":"main"},
            {"type":"archive","origin_repo_id":"Qwen/Qwen3.5-4B",
             "origin_revision":pin,"origin_tree_sha256":"1"*64},
        ):
            with self.subTest(src=src):
                with self.assertRaises(ValueError):
                    serve._assert_model_origin({"role":"model","source":src},
                                               "Qwen/Qwen3.5-4B")

    def test_normal_experiment_cannot_inherit_author_da_activation(self):
        args=SimpleNamespace(execute=True,run_id="audit-01",port=8000,
            gpu_memory_utilization=0.85,max_model_len=32768,max_images=1,
            experiment_mode="normal")
        with patch.dict(serve.os.environ,
                        {"H2O_VLLM_MODE":"disabled","DATTN_MODEL":"Qwen/Qwen3.5-4B"}):
            with patch.object(serve,"verify_asset",
                              side_effect=AssertionError("should fail before assets")):
                with self.assertRaisesRegex(RuntimeError,"clean isolated engine"):
                    serve.start(args)

    def test_h2o_passthrough_cannot_contaminate_benchmark_process(self):
        args=SimpleNamespace(execute=True,run_id="audit-01",port=8000,
            gpu_memory_utilization=0.85,max_model_len=32768,max_images=1)
        with patch.dict(serve.os.environ,{"H2O_VLLM_MODE":"passthrough"}), patch.object(
                    serve,"verify_asset",side_effect=AssertionError("too late")):
            with self.assertRaises(RuntimeError):
                serve.start(args)


class ExperimentReceiptTests(unittest.TestCase):
    def test_normal_and_mtp_have_explicit_matching_arm(self):
        normal={"experiment_arm":{"id":"qwen35_4b-normal",
            "offline_plan_only":False,"declarative_attention_roles":[],
            "speculative_method":None}}
        self.assertEqual(runner.check_research_arm(normal,"qwen35_4b-normal")["id"],
                         "qwen35_4b-normal")
        with self.assertRaises(RuntimeError):
            runner.check_research_arm(normal,"qwen35_4b-speculative-mtp")
        mtp={"experiment_arm":{"id":"qwen35_4b-speculative-mtp",
            "offline_plan_only":False,"speculative_method":"mtp",
            "declarative_attention_roles":[]}}
        self.assertIsNotNone(runner.check_research_arm(mtp,"qwen35_4b-speculative-mtp"))

    def test_draft_requires_attested_bytes(self):
        eagle={"experiment_arm":{"id":"qwen35_4b-speculative-eagle3",
                 "offline_plan_only":False,"speculative_method":"eagle3",
                 "declarative_attention_roles":[]},
               "draft_asset_tree_sha256":None}
        with self.assertRaises(RuntimeError):
            runner.check_research_arm(eagle,"qwen35_4b-speculative-eagle3")
        eagle["draft_asset_tree_sha256"]="0"*64
        self.assertEqual(runner.check_research_arm(
            eagle,"qwen35_4b-speculative-eagle3")["speculative_method"],"eagle3")

    def test_da_without_qualified_gpu_parity_remains_blocked_even_with_forged_flag(self):
        record={"experiment_arm":{"id":"qwen35_4b-da_normal",
                "offline_plan_only":False,"declarative_attention_roles":["target"],
                "speculative_method":None},"da_hook_runtime_verified":True}
        with self.assertRaisesRegex(RuntimeError,"not yet GPU qualified"):
            runner.check_research_arm(record,"qwen35_4b-da_normal")
        record["da_hook_runtime_verified"]=False
        with self.assertRaises(RuntimeError):
            runner.check_research_arm(record,"qwen35_4b-da_normal")

    def test_legacy_server_cannot_be_relabelled_as_research_mode(self):
        self.assertIsNone(runner.check_research_arm({"experiment_arm":None},None))
        with self.assertRaises(RuntimeError):
            runner.check_research_arm({"experiment_arm":None},"qwen35_4b-normal")
        with self.assertRaises(RuntimeError):
            runner.check_research_arm(
                {"experiment_arm":{"id":"qwen35_4b-normal","offline_plan_only":True}},
                "qwen35_4b-normal")


if __name__ == "__main__":
    unittest.main()
