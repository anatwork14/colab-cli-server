"""Offline acceptance tests for agent job status and no-cherry-picking collector."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from research_ops.assets import hash_file
from research_ops.throughput import collect, job, runner
from research_ops.throughput.plan import DEFAULT_CONFIG,read_config,cells


def complete(cell,seed):
    observations=[{"status":"SUCCESS","ttft_ms":20.0,"e2e_ms":300.0,
        "tpot_ms_estimate":1.1,"prompt_tokens":cell["input_tokens"],
        "completion_tokens":256} for _ in range(cell["requests"])]
    warmups=[{"status":"SUCCESS"} for _ in range(cell["warmups"])]
    record=runner.summarize(cell,observations,12.0,warmups,seed,
        ["A100"],{"model_commit":"a"*40,"model_tree_sha256":"b"*64},
        hash_file(DEFAULT_CONFIG))
    record["attempt_id"]="attempt-{}".format(seed)
    return record


class OfflineSweepCollectionTests(unittest.TestCase):
    def test_empty_results_never_report_complete(self):
        with TemporaryDirectory() as folder:
            rows,report=collect.collect(folder,DEFAULT_CONFIG)
        self.assertEqual(rows,[])
        self.assertEqual(report["base_cells_expected"],84)
        self.assertEqual(len(report["base_missing"]),84)
        self.assertFalse(report["valid_base_campaign"])
        self.assertFalse(report["all_252_cells_ready"])

    def test_one_accepted_real_cell_still_not_full_campaign(self):
        cell=cells(model="qwen35_08b",arm="base",context=128,ccu=1)[0]
        with TemporaryDirectory() as folder:
            path=Path(folder)/"attempt-one"
            path.mkdir()
            (path/"summary.json").write_text(json.dumps(complete(cell,11)))
            rows,report=collect.collect(folder,DEFAULT_CONFIG)
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]["id"],cell["id"])
        self.assertEqual(report["base_cells_valid"],1)
        self.assertFalse(report["valid_base_campaign"])

    def test_duplicate_successes_must_be_blocked_not_cherry_picked(self):
        cell=cells(model="qwen35_08b",arm="base",context=128,ccu=1)[0]
        with TemporaryDirectory() as folder:
            for name,seed in (("attempt-one",11),("attempt-two",12)):
                path=Path(folder)/name
                path.mkdir()
                (path/"summary.json").write_text(json.dumps(complete(cell,seed)))
            rows,report=collect.collect(folder,DEFAULT_CONFIG)
        self.assertEqual(rows,[])
        self.assertTrue(any("Multiple complete attempts" in x["reason"]
                        for x in report["blocked_attempts"]))

    def test_h2o_and_da_can_never_claim_full_gpu_results(self):
        with TemporaryDirectory() as folder:
            for arm in ("h2o","da"):
                cell=cells(model="qwen35_4b",arm=arm,context=512,ccu=4)[0]
                path=Path(folder)/arm
                path.mkdir()
                (path/"summary.json").write_text(json.dumps(complete(cell,13)))
            rows,report=collect.collect(folder,DEFAULT_CONFIG)
        self.assertEqual(rows,[])
        self.assertEqual(len(report["blocked_attempts"]),2)
        self.assertEqual(report["h2o_gpu_cells_qualified"],0)
        self.assertEqual(report["da_gpu_cells_qualified"],0)

    def test_short_or_missing_requests_never_qualify(self):
        cell=cells(model="qwen35_08b",arm="base",context=128,ccu=1)[0]
        r=complete(cell,2)
        r["success_count"]-=1
        with TemporaryDirectory() as folder:
            path=Path(folder)/"bad"
            path.mkdir()
            (path/"summary.json").write_text(json.dumps(r))
            rows,report=collect.collect(folder,DEFAULT_CONFIG)
        self.assertEqual(rows,[])
        self.assertEqual(len(report["blocked_attempts"]),1)


class AgentSafetyTests(unittest.TestCase):
    def test_no_gpu_job_can_start_without_execute(self):
        with self.assertRaisesRegex(RuntimeError,"Explicit --execute"):
            job.start(SimpleNamespace(execute=False))

    def test_orphaned_remote_job_cannot_claim_healthy_status(self):
        with TemporaryDirectory() as folder:
            job_dir=Path(folder)/"attempt-001"
            job_dir.mkdir()
            meta={"run_id":"attempt-001","pid":9999999,
                  "start_ticks":123,"results":str(Path(folder)/"other"),
                  "log":str(job_dir/"runner.log")}
            (job_dir/"job.json").write_text(json.dumps(meta))
            x=job.status(SimpleNamespace(run_id="attempt-001",job_root=folder))
        self.assertEqual(x["status"],"STOPPED_WITHOUT_SUMMARY")
        self.assertFalse(x["pid_owned"])

    def test_approved_colab_sync_includes_protocol_and_model_catalog(self):
        f=Path("research/benchmarks/colab-project.json")
        cfg=json.loads(f.read_text(encoding="utf8"))
        include=set(cfg["workspace"]["include"])
        self.assertIn("research_ops/**",include)
        self.assertIn("research/vllm_engine/checkpoint_catalog.json",include)
        self.assertIn("research/benchmarks/qwen35_inference_sweep.json",include)
        self.assertFalse(any("weights" in x or "datasets" in x for x in include))


if __name__=="__main__":
    unittest.main()
