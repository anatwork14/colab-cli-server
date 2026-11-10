"""Offline throughput matrix, exact chat length and GPU safety controls."""
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from research_ops.throughput.plan import (
    read_config,cells,summary,MODELS,ARMS,CCU,LENGTHS)
from research_ops.throughput import workload,runner
from research_ops.benchmarks import serve


class FakeTokenizer:
    vocab_size=200
    all_special_ids=[0,1]
    def decode(self,ids,skip_special_tokens=True,clean_up_tokenization_spaces=False):
        return " ".join("word"+str(int(x)%30) for x in ids)
    def apply_chat_template(self,messages,tokenize=True,
                            add_generation_prompt=True,enable_thinking=True):
        if enable_thinking:
            raise AssertionError("Thinking not disabled")
        text=messages[0]["content"]
        return [10,11,12,13]+list(range(len(text.split())))


class MatrixTests(unittest.TestCase):
    def test_required_252_cells(self):
        cfg=read_config()
        result=cells(cfg)
        self.assertEqual(len(result),252)
        self.assertEqual(summary(result)["cells_by_arm"],
                         {"base":84,"h2o":84,"da":84})
        self.assertEqual(summary(result)["measured_requests"],11424)
        self.assertEqual(summary(result)["warmup_requests"],2646)
        self.assertEqual(summary(result)["all_requests"],14070)
        self.assertEqual({x["ccu"] for x in result},set(CCU))
        self.assertEqual({x["input_tokens"] for x in result},set(LENGTHS))

    def test_single_run_counts_and_blocked_arms(self):
        c=cells(model="qwen35_08b",arm="base",context=16384,ccu=32)
        self.assertEqual(len(c),1)
        self.assertEqual(c[0]["requests"],128)
        self.assertEqual(c[0]["warmups"],32)
        for i in (1,2,4):
            self.assertEqual(cells(model="qwen35_4b",arm="base",
                                   context=128,ccu=i)[0]["requests"],16)
        self.assertIn("BLOCKED",cells(model="qwen35_4b",arm="h2o")[0]["status"])
        self.assertIn("BLOCKED",cells(model="qwen35_4b",arm="da")[0]["status"])

    def test_protocol_rejects_cache_and_unrealistic_context(self):
        with TemporaryDirectory() as folder:
            config=read_config()
            for variant in ("cache","max_seq","model_len","output"):
                obj=json.loads(json.dumps(config))
                if variant=="cache":obj["server"]["enable_prefix_caching"]=True
                if variant=="max_seq":obj["server"]["max_num_seqs"]=16
                if variant=="model_len":obj["server"]["max_model_len"]=16384
                if variant=="output":obj["output_tokens"]=128
                file=Path(folder)/(variant+".json")
                file.write_text(json.dumps(obj))
                with self.subTest(variant=variant):
                    with self.assertRaises(ValueError):
                        read_config(file)

    def test_no_unknown_cell_parameters(self):
        for kw in ({"model":"wrong"},{"arm":"fake"},
                   {"context":1025},{"ccu":64}):
            with self.subTest(kw=kw):
                with self.assertRaises(ValueError):
                    cells(**kw)


class PromptTests(unittest.TestCase):
    def test_random_ids_render_to_exact_chat_length(self):
        tok=FakeTokenizer()
        for size in (128,512,1024):
            p=workload.build_prompt(tok,size,12345+size)
            self.assertEqual(p["prompt_tokens"],size)
            self.assertEqual(len(workload.render_chat(tok,p["messages"][0]["content"])),size)
            self.assertEqual(p["origin"],"random_token_ids_decoded_then_chat_rendered")
            self.assertNotIn("<|im_start|>",p["messages"][0]["content"])

    def test_every_warmup_and_measurement_has_unique_seed_and_content(self):
        data=runner.prepare_prompts(FakeTokenizer(),128,18,123456)
        self.assertEqual(len({p["prompt_sha256"] for p in data}),18)
        self.assertEqual(len({p["seed"] for p in data}),18)
        self.assertTrue(all(p["prompt_tokens"]==128 for p in data))

    def test_random_prompt_seed_reproduces_exact_hash(self):
        a=workload.build_prompt(FakeTokenizer(),128,33)
        b=workload.build_prompt(FakeTokenizer(),128,33)
        self.assertEqual(a["prompt_sha256"],b["prompt_sha256"])
        self.assertNotEqual(workload.child_seed(33,0),
                            workload.child_seed(33,1))

    def test_stream_payload_is_non_thinking_fixed_length(self):
        prompt=workload.build_prompt(FakeTokenizer(),128,33)
        b=workload.payload("Qwen/Qwen3.5-0.8B",prompt,999)
        self.assertTrue(b["stream"])
        self.assertTrue(b["ignore_eos"])
        self.assertEqual(b["max_completion_tokens"],256)
        self.assertEqual(b["min_tokens"],256)
        self.assertEqual(b["temperature"],0)
        self.assertFalse(b["chat_template_kwargs"]["enable_thinking"])
        self.assertTrue(b["return_token_ids"])


class FakeStream(io.BytesIO):
    status=200


class StreamMeasurementTests(unittest.TestCase):
    def _stream(self,usage_prompt=128,usage_output=256):
        role={"choices":[{"delta":{"role":"assistant"}}]}
        first={"choices":[{"delta":{"content":"hello"},"token_ids":[11]}]}
        rest={"choices":[{"delta":{"content":" world"},
                          "token_ids":list(range(255)),
                          "finish_reason":"length"}]}
        usage={"choices":[],"usage":{"prompt_tokens":usage_prompt,
                                     "completion_tokens":usage_output}}
        events=[role,first,rest,usage]
        return FakeStream(("\n".join("data: "+json.dumps(x)+"\n"
                     for x in events)+"\ndata: [DONE]\n").encode())

    def test_role_only_event_does_not_count_as_first_token(self):
        prompt=workload.build_prompt(FakeTokenizer(),128,33)
        with patch.object(workload,"urlopen",return_value=self._stream()):
            times=iter(i/10.0 for i in range(100))
            with patch.object(workload.time,"perf_counter",side_effect=lambda:next(times)):
                x=workload.request_stream("http://127.0.0.1:8000/v1/chat/completions",
                                          "Qwen/Qwen3.5-0.8B",prompt,123)
        self.assertEqual(x["status"],"SUCCESS")
        self.assertGreaterEqual(x["ttft_ms"],300.0)
        self.assertEqual(x["completion_tokens"],256)
        self.assertEqual(x["prompt_tokens"],128)
        self.assertTrue(x["tpot_note"].startswith("client_"))

    def test_short_output_or_wrong_effective_input_fails(self):
        prompt=workload.build_prompt(FakeTokenizer(),128,33)
        with patch.object(workload,"urlopen",return_value=self._stream(128,240)):
            with self.assertRaisesRegex(RuntimeError,"Truncated"):
                workload.request_stream("http://127.0.0.1:8000/v1/chat/completions",
                                        "Qwen/Qwen3.5-0.8B",prompt,123)
        with patch.object(workload,"urlopen",return_value=self._stream(127,256)):
            with self.assertRaisesRegex(RuntimeError,"disagrees"):
                workload.request_stream("http://127.0.0.1:8000/v1/chat/completions",
                                        "Qwen/Qwen3.5-0.8B",prompt,123)

    def test_failed_runs_never_claim_results(self):
        cell=cells(model="qwen35_08b",arm="base",context=128,ccu=1)[0]
        good={"status":"SUCCESS","ttft_ms":100.0,"e2e_ms":900.0,
              "tpot_ms_estimate":2.0,"prompt_tokens":128,"completion_tokens":256}
        failed={"status":"FAILED","error_type":"RuntimeError"}
        result=runner.summarize(cell,[good,failed],2.0,[good],
                                123,["A100"],{"version":"0.29.0"},"z"*64)
        self.assertEqual(result["status"],"INCOMPLETE_NOT_COMPARABLE")
        self.assertEqual(result["failure_count"],1)
        self.assertIsNone(result["official_memlens_accuracy"])
        self.assertIsNone(result["physical_kv_bytes_saved"])


class ModeQualificationTests(unittest.TestCase):
    def test_h2o_and_da_block_before_any_gpu_file_access(self):
        for arm in ("h2o","da"):
            with self.subTest(arm=arm):
                with self.assertRaisesRegex(RuntimeError,"BLOCKED"):
                    runner.attest_server("/no-such-server", "qwen35_08b",
                                         arm,read_config(),
                                         "http://127.0.0.1:8000")

    def test_server_attestation_rejects_prefix_cache_and_wrong_ccu_cap(self):
        cfg=read_config()
        arm={"id":"qwen35_08b-normal","offline_plan_only":False}
        args=["vllm","serve","/tmp/model","--host","127.0.0.1",
              "--port","8000","--max-num-seqs","32","--max-model-len","18432",
              "--default-chat-template-kwargs", '{"enable_thinking":false}']
        frozen={"experiment_arm":arm,"serving_model":"Qwen/Qwen3.5-0.8B",
                "target_origin":{"repo_id":"Qwen/Qwen3.5-0.8B","revision":"a"*40},
                "engine":{"installed_version":"0.29.0"},
                "argv_without_tokens":args,"pid":999999,"start_ticks":1,
                "model_asset_sha256":"b"*64}
        with TemporaryDirectory() as folder:
            fp=Path(folder)/"server.json"
            with patch.object(runner,"_owned",return_value=True):
                fp.write_text(json.dumps(frozen))
                with self.assertRaisesRegex(RuntimeError,"Prefix caching"):
                    runner.attest_server(fp,"qwen35_08b","base",cfg,
                                         "http://127.0.0.1:8000")
                frozen["argv_without_tokens"]+=["--no-enable-prefix-caching"]
                frozen["argv_without_tokens"][frozen["argv_without_tokens"].index("--max-num-seqs")+1]="16"
                fp.write_text(json.dumps(frozen))
                with self.assertRaisesRegex(RuntimeError,"32 scheduled"):
                    runner.attest_server(fp,"qwen35_08b","base",cfg,
                                         "http://127.0.0.1:8000")

    def test_server_microbenchmark_launch_args_frozen(self):
        with TemporaryDirectory() as folder:
            args=SimpleNamespace(execute=True,run_id="base-sweep-001",port=8000,
                gpu_memory_utilization=0.85,max_model_len=18432,max_images=1,
                assets="/content/assets.json",model_asset="qwen-model",
                scratch="/content/research-assets",experiment_mode="normal",
                model_profile="qwen35_08b",spec_method=None,draft_asset=None,
                speculative_config_json=None,engine_mode="stock",
                source_root=None,source_revision=None,
                model="Qwen/Qwen3.5-0.8B",output_root=folder,
                throughput_sweep=True,h2o_policy_manifest=None)
            receipt={"role":"model","tree_sha256":"b"*64,
                 "staged_path":"/content/research-assets/qwen-model",
                 "source":{"type":"huggingface","repo_id":"Qwen/Qwen3.5-0.8B",
                           "revision":"a"*40}}
            with patch.dict(serve.os.environ,{"H2O_VLLM_MODE":"disabled"},clear=True):
                with patch.object(serve,"verify_asset",return_value=receipt),patch.object(
                    serve,"verify_engine",return_value={"installed_version":"0.29.0"}),patch.object(
                    serve,"_start_ticks",return_value=7),patch.object(
                    serve.subprocess,"Popen",return_value=SimpleNamespace(pid=8888)):
                    record=serve.start(args)
            self.assertIn("--no-enable-prefix-caching",record["argv_without_tokens"])
            self.assertEqual(record["argv_without_tokens"][
                record["argv_without_tokens"].index("--max-num-seqs")+1],"32")
            self.assertTrue(record["throughput_sweep_server"])

    def test_gpu_cell_without_execute_always_blocks(self):
        from research_ops.throughput.plan import DEFAULT_CONFIG
        args=SimpleNamespace(protocol=str(DEFAULT_CONFIG),model="qwen35_08b",
              arm="base",context=128,ccu=1,execute=False)
        with self.assertRaisesRegex(ValueError,"Explicit --execute"):
            runner.run_cell(args)


if __name__=="__main__":
    unittest.main()
