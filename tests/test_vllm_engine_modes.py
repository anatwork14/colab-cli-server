"""Offline six-mode/DA lifecycle checks; no torch, GPUs, or synthetic benchmarks."""
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from research_ops.vllm_engine.matrix import make_arm, matrix, command_for_arm, load_catalog, MODES, METHODS
from research_ops.vllm_engine.da import (
    RequestDAState, DAController, DAQualificationError, check_engine_proof,
    token_spans, GATES, SPEC_GATE,
)
from research_ops.benchmarks import serve


class EngineMatrixTests(unittest.TestCase):
    def test_six_categories_expand_to_fourteen_real_arms(self):
        arms=matrix()
        self.assertEqual(len(arms),14)
        self.assertEqual(len({a.slug for a in arms}),14)
        self.assertEqual({a.mode for a in arms},set(MODES))
        self.assertEqual({a.method for a in arms if a.method},set(METHODS))
        self.assertEqual(sum(not a.da_roles for a in arms),4)
        self.assertEqual(sum(bool(a.da_roles) for a in arms),10)

    def test_target_is_posttrained_vision_qwen_under_ten_billion(self):
        m=load_catalog()["models"]["qwen35_4b"]
        self.assertEqual(m["target_id"],"Qwen/Qwen3.5-4B")
        self.assertEqual(m["model_stage"],"post_trained")
        self.assertTrue(m["vision_capable"])
        self.assertLess(m["parameters_billion"],10)
        self.assertIsNone(m["hub_revision"])  # Do not invent 40-char HF SHAs.

    def test_native_mtp_requires_no_external_draft(self):
        a=make_arm("qwen35_4b","speculative","mtp")
        argv=command_for_arm(a,"/content/staged/model")
        self.assertNotIn("z-lab",str(argv))
        cfg=json.loads(argv[argv.index("--speculative-config")+1])
        self.assertEqual(cfg,{"method":"mtp","num_speculative_tokens":1})
        with self.assertRaises(ValueError):
            command_for_arm(a,"/content/model",draft_path="/content/draft")

    def test_exact_dflash_and_eagle3_draft_pairs(self):
        for method,expected in (
                ("dflash","z-lab/Qwen3.5-4B-DFlash"),
                ("eagle3","yuyijiong/Qwen3.5-4B-Eagle3")):
            with self.subTest(method=method):
                a=make_arm("qwen35_4b","speculative",method)
                self.assertEqual(a.draft_id,expected)
                with self.assertRaises(ValueError):
                    command_for_arm(a,"/content/target")
                argv=command_for_arm(a,"/content/target",draft_path="/content/draft")
                cfg=json.loads(argv[argv.index("--speculative-config")+1])
                self.assertEqual(cfg["model"],"/content/draft")
                self.assertEqual(cfg["method"],method)

    def test_da_never_masquerades_as_stock_attention(self):
        for arm in matrix():
            if arm.da_roles:
                with self.assertRaises(RuntimeError):
                    command_for_arm(arm,"/content/model",
                                    draft_path="/content/draft")
        self.assertEqual(make_arm("qwen35_4b","da_both","eagle3").da_roles,
                         ("target","draft"))
        self.assertEqual(make_arm("qwen35_4b","da_draft","mtp").da_roles,
                         ("draft",))

    def test_reject_non_matching_method_and_missing_checkpoint(self):
        with self.assertRaises(ValueError):
            make_arm("qwen35_4b","normal","mtp")
        with self.assertRaises(ValueError):
            make_arm("qwen35_4b","speculative",None)
        with self.assertRaises(ValueError):
            make_arm("qwen35_08b","speculative","dflash")
        with self.assertRaises(ValueError):
            make_arm("qwen35_2b","speculative","eagle3")


class FakeMachine:
    """Test-only state machine with deliberately visible mutation."""
    def __init__(self,spans,**kwargs):
        self.seen=[]
        self.spans=spans
    def step(self,token):
        self.seen.append(token)
    def get_kept_blocks(self,length):
        return [0] if not self.seen else list(self.seen)


class DeclarativeAttentionIsolationTests(unittest.TestCase):
    def test_processor_spans_not_character_offsets_or_overlap(self):
        self.assertEqual(token_spans({3:(10,15),1:(1,3)},32),
                         {1:(1,3),3:(10,15)})
        for val in ({1:(3,100)}, {1:(1,8),2:(7,9)}, {1:(1,1)}, {}):
            with self.subTest(spans=val):
                with self.assertRaises(ValueError):
                    token_spans(val,20)

    def test_two_requests_have_isolated_accepted_state(self):
        ctl=DAController(machine_factory=FakeMachine)
        a=ctl.register("req-a",{1:(1,3)},16,("draft","target"))
        b=ctl.register("req-b",{1:(6,8)},16,("draft",))
        a.commit([12])
        self.assertEqual(a.kept_blocks("target",17),[12])
        self.assertEqual(b.kept_blocks("draft",16),[0])
        ctl.unregister("req-a")
        self.assertNotIn("req-a",ctl.requests)

    def test_rejected_eagle_or_dflash_proposal_does_not_advance_target(self):
        s=RequestDAState("r",{1:(0,4)},16,("draft","target"),FakeMachine)
        candidate=s.preview_draft([1,2,3])
        self.assertEqual(candidate.seen,[1,2,3])
        self.assertEqual(s.kept_blocks("target",16),[0])
        s.commit([1,2,99],accepted_draft_count=2)
        self.assertEqual(s.accepted_tokens,[1,2,99])
        self.assertEqual(s.kept_blocks("draft",19),[1,2,99])
        self.assertEqual(s.kept_blocks("target",19),[1,2,99])
        self.assertIsNone(s.proposed_tokens)

    def test_verifier_mismatch_does_not_mutate_state(self):
        s=RequestDAState("r",{1:(0,4)},16,("draft",),FakeMachine)
        s.preview_draft([4,5])
        with self.assertRaises(DAQualificationError):
            s.commit([4,8],accepted_draft_count=2)
        self.assertEqual(s.accepted_tokens,[])
        s.discard_draft()
        s.commit([12])
        self.assertEqual(s.accepted_tokens,[12])

    def test_rollback_after_aborted_tag(self):
        s=RequestDAState("r",{1:(0,3)},16,("draft","target"),FakeMachine)
        s.preview_draft([8,9,10])
        s.discard_draft()
        s.commit([42])
        self.assertEqual(s.machines["target"].seen,[42])
        self.assertEqual(s.machines["draft"].seen,[42])

    def test_da_runtime_proof_required_even_if_engine_claims_da(self):
        with self.assertRaises(DAQualificationError):
            check_engine_proof({})
        with self.assertRaises(DAQualificationError):
            check_engine_proof({x:True for x in GATES},speculative=True)
        p={x:True for x in GATES}
        p[SPEC_GATE]=True
        p["vllm_engine_commit"]="f"*40
        self.assertIn("PROOF_PRESENT",check_engine_proof(p,speculative=True))
        p["image_patch_offsets_verified"]=False
        with self.assertRaises(DAQualificationError):
            check_engine_proof(p,speculative=True)


class ServingGuardTests(unittest.TestCase):
    def test_reject_da_before_any_gpu_side_effect(self):
        args=SimpleNamespace(execute=True,run_id="da-gate-01",
              port=8000,gpu_memory_utilization=0.85,max_model_len=4096,
              max_images=2,assets="/unavailable",model_asset="model",
              scratch="/content/assets",experiment_mode="da_normal",
              model_profile="qwen35_4b",spec_method=None,
              draft_asset=None,speculative_config_json=None)
        fake={"tree_sha256":"z"*64,"staged_path":"/content/model",
              "source":{"type":"huggingface","repo_id":"Qwen/Qwen3.5-4B"}}
        with patch.object(serve,"verify_asset",return_value=fake):
            with self.assertRaises(RuntimeError):
                serve.start(args)


if __name__=="__main__":
    unittest.main()
