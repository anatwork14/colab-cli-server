"""H2O + DA control-plane boundary: no fake runtime eviction/GPU calls."""
import unittest

from research_ops.vllm_engine.h2o_bridge import (
    H2OIntegrationBlocked, PINNED_ENGINE, PINNED_H2O, PINNED_VLLM,
    validate_h2o_policy_receipt, plan_protected_eviction_candidates,
    reject_h2o_gpu_execution,
)


def receipt(block_size=4):
    return {
        "schema_version":"h2o-vllm/1", "engine":"vllm",
        "engine_version":PINNED_ENGINE, "engine_tag_sha":PINNED_VLLM,
        "method":"h2o-policy-simulation", "mode":"policy-only",
        "runtime_verified":False, "attention_probabilities_collected":False,
        "physical_kv_eviction":False, "real_h2o_eviction":False,
        "gpu_qualified":False, "performance_claims_allowed":False,
        "source":{"repository":"https://github.com/FMInference/H2O",
                  "commit":PINNED_H2O},
        "policy":{"heavy_hitter_tokens":4,"recent_tokens":4,
                  "sink_tokens":0,"block_size":block_size},
    }


class H2OBridgeReviewTests(unittest.TestCase):
    def test_policy_provenance_must_be_pinned(self):
        self.assertEqual(validate_h2o_policy_receipt(receipt())["engine_version"],"0.29.0")
        for key, bad in (("engine_version","0.28.0"),
                         ("physical_kv_eviction",True),
                         ("gpu_qualified",True),
                         ("attention_probabilities_collected",True),
                         ("method","h2o"),
                         ("mode","passthrough")):
            with self.subTest(key=key):
                manifest=receipt()
                manifest[key]=bad
                with self.assertRaises(H2OIntegrationBlocked):
                    validate_h2o_policy_receipt(manifest)

    def test_da_evidence_and_response_protect_candidate_blocks(self):
        args=dict(manifest=receipt(),sequence_length=20,block_size=4,
                  attention_group="full_attention",stage="decode",
                  per_head_h2o_evict_positions=[list(range(16)),list(range(16))],
                  da_important_positions=[4,5],
                  grounded_evidence_positions=[8],
                  current_response_positions=[12,13],
                  sink_positions=[0])
        p=plan_protected_eviction_candidates(**args)
        self.assertEqual(p.candidate_full_blocks,())
        self.assertFalse(p.physical_eviction_performed)
        self.assertIsNone(p.physical_kv_bytes_freed)
        self.assertFalse(p.gpu_qualified)

    def test_only_common_full_blocks_can_be_nominated(self):
        p=plan_protected_eviction_candidates(
            manifest=receipt(),sequence_length=19,block_size=4,
            per_head_h2o_evict_positions=[list(range(4,19)),list(range(4,18))],
            da_important_positions=[8],grounded_evidence_positions=[13],
            current_response_positions=[16,17,18],sink_positions=[0,1,2,3])
        # Block 1 is the only full block all heads nominate and unprotected.
        self.assertEqual(p.candidate_full_blocks,(1,))
        self.assertEqual(p.status,"PROTECTION_POLICY_ONLY_PHYSICAL_EVICTION_BLOCKED")

    def test_head_disagreement_is_not_physical_eviction(self):
        p=plan_protected_eviction_candidates(
            manifest=receipt(),sequence_length=12,block_size=4,
            per_head_h2o_evict_positions=[[4,5,6,7],[4,5,6]],
            da_important_positions=[],grounded_evidence_positions=[],
            current_response_positions=[8,9,10,11],sink_positions=[0,1,2,3])
        self.assertEqual(p.candidate_full_blocks,())

    def test_denies_gdn_prefill_speculative_and_block_size_drift(self):
        base=dict(manifest=receipt(),sequence_length=16,block_size=4,
                  per_head_h2o_evict_positions=[list(range(4,8))],
                  da_important_positions=[],grounded_evidence_positions=[],
                  current_response_positions=[12,13,14,15],sink_positions=[0,1,2,3])
        for extra in ({"attention_group":"gated_delta_net"},
                      {"stage":"prefill"},{"speculative_method":"mtp"},
                      {"speculative_method":"dflash"},
                      {"speculative_method":"eagle3"},{"block_size":16}):
            with self.subTest(extra=extra):
                with self.assertRaises(H2OIntegrationBlocked):
                    plan_protected_eviction_candidates(**dict(base,**extra))

    def test_fails_closed_on_incomplete_signals(self):
        common=dict(manifest=receipt(),sequence_length=16,block_size=4,
                    per_head_h2o_evict_positions=[[4,5,6,7]],
                    da_important_positions=[],grounded_evidence_positions=[],
                    current_response_positions=[12,13,14,15],sink_positions=[0,1,2,3])
        for extra in ({"da_important_positions":None},
                      {"grounded_evidence_positions":None},
                      {"per_head_h2o_evict_positions":[]},
                      {"current_response_positions":[999]},
                      {"sink_positions":[False]}):
            with self.subTest(extra=extra):
                with self.assertRaises(ValueError):
                    plan_protected_eviction_candidates(**dict(common,**extra))
        with self.assertRaises(H2OIntegrationBlocked):
            reject_h2o_gpu_execution()


if __name__ == "__main__":
    unittest.main()
