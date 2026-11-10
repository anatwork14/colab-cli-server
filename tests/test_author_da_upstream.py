"""Offline authors' canonical source and provisional profile checks."""
import unittest
from research_ops.vllm_engine.author_da import (
    validate_vendor, candidate_profile, installation_plan, AUTHOR, AuthorDAError
)


class CanonicalAuthorsDA(unittest.TestCase):
    def test_exact_178_blobs(self):
        proof=validate_vendor()
        self.assertEqual(proof["file_count"],178)
        self.assertTrue(proof["all_original_git_blobs_exact"])
        self.assertEqual(proof["commit"],"24110b7b441ad9423f55952a2c02d23eac80ec52")

    def test_author_plugin_and_custom_metadata_actual_source(self):
        root=AUTHOR/"src"/"dattn"/"engine"
        plugin=(root/"plugin.py").read_text()
        builder=(root/"builder.py").read_text()
        self.assertIn("register_backend",plugin)
        self.assertIn("DattnLogitsProcessor",plugin)
        self.assertIn("supports_update_block_table = False",builder)
        self.assertTrue((root/"worker_ext.py").is_file())
        self.assertTrue((AUTHOR/"tests"/"gpu"/"test_engine_dattn.py").is_file())

    def test_qwen35_a100_candidate_is_never_called_qualified(self):
        out=candidate_profile(hardware="a100",backend="TRITON_ATTN",
                local_model="/content/research-assets/qwen35-4b",max_model_len=32768)
        self.assertIn('model_id = "Qwen/Qwen3.5-4B"',out["profile_toml"])
        self.assertIn('capability = "sm80"',out["profile_toml"])
        self.assertIn("CUSTOM",out["profile_toml"])
        self.assertIn("DattnLogitsProcessor",out["profile_toml"])
        self.assertFalse(out["tested_with_vision_inputs"])
        self.assertFalse(out["tested_with_speculative_decoding"])

    def test_l4_flashinfer_needs_compacted_length_mirror(self):
        out=candidate_profile(hardware="l4",backend="FLASHINFER",
                local_model="/content/model",max_model_len=8192)
        self.assertIn('capability = "sm89"',out["profile_toml"])
        self.assertIn("needs_compacted_seq_lens_cpu = true",out["profile_toml"])

    def test_invalid_candidate_details_rejected(self):
        for kw in (
            dict(hardware="h100",backend="FLASHINFER",local_model="/a",max_model_len=1024),
            dict(hardware="a100",backend="INVALID",local_model="/a",max_model_len=1024),
            dict(hardware="a100",backend="FLASHINFER",local_model="relative",max_model_len=1024),
            dict(hardware="a100",backend="FLASHINFER",local_model="/a",max_model_len=1)
        ):
            with self.subTest(kw=kw):
                with self.assertRaises(AuthorDAError):
                    candidate_profile(**kw)

    def test_offline_installation_plan_not_gpu_execution(self):
        plan=installation_plan("/content/candidate.toml","candidate_sm80")
        self.assertIn("DRY_RUN",plan["status"])
        self.assertEqual(plan["python_required"],"3.13")
        self.assertFalse(any("execute" in x for x in plan["inspection"]))

if __name__ == "__main__":
    unittest.main()
