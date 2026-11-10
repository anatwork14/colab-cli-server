"""Offline publisher source tests; mocks only. Never generate benchmark records."""
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from research_ops.benchmarks import sources


class PublisherSourcesTests(unittest.TestCase):
    def test_catalog_only_authors(self):
        c=json.loads(Path("research/benchmarks/OFFICIAL_SOURCES.json").read_text())
        self.assertEqual({x["id"]:x["dataset_repo"] for x in c["selected"]},{
            "memlens":"xiyuRenBill/MEMLENS",
            "mmlongbench_doc":"yubo2333/MMLongBench-Doc"})
        self.assertTrue(all(x["hf_full_revision"].startswith("REQUIRES_SERVER_SIDE")
                            for x in c["selected"]))

    def test_release_selection(self):
        self.assertEqual(sources._select("memlens","32k")["questions"],"dataset_32k.json")
        self.assertEqual(sources._select("memlens","256k")["expected_questions"],789)
        self.assertEqual(sources._select("mmlongbench_doc","default")["questions"],"samples.json")
        with self.assertRaises(ValueError):
            sources._select("memlens","257k")

    def test_floating_and_short_revisions_rejected(self):
        for sha in ("main","v1","2ff6aa9","afa101a","A"*40,"a"*39):
            with self.subTest(revision=sha):
                with self.assertRaises(ValueError):
                    sources._pin(sha)
        self.assertEqual(sources._pin("a"*40),"a"*40)

    def test_memlens_resolve_checks_full_inventory(self):
        names=["dataset_32k.json"]+["release_images/haystack_images/%06d.jpg"%n for n in range(4695)]
        info=SimpleNamespace(sha="b"*40,
            siblings=[SimpleNamespace(rfilename=x) for x in names])
        with patch.object(sources,"_hf_api",return_value=SimpleNamespace(dataset_info=lambda **k:info)):
            r=sources.resolve("memlens","32k")
        self.assertFalse(r["has_downloaded_bytes"])
        self.assertEqual(r["hf_commit_sha"],"b"*40)

    def test_missing_media_blocks_author_pin(self):
        info=SimpleNamespace(sha="b"*40,
            siblings=[SimpleNamespace(rfilename="dataset_32k.json")])
        with patch.object(sources,"_hf_api",return_value=SimpleNamespace(dataset_info=lambda **k:info)):
            with self.assertRaises(RuntimeError):
                sources.resolve("memlens","32k")

    def test_mmlong_resolve_checks_pdf_count(self):
        names=["data/train-00000-of-00001.parquet"]+[
            "documents/doc-%04d.pdf"%n for n in range(135)]
        info=SimpleNamespace(sha="c"*40,
            siblings=[SimpleNamespace(rfilename=x) for x in names])
        with patch.object(sources,"_hf_api",return_value=SimpleNamespace(dataset_info=lambda **k:info)):
            r=sources.resolve("mmlongbench_doc","default")
        self.assertEqual(r["hf_commit_sha"],"c"*40)

    def test_plan_offline_does_not_allocate_or_download(self):
        with tempfile.TemporaryDirectory() as folder:
            target=Path(folder)/"future"
            with patch.object(sources,"_hf_api",side_effect=AssertionError("network forbidden")):
                r=sources.fetch("memlens","32k","a"*40,target,execute=False)
            self.assertEqual(r["status"],"PLAN_ONLY_NO_DOWNLOAD")
            self.assertFalse(target.exists())

    def test_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            target=Path(folder)/"existing"
            target.mkdir()
            with self.assertRaises(FileExistsError):
                sources.fetch("memlens","32k","a"*40,target,execute=True)

    def test_local_old_mirror_not_automatically_authenticated(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/"mirror"
            root.mkdir()
            fake={"tree":{"tree_sha256":"d"*64,"file_count":2,"total_bytes":100},
                  "question_json_sha256":"e"*64,
                  "audit":{"status":"COMPLETE_FOR_ENGINEERING_PILOT"}}
            with patch.object(sources,"_inventory",return_value=fake):
                r=sources.audit_existing("memlens","32k",root)
            self.assertEqual(r["status"],
                             "AUDITED_LOCAL_DATA_WITHOUT_CONFIRMED_HF_PROVENANCE")


if __name__=="__main__":
    unittest.main()
