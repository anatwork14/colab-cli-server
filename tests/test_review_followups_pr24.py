"""Opus checkpoint-2 follow-ups: Drive read-path validation, query escaping, manifest name validation."""
import tempfile
import unittest
from pathlib import Path

from colab_adapter.errors import ArtifactValidationError, ManifestValidationError
from colab_adapter.project_manifest import _parse_raw_manifest
from colab_adapter.storage.google_drive import GoogleDriveArtifactStore
from colab_adapter.storage.validation import validate_identifier


class _CountingClient:
    def __init__(self):
        self.calls = []

    def request(self, method, url, headers=None, data=None):
        self.calls.append((method, url))
        return 500, {}, b"{}"


def _store(root):
    st = GoogleDriveArtifactStore.__new__(GoogleDriveArtifactStore)
    st.staging_dir = Path(root)
    return st


class DriveReadPathValidationTest(unittest.TestCase):
    def test_read_paths_reject_traversal_and_newline_ids_without_side_effects(self):
        digest = "@sha256_" + "a" * 64
        bad = ["../x/f", "p/../f/x", "proj\n/run/f", "p/run\n/f"]
        with tempfile.TemporaryDirectory() as td:
            st = _store(td)
            before = sorted(Path(td).rglob("*"))
            for aid in bad:
                with self.assertRaises(ArtifactValidationError, msg=aid):
                    st.get_metadata(aid + digest)
                with self.assertRaises(ArtifactValidationError, msg=aid):
                    st.exists(aid + digest)
                with self.assertRaises(ArtifactValidationError, msg=aid):
                    st.retrieve(aid + digest, Path(td) / "out" / "dest")
            for pid, rid in [("..", "r"), ("p", "r\n"), ("p\n", "r")]:
                with self.assertRaises(ArtifactValidationError):
                    st.list_artifacts(pid, rid)
            self.assertEqual(before, sorted(Path(td).rglob("*")))

    def test_identifier_rule_rejects_trailing_newline(self):
        for v in ("run\n", "proj\n", "a\x00b", "kK"):
            with self.assertRaises(ArtifactValidationError):
                validate_identifier(v)


class DriveQueryEscapingTest(unittest.TestCase):
    def test_backslash_cannot_break_out_of_query_quote(self):
        st = GoogleDriveArtifactStore.__new__(GoogleDriveArtifactStore)
        st.list_page_size = 100
        captured = []

        def fake(method, url, headers=None, data=None):
            captured.append(url)
            return 200, {}, b'{"files": []}'

        st._request_with_retry = fake
        st._list_files_in_folder("a\\' or name contains '", "folder1")
        import urllib.parse

        q = urllib.parse.unquote(captured[0])
        self.assertIn("name = 'a\\\\\\' or name contains \\''", q)


class ManifestNameValidationTest(unittest.TestCase):
    def test_unusable_project_name_rejected_at_manifest_load(self):
        with tempfile.TemporaryDirectory() as td:
            for bad in ("my project", "x/y", "a\nb", "n" * 200):
                with self.assertRaises(ManifestValidationError, msg=bad):
                    _parse_raw_manifest(
                        {"project": {"name": bad}, "execution": {"entrypoint": "train.py"}},
                        Path(td),
                    )


if __name__ == "__main__":
    unittest.main()
