"""Permission, atomicity, and symlink safety of DriveCredentials.save().

Named without the word "credential" on purpose: the owner's settings deny
Read(**/*credential*) and the owner chose a different filename over an exception.
All tokens below are fake placeholders used only inside temporary directories.
"""
import json
import os
import shutil
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from colab_adapter.errors import DriveAuthenticationError
from colab_adapter.storage.google_drive import DriveCredentials, GoogleDriveArtifactStore

ACCESS_TOKEN = "ya29.SECRET-ACCESS-TOKEN-XYZ"
REFRESH_TOKEN = "1//SECRET-REFRESH-TOKEN-XYZ"
CLIENT_SECRET = "SECRET-CLIENT-SECRET-XYZ"


def _mode(p):
    return stat.S_IMODE(os.stat(p).st_mode)


def _creds(access=ACCESS_TOKEN):
    return DriveCredentials(
        client_id="cid",
        client_secret=CLIENT_SECRET,
        refresh_token=REFRESH_TOKEN,
        access_token=access,
        token_expiry=9999999999.0,
    )


class _FakeHttp:
    def __init__(self, access_token="ya29.REFRESHED-TOKEN-ABC"):
        self.access_token = access_token
        self.calls = []

    def request(self, method, url, headers=None, data=None, timeout=30.0):
        self.calls.append((method, url))
        body = json.dumps({"access_token": self.access_token, "expires_in": 3600}).encode("utf-8")
        return 200, {}, body


@unittest.skipUnless(os.name == "posix", "POSIX permission semantics")
class TestDriveCredentialsPermissions(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="drive_creds_perm_")
        self.addCleanup(lambda: shutil.rmtree(self.test_dir, ignore_errors=True))
        self.target = Path(self.test_dir) / "credentials.json"

    def _write_old(self, mode, content=None):
        if content is None:
            content = json.dumps({"access_token": "OLD-TOKEN", "refresh_token": "OLD-REFRESH"})
        self.target.write_text(content, encoding="utf-8")
        os.chmod(self.target, mode)

    def _assert_no_leftover_tmp(self, directory):
        leftovers = [n for n in os.listdir(directory) if ".tmp." in n]
        self.assertEqual(leftovers, [], f"leftover temp files: {leftovers}")

    # --- permission tightening on an existing file -------------------------

    def test_existing_0644_file_becomes_0600_with_correct_content(self):
        self._write_old(0o644)
        _creds().save(self.target)
        self.assertEqual(_mode(self.target), 0o600)
        data = json.loads(self.target.read_text(encoding="utf-8"))
        self.assertEqual(data["access_token"], ACCESS_TOKEN)
        self.assertEqual(data["refresh_token"], REFRESH_TOKEN)
        self.assertEqual(data["token_uri"], "https://oauth2.googleapis.com/token")
        self._assert_no_leftover_tmp(self.test_dir)

    def test_existing_0666_file_becomes_0600(self):
        self._write_old(0o666)
        _creds().save(self.target)
        self.assertEqual(_mode(self.target), 0o600)

    def test_existing_0777_file_becomes_0600(self):
        self._write_old(0o777)
        _creds().save(self.target)
        self.assertEqual(_mode(self.target), 0o600)

    def test_new_file_is_0600(self):
        _creds().save(self.target)
        self.assertEqual(_mode(self.target), 0o600)

    def test_new_token_never_visible_in_world_readable_file(self):
        """Capture the temp file right before it is renamed into place."""
        self._write_old(0o644)
        seen = {}
        real_replace = os.replace

        def spy_replace(src, dst, *a, **kw):
            st = os.stat(src)
            seen["mode"] = stat.S_IMODE(st.st_mode)
            with open(src, "rb") as fh:
                seen["bytes"] = fh.read()
            return real_replace(src, dst, *a, **kw)

        with mock.patch("colab_adapter.storage.google_drive.os.replace", side_effect=spy_replace):
            _creds().save(self.target)
        self.assertEqual(seen["mode"] & 0o077, 0, "temp file must not be group/other accessible")
        self.assertIn(ACCESS_TOKEN.encode(), seen["bytes"])
        self.assertEqual(_mode(self.target), 0o600)

    # --- failure handling and atomicity -----------------------------------

    def test_failure_during_rename_keeps_previous_file_intact(self):
        self._write_old(0o644)
        before = self.target.read_bytes()
        with mock.patch(
            "colab_adapter.storage.google_drive.os.replace", side_effect=OSError("boom")
        ):
            with self.assertRaises(DriveAuthenticationError):
                _creds().save(self.target)
        self.assertEqual(self.target.read_bytes(), before)
        self.assertEqual(_mode(self.target), 0o644)
        self._assert_no_leftover_tmp(self.test_dir)

    def test_failure_during_fsync_cleans_temp_and_keeps_previous_file(self):
        self._write_old(0o600)
        before = self.target.read_bytes()
        with mock.patch("colab_adapter.storage.google_drive.os.fsync", side_effect=OSError("disk gone")):
            with self.assertRaises(DriveAuthenticationError):
                _creds().save(self.target)
        self.assertEqual(self.target.read_bytes(), before)
        self._assert_no_leftover_tmp(self.test_dir)

    def test_error_message_never_contains_token_values(self):
        self._write_old(0o600)
        with mock.patch(
            "colab_adapter.storage.google_drive.os.replace",
            side_effect=OSError(28, "No space left on device"),
        ):
            with self.assertRaises(DriveAuthenticationError) as cm:
                _creds().save(self.target)
        text = str(cm.exception)
        for secret in (ACCESS_TOKEN, REFRESH_TOKEN, CLIENT_SECRET):
            self.assertNotIn(secret, text)

    # --- symlink handling --------------------------------------------------

    def test_symlink_target_is_not_written_through(self):
        real = Path(self.test_dir) / "real_secret_store.json"
        real.write_text("ORIGINAL-CONTENT", encoding="utf-8")
        os.chmod(real, 0o644)
        os.symlink(real, self.target)

        _creds().save(self.target)

        self.assertEqual(real.read_text(encoding="utf-8"), "ORIGINAL-CONTENT")
        self.assertEqual(_mode(real), 0o644)
        self.assertFalse(os.path.islink(self.target))
        self.assertEqual(_mode(self.target), 0o600)
        self.assertEqual(json.loads(self.target.read_text())["access_token"], ACCESS_TOKEN)

    def test_dangling_symlink_target_is_replaced_not_followed(self):
        outside = Path(self.test_dir) / "outside_new_file.json"
        os.symlink(outside, self.target)
        _creds().save(self.target)
        self.assertFalse(outside.exists(), "save must not create the link's destination")
        self.assertFalse(os.path.islink(self.target))
        self.assertEqual(_mode(self.target), 0o600)

    # --- parent directory handling ----------------------------------------

    def test_missing_parent_dirs_created_0700(self):
        nested = Path(self.test_dir) / "a" / "b" / "creds.json"
        _creds().save(nested)
        self.assertEqual(_mode(Path(self.test_dir) / "a"), 0o700)
        self.assertEqual(_mode(Path(self.test_dir) / "a" / "b"), 0o700)
        self.assertEqual(_mode(nested), 0o600)

    def test_existing_parent_dir_permissions_untouched(self):
        os.chmod(self.test_dir, 0o755)
        _creds().save(self.target)
        self.assertEqual(_mode(self.test_dir), 0o755)
        self.assertEqual(_mode(self.target), 0o600)

    # --- refresh path ------------------------------------------------------

    def test_refresh_path_produces_0600_file_when_previous_was_0644(self):
        self._write_old(0o644)
        creds = _creds(access="expired-token")
        creds.token_expiry = 1.0  # force refresh
        store = GoogleDriveArtifactStore(
            credentials=creds,
            credentials_file=self.target,
            local_staging_dir=Path(self.test_dir) / "staging",
            http_client=_FakeHttp(),
            initial_backoff_sec=0.01,
        )
        token = store._ensure_token()
        self.assertEqual(token, "ya29.REFRESHED-TOKEN-ABC")
        self.assertEqual(_mode(self.target), 0o600)
        data = json.loads(self.target.read_text(encoding="utf-8"))
        self.assertEqual(data["access_token"], "ya29.REFRESHED-TOKEN-ABC")
        self._assert_no_leftover_tmp(self.test_dir)

    # --- format compatibility ----------------------------------------------

    def test_roundtrip_load_after_save(self):
        _creds().save(self.target)
        loaded = DriveCredentials.load(self.target)
        self.assertEqual(loaded.access_token, ACCESS_TOKEN)
        self.assertEqual(loaded.refresh_token, REFRESH_TOKEN)
        self.assertEqual(loaded.client_secret, CLIENT_SECRET)


if __name__ == "__main__":
    unittest.main()
