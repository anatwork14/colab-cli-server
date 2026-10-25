"""WorkspaceBroker token file: atomic 0600 writes, no symlink write-through, parent dir left alone."""
import json
import os
import stat
import tempfile
import unittest
from pathlib import Path

from colab_adapter.broker import WorkspaceBroker
from colab_adapter.provider import ManagedSessionRegistry


def _mode(p):
    return stat.S_IMODE(os.lstat(p).st_mode)


class BrokerTokenFileTest(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.tmp = Path(self._td.name)
        self.registry = ManagedSessionRegistry(self.tmp / "sessions.json")

    def tearDown(self):
        self._td.cleanup()

    def _broker(self, tokens_file, tokens=("fake-token-A",)):
        return WorkspaceBroker(registry=self.registry, auth_tokens=set(tokens), tokens_file=tokens_file,
                               artifacts_dir=self.tmp / "art")

    def test_new_file_0600_and_existing_parent_untouched(self):
        parent = self.tmp / "shared"
        parent.mkdir()
        os.chmod(parent, 0o755)
        tf = parent / "tokens.json"
        self._broker(tf)
        self.assertEqual(_mode(tf), 0o600)
        self.assertEqual(_mode(parent), 0o755)
        self.assertEqual(json.loads(tf.read_text())["tokens"], ["fake-token-A"])

    def test_created_parent_is_0700(self):
        tf = self.tmp / "new" / "deep" / "tokens.json"
        self._broker(tf)
        self.assertEqual(_mode(tf.parent), 0o700)

    def test_resave_over_existing_0644_file_ends_0600_without_leftovers(self):
        tf = self.tmp / "tokens.json"
        tf.write_text("{}")
        os.chmod(tf, 0o644)
        b = self._broker(tf, tokens=("fake-token-B",))
        # constructor only saves when the file is absent; force the rewrite path explicitly
        b._save_tokens_to_file()
        self.assertEqual(_mode(tf), 0o600)
        self.assertEqual(json.loads(tf.read_text())["tokens"], ["fake-token-B"])
        self.assertEqual([p.name for p in self.tmp.iterdir() if ".tmp." in p.name], [])

    def test_operator_configured_symlink_is_resolved_once_and_target_is_private(self):
        # The broker resolves the operator-configured path at construction (load and save agree on the
        # target), so a symlinked tokens_file means "use that target". The target must end up 0600.
        real = self.tmp / "volume" / "tokens.json"
        real.parent.mkdir()
        tf = self.tmp / "tokens.json"
        os.symlink(str(real), str(tf))
        self._broker(tf)
        self.assertEqual(_mode(real), 0o600)
        self.assertEqual(json.loads(real.read_text())["tokens"], ["fake-token-A"])
        self.assertEqual([p.name for p in real.parent.iterdir() if ".tmp." in p.name], [])

    def test_failed_replace_keeps_previous_file_and_removes_temp(self):
        from unittest import mock

        tf = self.tmp / "tokens.json"
        tf.write_text('{"tokens": ["old"]}')
        os.chmod(tf, 0o600)
        b = self._broker(tf)
        with mock.patch("os.replace", side_effect=OSError("boom")):
            with self.assertRaises(OSError):
                b._save_tokens_to_file()
        self.assertEqual(json.loads(tf.read_text()), {"tokens": ["old"]})
        self.assertEqual([p.name for p in self.tmp.iterdir() if ".tmp." in p.name], [])


if __name__ == "__main__":
    unittest.main()
