"""Check the actual campaign exec boundary without starting an engine."""

import errno
import os
import socket
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from dattn import cli
from dattn.campaign import driver

pytestmark = pytest.mark.contract


class CapturedExec(Exception):
    pass


@pytest.fixture(params=[(model, judge) for model in driver.MODELS for judge in (False, True)])
def engine_command(request, tmp_path, monkeypatch):
    model, judge = request.param
    captured = {}

    def capture(command, **kwargs):
        captured.update(command=command, env=kwargs["env"])
        raise CapturedExec

    tok = SimpleNamespace(
        get_chat_template=lambda: "{{ messages }}",
        backend_tokenizer=SimpleNamespace(to_str=lambda: "{}"),
    )
    # Stop at Popen. Do not query a GPU, bind a real port, or start any engine.
    with monkeypatch.context() as patch:
        patch.setattr(driver, "assert_free", lambda *args: None)
        patch.setattr(driver, "tokenizer", lambda model: tok)
        patch.setattr("socket.socket", MagicMock())
        patch.setattr(driver.subprocess, "Popen", capture)
        with (
            pytest.raises(CapturedExec),
            driver.server(
                SimpleNamespace(root=tmp_path),
                "2",
                18400,
                model,
                tmp_path / "engine.log",
                threading.Event(),
                judge=judge,
            ),
        ):
            pytest.fail("an engine must never start in this test")
    return model, judge, captured


def test_campaign_engine_command_is_executable(engine_command):
    _, judge, captured = engine_command
    command = captured["command"]
    executable = Path(command[0])
    assert executable == Path(sys.executable).with_name("vllm" if judge else "dattn")
    assert executable.is_file() and os.access(executable, os.X_OK)
    assert command[1] == "serve"  # Reject a regression to python -m dattn.
    # Exercise the real installed script and its imports with GPUs hidden and no PATH.
    env = dict(driver.cpu_env(), PATH="")
    result = subprocess.run([command[0], "--help"], env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def test_campaign_engine_flags_parse_through_both_clis(engine_command, monkeypatch):
    from vllm.entrypoints.cli.serve import ServeSubcommand
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    model, judge, captured = engine_command
    command = captured["command"]
    if not judge:
        args = cli.build_parser().parse_args(command[1:])
        assert args.command == "serve" and args.exec
        assert args.model == driver.MODELS[model] and args.profile == "parity"
        assert args.host == "127.0.0.1" and args.port == 18400
        # REMAINDER must contain only the intended vLLM flags. A misplaced dattn
        # flag after --extra can otherwise be silently consumed by the wrong parser.
        assert args.extra == ["--seed", command[-1]]
        assert command[command.index("--extra") + 1 :] == args.extra
        executed = []

        def capture(executable, argv):
            assert executable == argv[0]
            executed.extend(argv)
            raise CapturedExec

        with monkeypatch.context() as patch:
            for key in os.environ:
                if key.startswith("DATTN_"):
                    patch.delenv(key)
            patch.setenv("VLLM_PLUGINS", "dattn")
            patch.setenv("PATH", "")
            patch.setattr(cli, "_check_templates", lambda profile: 0)
            patch.setattr(os, "execvp", capture)
            # _cmd_serve mutates the environment, restore it after interception.
            patch.setattr(os, "environ", os.environ.copy())
            with pytest.raises(CapturedExec):
                cli._cmd_serve(args)
        command = executed
        executable = Path(command[0])
        assert executable == Path(sys.executable).with_name("vllm")
        assert executable.is_file() and os.access(executable, os.X_OK)

    parser = FlexibleArgumentParser()
    serve = ServeSubcommand()
    serve.subparser_init(parser.add_subparsers())
    args = parser.parse_args(command[1:])
    serve.validate(args)
    assert args.model_tag == ("Qwen/Qwen3.5-4B" if judge else driver.MODELS[model])
    assert args.host == "127.0.0.1" and args.port == 18400
    assert args.seed == int(captured["command"][-1])
    if judge:
        assert Path(args.chat_template).read_text() == "{{ messages }}"
        assert args.gpu_memory_utilization == 0.85
        assert captured["env"]["VLLM_PLUGINS"] == ""
    else:
        assert args.kv_cache_memory_bytes == 107374182400
        assert args.gpu_memory_utilization == 0.85
        assert args.max_num_batched_tokens == 16384
        assert args.max_num_seqs == (64 if model == "gemma" else 128)


@pytest.mark.parametrize("released", [False, True], ids=["occupied", "time_wait"])
def test_engine_port_probe_rejects_listener_but_accepts_time_wait(tmp_path, monkeypatch, released):
    def capture(*args, **kwargs):
        raise CapturedExec

    monkeypatch.setattr(driver, "assert_free", lambda *args: None)
    monkeypatch.setattr(driver.subprocess, "Popen", capture)
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        listener.settimeout(2)
        address = listener.getsockname()
        if released:
            # Make the server side close first so its port really enters TIME_WAIT.
            with socket.create_connection(address, timeout=2) as peer, listener.accept()[0] as accepted:
                accepted.settimeout(2)
                accepted.shutdown(socket.SHUT_WR)
                assert peer.recv(1) == b""
                peer.shutdown(socket.SHUT_WR)
                assert accepted.recv(1) == b""
            listener.close()
        # Prove this fixture would fail the original plain-bind probe.
        with socket.socket() as plain, pytest.raises(OSError) as unavailable:
            plain.bind(address)
        assert unavailable.value.errno == errno.EADDRINUSE

        with (
            pytest.raises(CapturedExec if released else OSError) as result,
            driver.server(
                SimpleNamespace(root=tmp_path),
                "2",
                address[1],
                "gemma",
                tmp_path / "engine.log",
                threading.Event(),
            ),
        ):
            pytest.fail("an engine must never start in this test")
        if not released:
            assert result.value.errno == errno.EADDRINUSE


def test_judge_refuses_gpu_still_occupied_by_generation(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, "gpu_processes", lambda: ({"2": "GPU-a"}, {"GPU-a": {12345}}))
    monkeypatch.setattr(driver, "tokenizer", lambda *args: pytest.fail("prepared judge on occupied GPU"))
    monkeypatch.setattr(driver.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("launched on occupied GPU"))
    with (
        pytest.raises(RuntimeError, match="GPU 2 REFUSED"),
        driver.server(
            SimpleNamespace(root=tmp_path), "2", 18400, "", tmp_path / "judge.log", threading.Event(), judge=True
        ),
    ):
        pytest.fail("judge shared the generation GPU")
