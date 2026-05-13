import asyncio
import importlib
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch


def _install_astrbot_stub():
    if "astrbot.api" in sys.modules:
        return
    astrbot_module = types.ModuleType("astrbot")
    api_module = types.ModuleType("astrbot.api")
    api_module.logger = SimpleNamespace(
        debug=lambda *args, **kwargs: None,
        info=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
    )
    sys.modules.setdefault("astrbot", astrbot_module)
    sys.modules["astrbot.api"] = api_module


class SummaryManagerSubprocessTests(unittest.IsolatedAsyncioTestCase):
    async def test_probe_duration_detaches_child_stdin(self):
        _install_astrbot_stub()
        manager_module = importlib.import_module("core.summary.manager")

        process = SimpleNamespace(
            communicate=lambda: asyncio.sleep(0, result=(b"12.5", b"")),
            returncode=0,
        )

        async def fake_create_subprocess_exec(*args, **kwargs):
            return process

        manager = object.__new__(manager_module.AISummaryManager)
        manager._ffprobe_path = lambda: "ffprobe"

        with patch.object(
            manager_module.asyncio,
            "create_subprocess_exec",
            side_effect=fake_create_subprocess_exec,
        ) as create_subprocess_exec:
            duration = await manager._probe_duration("video.mp4")

        self.assertEqual(duration, 12.5)
        self.assertEqual(
            create_subprocess_exec.call_args.kwargs["stdin"],
            manager_module.asyncio.subprocess.DEVNULL,
        )


if __name__ == "__main__":
    unittest.main()
