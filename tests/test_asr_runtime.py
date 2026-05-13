import subprocess
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


_install_astrbot_stub()

from core.summary import asr_runtime


class AsrRuntimeSubprocessTests(unittest.TestCase):
    def test_python_module_probe_detaches_child_stdin(self):
        process = SimpleNamespace(
            poll=lambda: 0,
            communicate=lambda timeout=1: ("", ""),
        )

        with patch.object(
            asr_runtime.subprocess,
            "Popen",
            return_value=process,
        ) as popen:
            self.assertTrue(asr_runtime._python_has_modules("python", ["json"]))

        self.assertEqual(popen.call_args.kwargs["stdin"], subprocess.DEVNULL)


if __name__ == "__main__":
    unittest.main()
