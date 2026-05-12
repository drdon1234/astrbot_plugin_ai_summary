"""Logger shim for AstrBot and local syntax checks."""

try:
    from astrbot.api import logger
except ImportError:
    import logging

    logger = logging.getLogger("astrbot_plugin_ai_summary")

