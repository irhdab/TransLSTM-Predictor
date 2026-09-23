"""Config package - re-exports config module for backward compatibility.

Supports both:
    import config.config as config
    from config import config
"""
from .config import *  # noqa: F401,F403
from . import config as _config_module  # noqa: F401
