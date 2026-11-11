"""
# limiter.py - the shared slowapi rate limiter

- Lives outside main.py so routes/*.py can import it without a circular import.
"""

import os

from slowapi import Limiter
from slowapi.util import get_remote_address

# config_filename=devnull: slowapi would otherwise parse .env with the platform codec and
# crash on Windows (cp1252); none of its env-driven settings are used.
limiter = Limiter(key_func=get_remote_address, config_filename=os.devnull)
