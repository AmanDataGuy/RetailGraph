"""
RetailGraph — shared rate limiter instance.

A separate module (not defined in main.py) so route modules can import it
without a circular import: routes/*.py are imported BY main.py to build the
app, so they can't import `limiter` back out of main.py.
"""

import os

from slowapi import Limiter
from slowapi.util import get_remote_address

# slowapi auto-reads .env for its own RATELIMIT_* settings (unused here) via
# Starlette's Config, which decodes the file with the platform's default
# codec — cp1252 on Windows — and crashes on this repo's real .env (contains
# a non-cp1252 byte). We don't use slowapi's env-driven config at all, so
# point it at the null device instead of letting it guess.
limiter = Limiter(key_func=get_remote_address, config_filename=os.devnull)
