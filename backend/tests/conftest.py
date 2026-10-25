"""Test-wide setup.

API key auth is off by default in tests, whatever API_KEY the local .env has (environment
variables override .env). Tests that check auth turn it on with a dependency override.
"""

import os

os.environ["API_KEY"] = ""
