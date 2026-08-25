"""Run the probe as a module: python -m llm_preflight.probe [base_url] [model]."""

import sys

from .probe import main

sys.exit(main())
