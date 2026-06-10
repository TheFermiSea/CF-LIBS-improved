"""Path bootstrap: make scripts/campaign1 modules importable in tests."""

import sys
from pathlib import Path

CAMPAIGN_DIR = Path(__file__).resolve().parents[2] / "scripts" / "campaign1"
if str(CAMPAIGN_DIR) not in sys.path:
    sys.path.insert(0, str(CAMPAIGN_DIR))
