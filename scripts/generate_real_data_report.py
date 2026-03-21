"""Generate real-data confirmation report."""
import json
from pathlib import Path

from scripts._report_data import REPORT


def main():
    out = Path("output/validation/real_data_confirmation_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(REPORT, indent=2))
    print(f"Report: {out}")


if __name__ == "__main__":
    main()
