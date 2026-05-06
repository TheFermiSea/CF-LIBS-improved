#!/usr/bin/env python3
"""
Orphan Claim Watchdog
Reverts 'in_progress' claims to 'open' if they have been stale for >4 hours.
Prevents deadlocks when agent workers crash or are killed without unclaiming.
"""
import subprocess
import json
from datetime import datetime, timezone, timedelta

# Safety threshold: 4 hours.
THRESHOLD = timedelta(hours=4)

def get_stale_claims():
    cmd = ["bd", "query", "status=in_progress", "--json"]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        issues = json.loads(res.stdout)
    except Exception:
        return []

    now = datetime.now(timezone.utc)
    stale = []
    for issue in issues:
        updated_at_str = issue.get("updated_at")
        if not updated_at_str:
            continue
        try:
            # bd uses Z suffix for UTC
            dt = datetime.fromisoformat(updated_at_str.replace("Z", "+00:00"))
        except ValueError:
            continue
        if now - dt > THRESHOLD:
            stale.append(issue["id"])
    return stale

def main():
    for issue_id in get_stale_claims():
        print(f"Watchdog: Unclaiming stale issue {issue_id}")
        # bd unclaim is status-aware: only reverts in_progress -> open.
        subprocess.run(["bd", "unclaim", issue_id], check=True)

if __name__ == "__main__":
    main()
