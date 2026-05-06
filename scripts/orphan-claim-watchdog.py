#!/usr/bin/env python3
"""
Orphan Claim Watchdog
Reverts 'in_progress' claims to 'open' if they have been stale for >4 hours.
Prevents deadlocks when agent workers crash or are killed without unclaiming.
"""
import json
import logging
import os
import subprocess
from datetime import datetime, timezone, timedelta

# Safety threshold: 4 hours.
THRESHOLD = timedelta(hours=4)


logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger("orphan-claim-watchdog")


def _bd_env():
    env = os.environ.copy()
    env.setdefault("BEADS_NO_DAEMON", "1")
    return env


def get_stale_claims():
    cmd = ["bd", "list", "--status=in_progress", "--json"]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True, env=_bd_env())
        issues = json.loads(res.stdout)
    except (subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        logger.exception("Unable to list in-progress bead claims: %s", exc)
        return []
    if isinstance(issues, dict):
        issues = issues.get("issues", [])

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
        logger.info("Unclaiming stale issue %s", issue_id)
        # bd unclaim is status-aware: only reverts in_progress -> open.
        try:
            subprocess.run(["bd", "unclaim", issue_id], check=True, env=_bd_env())
        except subprocess.CalledProcessError as exc:
            logger.exception("Failed to unclaim stale issue %s: %s", issue_id, exc)


if __name__ == "__main__":
    main()
