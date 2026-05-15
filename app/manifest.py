"""Run manifest writer — records metadata for every analysis run."""

import json
import os
from datetime import datetime, timezone


def write_manifest(output_dir: str, **fields) -> str:
    """Write a run_manifest.json to *output_dir* and return the path."""
    os.makedirs(output_dir, exist_ok=True)
    now = datetime.now(timezone.utc)
    manifest = {
        "run_id": now.strftime("%Y%m%dT%H%M%SZ"),
        "timestamp": now.isoformat(),
        **fields,
    }
    path = os.path.join(output_dir, "run_manifest.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    return path
