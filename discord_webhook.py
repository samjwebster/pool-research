"""Optional Discord notifications for pooling experiments."""

import os
from pathlib import Path

import requests


CONFIG_PATH = Path(
    os.environ.get(
        "POOL_RESEARCH_SECRETS_FILE",
        Path.home() / ".config" / "pool-research" / "discord.env",
    )
)


def _webhook_url():
    url = os.environ.get("DISCORD_WEBHOOK_URL")
    if url:
        return url
    if not CONFIG_PATH.is_file():
        return None
    for line in CONFIG_PATH.read_text().splitlines():
        line = line.strip()
        if line.startswith("DISCORD_WEBHOOK_URL="):
            return line.partition("=")[2].strip().strip("\"'")
    return None


def send_evaluation(result, baseline_or_best=None):
    """Send one evaluation result; missing webhook configuration is a no-op.

    The environment variable takes precedence over the persistent per-user
    config file so CI and explicit shell configuration remain supported.
    """
    url = _webhook_url()
    if not url:
        return
    status = "NEW BEST" if result["all_time_best"] else "DISCARDED"
    color = 0x2ECC71 if result["all_time_best"] else 0xE74C3C
    delta = result["mean_accuracy"] - baseline_or_best if baseline_or_best is not None else 0.0
    datasets = "\n".join(
        f"**{name}:** {result[name]:.4f}"
        for name in ("Dataset_1", "Dataset_2", "Dataset_3", "Dataset_4", "Dataset_5")
    )
    payload = {"embeds": [{
        "title": f"{status}: {result['method_name']}",
        "description": result["description"],
        "color": color,
        "fields": [
            {"name": "Per-dataset accuracy", "value": datasets, "inline": False},
            {"name": "Mean accuracy", "value": f"{result['mean_accuracy']:.4f}", "inline": True},
            {"name": "Delta vs prior best", "value": f"{delta:+.4f}", "inline": True},
        ],
    }]}
    response = requests.post(url, json=payload, timeout=10)
    response.raise_for_status()
