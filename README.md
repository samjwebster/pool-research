# Autonomous VFM pooling search

This repository searches for generalizable pooling functions over DINOv3
patch-level features. The five source datasets are exposed to the evaluator
only as opaque `Dataset_1` through `Dataset_5` identifiers.

## Quick start

Use the `pool-eval` environment:

```bash
conda run -n pool-eval python prepare.py
./run_eval.sh --force-baselines
./run_eval.sh --method MyPoolingMethod
```

Add methods as `BasePoolingMethod` subclasses in
[`pooling_methods/`](./pooling_methods/). Input features are kept in
[`data/features_unpooled/`](./data/features_unpooled/), and results are written
to `progress.csv` and `results.json`.

For persistent access across terminals and agent sessions, store the webhook
outside the repository:

```bash
mkdir -p ~/.config/pool-research
umask 077
printf '%s\n' 'DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...' \
  > ~/.config/pool-research/discord.env
chmod 600 ~/.config/pool-research/discord.env
```

`run_eval.sh` and `discord_webhook.py` load this file automatically. An
explicit `DISCORD_WEBHOOK_URL` environment variable still takes precedence.
The secrets file is intentionally outside Git.
