#!/bin/bash
# run_eval.sh

# Set the environment variable to use GPU 1
export CUDA_VISIBLE_DEVICES=1

# Load the per-user secret for subprocesses launched by this wrapper. The
# Python module also reads this file, covering direct evaluator invocation.
SECRETS_FILE="${POOL_RESEARCH_SECRETS_FILE:-$HOME/.config/pool-research/discord.env}"
if [ -f "$SECRETS_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  . "$SECRETS_FILE"
  set +a
fi

# Run using the pool-eval conda environment. Pass --method to evaluate one
# agent-created class; without it, all baselines are warmed up first.
exec conda run --no-capture-output -n pool-eval python evaluate.py "$@"