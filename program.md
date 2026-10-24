# Autonomous VFM pooling search

You are searching for generalizable pooling functions over anonymized DINOv3
patch features. Dataset names are intentionally opaque: do not infer domains
from `Dataset_1` through `Dataset_5`, and never inspect or use image paths.

## Required experiment loop

1. Create exactly one new `BasePoolingMethod` subclass in
   `pooling_methods/<method_name>.py`. Include a class docstring describing the
   mathematical hypothesis and mechanics. Implement `pool`; use `fit` only for
   state that is learned from the current dataset's unpooled features.
2. The package registry is dynamic, but verify that the new class is imported
   and appears in `python -c "from pooling_methods import POOLING_METHODS"`.
3. Run `./run_eval.sh --method <ClassName>` on CUDA device 1. The evaluator loads only `data/features_unpooled/*_dinov3_features.h5` through anonymized keys,
   runs a deterministic linear probe on all five datasets, and appends
   `results.json` and `progress.csv`.
4. Parse the JSON result. A method is a new best only when its
   `mean_accuracy` is strictly higher than the tracked best. Commit the new
   method and its result only for a new best; for an inferior result or an
   evaluation error, discard the experiment with `git reset --hard` to the
   pre-experiment commit. Never reset unrelated user changes.
5. `evaluate.py` sends the result through the persistent webhook configuration
   when configured,
   highlighting `NEW BEST` in green and `DISCARDED` in red. Webhook failure is
   an evaluation error and must be surfaced.

Run `./run_eval.sh --force-baselines` before the first search to establish the
seven anchor rows. Do not modify the feature files, anonymization map, probe
split, or evaluation metric to improve a method.
