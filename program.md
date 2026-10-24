# Autonomous VFM Pooling Search

You are searching for generalizable pooling functions over anonymized DINOv3
patch features. Dataset names are intentionally opaque: do not infer domains
from `Dataset_1` through `Dataset_5`, and never inspect or use image paths.

---

## 🎯 Search Philosophy & Strategy Guidelines

1. **Aim for Structural & Conceptual Innovation (No Trivial Tweaks):**
   - **Do NOT** perform minor hyperparameter fine-tuning iterations (e.g., changing a temperature parameter from `0.5` to `0.7`, or tweaking learning rates/decay rates).
   - Implement **configurably simple** classes that test distinct, principled mathematical hypotheses about patch token interactions (e.g., spatial geometry, feature variance/kurtosis, dynamic attention heads, manifold projection, rank-based filtering, topological distance).

2. **Exploit Winning Threads (Iterative Evolution):**
   - When a new idea achieves a **`NEW BEST`**, analyze *why* it succeeded.
   - Continue developing that conceptual thread by adding meaningful architectural improvements, complementary operations, or hybrid mechanisms until performance plateaus.

3. **Pivot Intelligently on Plateaus:**
   - If a line of research fails or yields diminishing returns after 2–3 variations, **do not get stuck in a local minimum**.
   - Pivot cleanly: return to the last high-performing baseline/commit or branch out into a completely novel mathematical direction.

4. **Be Scientifically Creative:**
   - Draw inspiration from diverse fields: differential geometry, information theory, robust statistics, kernel methods, optimal transport, and signal processing.
   - Every experiment must test a clear, novel hypothesis stated in the class docstring.

---

## 🔄 Required Experiment Loop

1. **Formulate Hypothesis & Implement Code:**
   - Create exactly one new `BasePoolingMethod` subclass in `pooling_methods/<method_name>.py`.
   - Include a class docstring describing the mathematical hypothesis and mechanics.
   - Implement `pool`; use `fit` only for state that is learned from the current dataset's unpooled features.

2. **Verify Dynamic Import:**
   - Verify that the new class is imported and appears in:
     `python -c "from pooling_methods import POOLING_METHODS"`

3. **Execute Evaluation:**
   - Run `./run_eval.sh --method <ClassName>` on CUDA device 1.
   - The evaluator loads only `data/features_unpooled/*_dinov3_features.h5` through anonymized keys, runs a deterministic linear probe on all five datasets, and appends `results.json` and `progress.csv`.

4. **Parse & Decide (Commit vs. Revert):**
   - Parse the JSON result. A method is a new best only when its `mean_accuracy` is strictly higher than the tracked best.
   - **New Best:** Commit the new method and its result (`git commit -am "idea: <description>"`). Update your mental state to explore deeper along this branch.
   - **Inferior or Crash:** Discard the experiment completely using `git reset --hard` to return to the pre-experiment commit. Never reset unrelated user changes.

5. **Notification:**
   - `evaluate.py` sends the result through the persistent webhook configuration when configured, highlighting `NEW BEST` in green and `DISCARDED` in red. Webhook failure is an evaluation error and must be surfaced.

---

## 🚀 Warm-Up Requirement

Run `./run_eval.sh --force-baselines` before the first search to establish the seven anchor rows. Do not modify the feature files, anonymization map, probe split, or evaluation metric to improve a method.