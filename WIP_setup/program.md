# autoresearch

This is an experiment to have the LLM do its own research.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: Propose a tag based on today's date (e.g. `jun4`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from the current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:

   * `README.md` — repository context.
   * `prepare.py` — **CRITICAL**: Contains fixed constants, dataset configurations, image-to-mask transforms, and the locked evaluation standard. **Do not modify this file.**
   * `train.py` — **This is the file you modify.** Contains the dual-head U-Net model architecture, optimizer configurations, loss functions, and training loop logic.
4. **Verify data exists**: Check that `content/data/xview2_jpeg/` contains the satellite imagery folders (`tier1` and `test`). If not, tell the human to run `uv run prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 15 minutes** (wall clock training time, excluding startup/compilation). You launch it simply as:

```bash
uv run train.py
```

**What you CAN do:**

* Modify `train.py` — this is the only file you edit. Almost everything inside is fair game: model layers (encoder adjustments, skip connection configurations, dual-head layer decoding structures), optimizer choices, hyperparameters, loss function balances (e.g., BCE + Dice or Focal weights), batch size, etc.

**What you CANNOT do:**

* Modify `prepare.py`. It is read-only. It contains the data loading streaming frameworks, structural imagery transformations, channel normalization states, and the immutable evaluation benchmark.
* Install new packages or add dependencies. You can only use what's already inside `pyproject.toml`.
* Modify the evaluation harness. The `evaluate_xview2` function in `prepare.py` is the ground truth metric.

### Optimization Goals & Metric Traps

Your ground truth target metric is a composite score output by `evaluate_xview2`. Inside the evaluation harness, the model is scored on:

* **Intersection over Union (IoU) Score**
* **Dice Coefficient (F1 Score)**
* **Pixel Classification Accuracy**

The metric combines building localization (identifying where buildings are using pre-disaster data) and damage classification (grading the type of damage from 0 to 4 using post-disaster data).

**The goal is simple: get the highest final validation score.** Because the dual heads must solve two tasks simultaneously, you must find the best parameters for both. If an architectural modification or loss scaling adjustment improves localization metrics (IoU/Dice) but worsens damage classification accuracy (or vice versa), your job is to find a careful algorithmic balance that optimizes the unified macro-averaged framework.

> **Crucial Rule**: You must never alter or patch the evaluation metric calculation code. The ground truth evaluation must remain pristine and un-hacked.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful score gains, but it should not blow up dramatically or cause Out-of-Memory (OOM) faults.

**Simplicity criterion**: All else being equal, simpler is better. A small metric improvement that adds ugly, highly complex block logic is not worth it. Conversely, removing redundant layers or unneeded transforms and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 validation score improvement that adds 20 lines of hacky code? Probably not worth it. An improvement of ~0 but much simpler code? Keep.

**Improvement threshold:** Tiny score changes are often noise. A candidate experiment should generally only be considered a meaningful improvement if it exceeds the current best score by a small margin (for example, ~0.002). Improvements smaller than this should be treated cautiously unless they are repeatable.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script exactly as is.

## Output format

Once the script finishes it prints a summary like this:

```text
final_xview2_score: 0.684200
loc_f1_score:       0.821000
dmg_macro_f1:       0.625600
training_seconds:   900.2
total_seconds:      915.4
peak_vram_mb:       11240.5
num_steps:          1420
num_params_M:       24.3
```

Note that the script is configured to always stop after 15 minutes, so depending on the computing platform of this computer the numbers might look different. You can extract the key metric from the log file:

```bash
grep "^final_xview2_score:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```text
commit   val_score   memory_gb   status   description
```

1. Git commit hash (short, 7 chars)
2. Final composite `val_score` achieved (e.g. 0.684200) — use 0.000000 for crashes
3. Peak memory in GB, round to .1f (e.g. 11.0 — divide peak_vram_mb by 1024) — use 0.0 for crashes
4. Status: `keep`, `discard`, or `crash`
5. Short text description of what this experiment tried

Example:

```text
commit   val_score   memory_gb   status   description
a1b2c3d   0.612000   11.0        keep     baseline
b2c3d4e   0.634500   11.0        keep     switch building loss from BCE to Dice
c3d4e5f   0.601000   11.5        discard  increase encoder depth to ResNet50
d4e5f6g   0.000000   0.0         crash    double batch size to 32 (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/jun4`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on.
2. Tune `train.py` with an experimental computer vision idea by directly adjusting the network or optimization hyperparameters.
3. Git commit.
4. Run the experiment:

```bash
uv run train.py > run.log 2>&1
```

(redirect everything — do NOT use tee or let output flood your context).

5. Read out the results:

```bash
grep "^final_xview2_score:\|^peak_vram_mb:" run.log
```

6. If the grep output is empty, the run crashed. Run:

```bash
tail -n 50 run.log
```

to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up on that idea.

7. Record the results in the TSV (NOTE: do not commit the `results.tsv` file, leave it untracked by git).
8. If the validation score improved (higher score), you "advance" the branch, keeping the git commit.
9. If the score is equal or worse, you git reset back to where you started before this experiment step.

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate continuously. If you feel like you're getting stuck in some way, you can rewind but you should do this very sparingly (if ever).

**Timeout**: Each experiment should take ~15 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 25 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, a shape mismatch error, or a runtime bug), use your judgment: If it's something fast and easy to fix (e.g. a syntax typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken (like causing an explicit tensor size mismatch across U-Net skip connections), just skip it, log "crash" as the status in the TSV, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or away from the computer, and expects you to continue working indefinitely until you are manually stopped. You are autonomous. If you run out of ideas, think harder — review known image segmentation paradigms, re-read the model structural inputs for new scaling angles, or combine previous near-miss ideas. The loop runs until the human manually interrupts you, period.
