# Validation runners

Reproducible scripts to run the ilmarinen schemas across datasets and report the selected
architecture and accuracy. Run them from the package root (they add it to sys.path).

Requires: `torch`, `numpy`, and `aeon` (for the UCR/UEA time-series datasets).

## Scripts

### `run_unified_synthetic.py` -- schema on synthetic tasks (known ground truth)
```
python validation_runners/experiments/run_unified_synthetic.py --task recall
python validation_runners/experiments/run_unified_synthetic.py --task adding --T 80
python validation_runners/experiments/run_unified_synthetic.py --task copy  --delay 15
```
Reports the architecture (per-layer selected primitive) + performance. Tasks: `copy`
(recurrent memory), `adding` (long-range; attention/gated), `recall` (attention routing).

### `run_unified_ucr.py` -- schema on a real UCR/UEA dataset
```
python validation_runners/experiments/run_unified_ucr.py --dataset GunPoint
python validation_runners/experiments/run_unified_ucr.py --dataset ACSF1 --max_t 120 --epochs 25
python validation_runners/experiments/run_unified_ucr.py --dataset ItalyPowerDemand --bilevel
python validation_runners/experiments/run_unified_ucr.py --dataset BasicMotions --primitives plain,gated,lstm,conv,attention,dense,norm,spectral
```
Reports TEST accuracy (official split), majority baseline, selected architecture, and the
full per-layer alpha. Useful flags: `--depth`, `--width`, `--readout {mean,last}`,
`--primitives` (subset), `--max_t` (time-axis pooling for long series), `--chrono_tmax`
(LSTM chrono-init), `--bilevel` (see protocol below).

### `run_baselines_ucr.py` -- matched-complexity fixed baselines
```
python validation_runners/experiments/run_baselines_ucr.py --dataset GunPoint
```
Trains a plain GRU, a 2-layer 1-D CNN, and a 1-hidden-layer MLP at comparable parameter
count with the SAME preprocessing/split/budget, reporting TEST accuracy + param count for
each. This is the honest "NN of similar complexity" reference for the schema.

## Architecture-SIZE metaoptimization (width and depth as OUTPUTS)

The runners above fix width and depth; these three make them decisions of the metaoptimality
criterion, realizing the minimal-representation idea (width/depth chosen, not hyperparameters).

### `run_minimal_architecture.py` -- select primitive + width + depth at a given price
```
python validation_runners/experiments/run_minimal_architecture.py --dataset GunPoint --width_mu 0.002 --depth_mu 0.05
```
Sweeps widths and depths; selects the minimal width/depth whose marginal validation-loss
reduction per unit of added capacity exceeds the price (`--width_mu`, `--depth_mu`). With price
0 it reduces to the significant-elbow (add capacity while it demonstrably pays). Reports the
selected (primitive, width, depth) and TEST accuracy. Bilevel by construction.

### `run_frontier.py` -- trace the fit-vs-complexity Pareto frontier
```
python validation_runners/experiments/run_frontier.py --dataset GunPoint --mus 0.001,0.003,0.008,0.02
```
Sweeps the capacity price `--mus` and reports, per price, the selected (width, depth,
primitive), parameter count, and TEST accuracy -- the frontier of metaoptimal architectures.
As the price rises the selected model shrinks and accuracy drops: the honest "minimal
representation" answer is the frontier, not a single point. Depths restricted to {1,2} (no task
justifies deeper; see tests/depth_necessity_probe.md).

### `run_penalized_selection.py` -- compact-AND-accurate (differentiable complexity penalty)
```
python validation_runners/experiments/run_penalized_selection.py --dataset GunPoint --mu 0.3
```
Folds the complexity price directly into the differentiable architecture objective (following
the resource-aware differentiable-NAS literature, FBNet / SA-DARTS): the alpha-loss gains
`mu * sum_i softmax(alpha)_i * cost_i`, pulling selection toward cheap primitives DURING search,
plus an entropy-sharpening term (`--gamma`). Width uses accuracy-first compaction (smallest
width within `--acc_tol` of the best validation accuracy), which preserves the high-accuracy
end of the frontier that the marginal-threshold rule loses. This is the recommended runner for
obtaining models that are both small and accurate; `--mu` trades size against accuracy inside a
single objective aligned with the analytical action J = R + mu * complexity.

## Evaluation protocol (important)

- **Reported accuracy is TEST accuracy** on the official held-out split. The model never
  trains on the test data. Training uses only the train split; the reported number is
  `net(X_test)` accuracy.

- **Weights vs alpha (architecture) selection.** By default (`--bilevel` off) the network
  weights AND the architecture parameter alpha are both trained on the train split, and the
  test split is used only for the final reported accuracy. This is fast and the test number
  is legitimate, but the *architecture choice* was made on the same data the weights saw.

- **`--bilevel`** holds out a validation fraction (`--val_frac`, default 0.3) FROM the train
  split, trains the weights on the remaining train data and alpha on the held-out validation
  data. This is the honest architecture-selection protocol: alpha is chosen on data the
  weights did not fit, avoiding capacity-chasing. Use it when the architecture *selection*
  (not just the accuracy) needs to be defensible. The test split is untouched either way.

- **Long series** are average-pooled on the time axis to `--max_t` (default 120) so the
  pure-Python per-timestep recurrent scan stays tractable; the recurrent primitives are the
  performance bottleneck at large T. Set `--max_t 0` to disable pooling (slow for long T).

## Reproducing the checkpoint results

Synthetic (expect copy->recurrent, adding->attention, recall->attention):
```
for t in copy adding recall; do python validation_runners/experiments/run_unified_synthetic.py --task $t; done
```
Real UCR/UEA (expect recurrent family to win at this scale; see tests/ucr_unified_sweep.md):
```
for d in ItalyPowerDemand GunPoint OSULeaf BasicMotions ACSF1; do
  python validation_runners/experiments/run_unified_ucr.py --dataset $d
  python validation_runners/experiments/run_baselines_ucr.py --dataset $d
done
```
Numbers vary slightly by machine/seed/epoch budget; the selected architectures are stable.
Note (from tests/matched_complexity_comparison.md): the schema wins or ties matched-
complexity baselines on 4/5 datasets; it loses on ItalyPowerDemand because that task is
"effectively tabular" and the flatten-then-dense MLP -- a realization the mean-pooled readout
cannot currently express -- is optimal there.
```

## Standard validation across all 7 modalities (added)

Two harnesses validate the whole framework (every schema) on real natural-science/medical data:

- **run_quick_validation.py** -- FAST sanity check on SUBSETS of data already in this environment
  (tiny budgets, a few minutes total). Rough performance picture, not a benchmark. Datasets: ECG5000
  (seq), BloodMNIST (spatial), OrganMNIST3D (vol), ESOL (graph), rMD17 (equivariant), JetNet (set).
    python run_quick_validation.py                    # all
    python run_quick_validation.py --only graph,set   # subset
    python run_quick_validation.py --skip volumetric  # 3D conv is slowest

- **run_standard_validation.py** -- FULL validation for your own machine. Uses the official dataset
  PACKAGES where available (medmnist, deepchem+rdkit, jetnet, aeon) so nothing is hand-downloaded, with
  a local-file fallback for each. Every dataset in the registry at full size, held-out test splits,
  literature metrics (acc / R2 / ROC-AUC). A GPU is recommended for the 3D and set tasks.
    pip install torch numpy medmnist deepchem rdkit jetnet aeon
    python -m validation_runners.run_standard_validation                       # every dataset
    python -m validation_runners.run_standard_validation --contracts graph,equivariant
    python -m validation_runners.run_standard_validation --only ESOL,Tox21     # by DATASET name
    python -m validation_runners.run_standard_validation --device cuda --epochs_scale 2.0

  Dataset selection: --only / --skip take comma-separated DATASET names; --contracts takes contracts
  (sequence, spatial, volumetric, 4d, graph, equivariant, set, operator). There is no --epochs flag:
  each contract's budget lives in the BUDGET table at the top of the runner and --epochs_scale
  multiplies it.

  TRAINING TO CONVERGENCE. --auto_epoch {train,val} early-stops a deployed model on a plateau, and
  --auto_epoch_extend N turns the budget from a ceiling into a first block: a model that uses its whole
  budget WITHOUT plateauing is granted another N epochs and continues from the same weights and
  optimizer state, repeating until it plateaus or --auto_epoch_max is reached (that ceiling matters --
  a model that never leaves its initial loss plateau can never early-stop). --auto_epoch_restore_best
  then returns the best-scoring epoch's weights rather than the last. Each result records
  epochs_trained / converged, so a budget-limited number is distinguishable from a converged one.

  RESULTS. The runner accumulates rows into one merge-on-write JSON
  (--results_out, default <data-dir>/standard_val_rows.json), upserting by dataset name and flushing
  after every dataset -- so the suite can be run in batches and an interrupted run loses nothing.
  --save_models writes each fitted model to the package out/ folder. Render the table with:
    python -m validation_runners.make_results_table --insert-readme

- **make_results_table.py** -- renders that results JSON as the README's markdown table, grouped by
  contract. --insert-readme splices it between the <!-- BEGIN:stdval --> / <!-- END:stdval --> markers
  so the published table stays regenerable.

## Bring your own dataset: run_kaggle_validation.py

Every runner above trains on a CURATED dataset -- a registry entry whose loader already knows its target
column, its split, and its SOTA reference. **run_kaggle_validation.py** is the opposite: it takes an
arbitrary Kaggle handle, does the ingestion itself, and hands the result to the SAME pipeline (it imports
add_pipeline_args / resolve_pipeline / apply_opt_preset / make_allgraph / _eval_test / record_row from
run_standard_validation), so a Kaggle run is configured identically to a benchmark run and takes every one
of the shared flags.

    pip install "ilmarinen[kaggle]"        # kagglehub + pandas + pillow (none is needed by the rest of the
                                           # package; all three are imported lazily, per mode)

Credentials: create an API token at https://www.kaggle.com/settings/api, then either save kaggle.json to
~/.kaggle/kaggle.json or export KAGGLE_USERNAME/KAGGLE_KEY. Downloads are cached under
$ILMARINEN_DATA_DIR/kagglehub (--cache_dir "" restores kagglehub's own ~/.cache/kagglehub).

    python validation_runners/run_kaggle_validation.py --check_auth

ALWAYS --inspect FIRST. For an unfamiliar handle you do not know the file inventory, the column names, or
which column is the target, and every guess the script makes is a heuristic. --inspect prints the file tree,
the detected mode, a per-column profile with the verdict the encoder will apply, the target and task guesses,
the split, the chance baseline, the routed contract, and the budget -- and trains nothing.

    python validation_runners/run_kaggle_validation.py --handle uciml/iris --inspect
    python validation_runners/run_kaggle_validation.py --handle uciml/iris --epochs 40

THREE MODES, auto-detected (override with --mode):
  tabular  a CSV/TSV/parquet -> numeric + one-hot matrix -> rank 2, which routes to the SEQUENCE contract
  images   a class-directory tree (or train/test dirs, or a flat dir + labels CSV) -> (N,C,hw,hw) -> SPATIAL
  npy      a .npy/.npz -> rank passed straight to the grid-rank router (2=flat .. 6=4d)

    # tabular regression: MAE comes back in the original units via target_scale
    python validation_runners/run_kaggle_validation.py --handle camnugent/california-housing-prices \
        --target median_house_value --task regression --target_units USD
    # a COMPETITION takes a bare slug, and pick_table skips gender_submission.csv for you
    python validation_runners/run_kaggle_validation.py --source competition --handle titanic \
        --table train.csv --target Survived --drop_cols Name,Ticket,Cabin
    # images -> the spatial contract
    python validation_runners/run_kaggle_validation.py --handle tongpython/cat-and-dog \
        --mode images --hw 64 --per_class 400 --auto_epoch val
    # offline / air-gapped: skip kagglehub entirely
    python validation_runners/run_kaggle_validation.py --local_dir /path/to/extracted --inspect

BUDGET. Unlike the standard runner this one DOES take --width/--depth/--epochs, because there is no tuned
budget for a dataset nobody has seen. Left unset, it routes the data without training (a probe on
AllGraph.route) and takes that contract's entry from the same BUDGET table; --epochs_scale still multiplies.

HONESTY. A flat tabular feature vector routes to the sequence contract and is read as (n, T=n_features, C=1);
ilmarinen is a meta-optimizer over contracts, not a tabular specialist, so **expect to lose to XGBoost on
tabular Kaggle data** -- the point is watching routing and priced selection behave on untuned data. `chance`
is the TRAIN-majority predictor scored on the TEST split (not 1/K), so read `skill`, not accuracy, on an
imbalanced target. The split is ours, not the competition's hidden test set, so this is not a leaderboard
score. Every statistic is fit on train rows only. The module docstring lists all ten caveats.

RESULTS go to a SEPARATE document, <data-dir>/kaggle_val_rows.json. Do NOT feed it to
`make_results_table --insert-readme`: that renders standard_val_rows.json into the published benchmark table
in README.md, and an arbitrary Kaggle dataset has no business appearing there. Rows upsert by name (--name),
so repeated invocations accumulate; each row carries a `kaggle` sub-dict recording the handle, mode, target,
dropped columns, split sizes and route, so it is self-describing without a registry entry to look up.
