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
  a local-file fallback for each. Full data, proper budgets, held-out test splits, literature metrics
  (acc / R2 / ROC-AUC). A GPU is recommended for the 3D and set-attention tasks.
    pip install torch numpy medmnist deepchem rdkit jetnet aeon
    python run_standard_validation.py                          # every modality
    python run_standard_validation.py --only graph,equivariant
    python run_standard_validation.py --graph_dataset Tox21    # ESOL/Tox21/BBBP/BACE/FreeSolv/Lipo
    python run_standard_validation.py --device cuda --set_attention   # SAB/ISAB on GPU
    python run_standard_validation.py --device cuda --epochs_scale 2.0

  Per-modality dataset knobs: --seq_dataset, --graph_dataset, --rmd17/--rmd17_max, --jetnet_dir/
  --jetnet_per_class, --bloodmnist, --organmnist3d. --epochs_scale multiplies all budgets;
  --set_attention adds the O(N^2) attention set primitives (GPU recommended).
