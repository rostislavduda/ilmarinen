# Experiment runners (bypass AllGraph)

These are research / ablation drivers that build schemas directly or exercise individual machinery
pieces, rather than going through the `AllGraph` controller (the primary user interface). They are kept
for reproducibility of specific experiments but are not part of the maintained validation flow.

Maintained runners live one level up in `validation_runners/`:
`run_standard_validation.py`, `run_quick_validation.py`, `run_cellpainting_validation.py`,
`record_contract_corpus.py`.

Run any experiment script from the repo root, e.g. `python validation_runners/experiments/run_frontier.py`.
