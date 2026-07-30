"""T-EV: `_eval_test` must evaluate in EVAL mode, so a reported score cannot depend on test-split ordering.

`validation_runners/run_standard_validation._eval_test` is the single path that produces every number this
project publishes -- the benchmark table, the quick smoke, and the Kaggle runner all call this one function.
It used to wrap its forwards in `torch.no_grad()` but never call `net.eval()`, which is NOT the same thing:

  * Several schemas carry BatchNorm (the `norm` primitive; the spatial/volumetric stems). Left in TRAIN mode,
    BatchNorm normalizes each eval minibatch by that minibatch's OWN statistics rather than the running
    averages, so the score becomes a function of how the test split happens to be ordered. On a class-ordered
    split -- UCR splits, MedMNIST, and every class-directory image tree -- a 256-sample batch is nearly one
    class, and BatchNorm centres away exactly the between-class signal being classified. Measured on a
    4-class MRI model: 0.4800 class-ordered vs 0.8656 in eval mode, on identical weights.
  * A train-mode forward also UPDATES BatchNorm's running buffers, even under `no_grad` (the update is a
    buffer write, not an autograd op) -- so merely evaluating leaked test statistics into the model.

`AllGraph._forward_new` (behind predict/load) has always called `net.eval()`, so the runner and `predict`
disagreed on the same weights. These tests pin the invariant from the outside: a score must be invariant to
test-split permutation, and evaluating must not mutate the model.
"""

import os
import sys

import numpy as np
import pytest
import torch

from ilmarinen import AllData, AllGraph

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "validation_runners"))

from run_standard_validation import _eval_test  # noqa: E402


def _spatial_task(n=96, hw=8, seed=0):
    """Two classes separated by overall intensity -- learnable in a couple of epochs, and a signal BatchNorm
    demonstrably destroys when a batch holds only one class."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 1, hw, hw).astype("float32")
    y = (np.arange(n) % 2).astype("int64")
    X[y == 1] += 2.0
    return X, y


def _fit_small():
    mg = AllGraph(width=8, depth=1, epochs=2, device="cpu", verbose=False, seed=0)
    X, y = _spatial_task()
    mg.fit(AllData.dense_tensor(X, y, kind_hint="spatial"), task="classification", n_out=2)
    return mg


def _class_ordered_test(hw=8, per_class=64, seed=1):
    """A test split laid out one class after another -- what a class-directory image tree produces, and what
    several registry loaders produce too."""
    X, y = _spatial_task(n=2 * per_class, hw=hw, seed=seed)
    order = np.argsort(y, kind="stable")
    return X[order], y[order]


@pytest.mark.smoke
class TestEvalTestIsOrderInvariant:
    def test_score_does_not_depend_on_test_split_ordering(self):
        """T-EV-1: the load-bearing one. Same weights, same samples, different row order -> same score.

        With the bug, the class-ordered arrangement scored far below the shuffled one because each
        BatchNorm batch saw a single class.
        """
        mg = _fit_small()
        X, y = _class_ordered_test()
        perm = np.random.RandomState(0).permutation(len(y))

        _, ordered, _ = _eval_test(mg, AllData.dense_tensor(X, y, kind_hint="spatial"), "classification", dense_bs=32)
        _, shuffled, _ = _eval_test(
            mg, AllData.dense_tensor(X[perm], y[perm], kind_hint="spatial"), "classification", dense_bs=32
        )
        assert ordered == pytest.approx(shuffled, abs=1e-9), (
            f"class-ordered {ordered:.4f} != shuffled {shuffled:.4f} -- _eval_test is batch-order dependent, "
            f"which means a BatchNorm module was left in train mode"
        )

    def test_score_does_not_depend_on_eval_batch_size(self):
        """T-EV-2: the same invariant seen from the other side -- batching is an implementation detail, so
        the batch size must not move the number."""
        mg = _fit_small()
        X, y = _class_ordered_test()
        data = AllData.dense_tensor(X, y, kind_hint="spatial")
        scores = {bs: _eval_test(mg, data, "classification", dense_bs=bs)[1] for bs in (16, 64, 1000)}
        assert len(set(scores.values())) == 1, f"score varies with eval batch size: {scores}"

    def test_evaluating_does_not_mutate_the_model(self):
        """T-EV-3: a train-mode forward updates BatchNorm's running buffers even under no_grad, so merely
        scoring the test split used to leak test statistics into the deployed weights."""
        mg = _fit_small()
        before = {k: v.clone() for k, v in mg.net.state_dict().items()}
        X, y = _class_ordered_test()
        _eval_test(mg, AllData.dense_tensor(X, y, kind_hint="spatial"), "classification", dense_bs=32)
        after = mg.net.state_dict()
        drifted = [k for k in before if not torch.equal(before[k], after[k])]
        assert not drifted, f"evaluating changed these buffers/parameters: {drifted}"

    def test_leaves_the_net_in_eval_mode(self):
        """T-EV-4: cheap direct check of the mechanism the three invariants above depend on."""
        mg = _fit_small()
        X, y = _class_ordered_test()
        _eval_test(mg, AllData.dense_tensor(X, y, kind_hint="spatial"), "classification", dense_bs=32)
        assert not mg.net.training

    def test_agrees_with_predict_on_the_same_weights(self):
        """T-EV-5: the runner's eval path and the library's inference path (`_forward_new`, behind predict()
        and AllGraph.load) must not disagree about the same model -- that disagreement is what exposed this."""
        mg = _fit_small()
        X, y = _class_ordered_test()
        _, score, _ = _eval_test(mg, AllData.dense_tensor(X, y, kind_hint="spatial"), "classification", dense_bs=32)
        pred = mg.predict(AllData.dense_tensor(X, kind_hint="spatial"))
        assert score == pytest.approx(float((pred == y).mean()), abs=1e-9)
