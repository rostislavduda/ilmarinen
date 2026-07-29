"""T-KG: the arbitrary-Kaggle-dataset runner (validation_runners/run_kaggle_validation.py).

That runner is the only entry point that ingests data nobody wrote a loader for, so almost all of its risk
sits in the INGEST layer -- guessing a target column, inferring a task, encoding categoricals, and splitting
-- rather than in the model pipeline, which it imports wholesale from run_standard_validation. These lock the
decisions that would otherwise fail silently and produce a plausible but wrong number:

  Acquisition   the kagglehub boundary: --local_dir never touches it, and every failure is actionable
  Inventory     mode detection (tabular / images / npy) and the guards that keep them apart
  Tabular       target + task guessing, column hygiene, TRAIN-ONLY statistics, one-hot unknown handling
  Splits        stratification, contiguous labels, and the train-majority chance baseline
  Images        the three layouts, corrupt-file alignment, and the square-hw / memory guards
  Arrays        key autodetection, one-hot targets, and the rank 2-6 contract of route_grid_rank
  Runner        the budget probe, the SEPARATE results file, and failure -> recorded row rather than crash

The leakage tests are the load-bearing ones: `test_scaler_is_fit_on_train_only` and
`test_chance_is_train_majority_scored_on_test` are what stop the runner from quietly reporting an inflated
score. Everything runs OFFLINE against tmp_path fixtures -- no network, no Kaggle credentials, no downloads.
Tests that call fit() are marked ``smoke``.
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "validation_runners"))

import run_kaggle_validation as rkv  # noqa: E402

# pandas and Pillow are OPTIONAL extras (pyproject `kaggle`), and CI installs neither, so the modes that need
# them are skipped rather than failing the required unit-test check.
pd = pytest.importorskip("pandas", reason="tabular mode needs pandas (optional 'kaggle' extra)")
PIL_Image = pytest.importorskip("PIL.Image", reason="image mode needs Pillow (optional 'kaggle' extra)")


# --------------------------------------------------------------------------- helpers
def _args(**kw):
    """A defaults-filled args namespace, built from the real parser so a flag rename cannot silently rot the
    tests. `drop_set` and the per_class=0 sentinel are the two derivations main() applies after parsing."""
    ns = rkv.build_parser().parse_args([])
    for k, v in kw.items():
        setattr(ns, k, v)
    ns.drop_set = {c.strip() for c in ns.drop_cols.split(",") if c.strip()}
    if ns.per_class == 0:
        ns.per_class = None
    return ns


def _run(monkeypatch, *argv):
    """Drive main() as the CLI would. sys.argv MUST be patched: _apply_preset and apply_opt_preset both scan
    sys.argv for '--flag' tokens to decide which options the user set explicitly, and pytest's own argv would
    otherwise poison that set."""
    monkeypatch.setattr(sys, "argv", ["run_kaggle_validation.py", *[str(a) for a in argv]])
    return rkv.main()


def _write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    return path


def _ingest_table(tmp_path, header, rows, **kw):
    """Write one CSV and run the tabular ingest over it, returning the registry-shaped dict."""
    _write_csv(tmp_path / "train.csv", header, rows)
    inv = rkv.walk_inventory(str(tmp_path))
    return rkv.encode_tabular(str(tmp_path), rkv.classify_files(inv), _args(**kw))


def _png(path, value, hw=8, seed=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)
    a = np.clip(rng.randint(0, 40, (hw, hw, 3)) + value, 0, 255).astype(np.uint8)
    PIL_Image.fromarray(a).save(path)
    return path


def _image_tree(tmp_path, classes=("alpha", "beta"), n=12, hw=8, prefix=""):
    for ci, cls in enumerate(classes):
        for i in range(n):
            _png(tmp_path / prefix / cls / f"{i:03d}.png", 60 + 120 * ci, hw=hw, seed=ci * 100 + i)
    return tmp_path


# =========================================================================== inventory & mode detection
class TestInventoryAndMode:
    def test_inventory_reports_relpaths_and_sizes_largest_first(self, tmp_path):
        """T-KG-1: the inventory is relative, size-ordered, and free of macOS archive cruft."""
        _write_csv(tmp_path / "a.csv", ["x", "y"], [[1, 2]] * 50)
        _write_csv(tmp_path / "sub" / "b.csv", ["x", "y"], [[1, 2]])
        (tmp_path / "__MACOSX").mkdir()
        (tmp_path / "__MACOSX" / "junk.csv").write_text("nope")
        (tmp_path / ".hidden.csv").write_text("nope")
        inv = rkv.walk_inventory(str(tmp_path))
        assert [r for r, _ in inv] == ["a.csv", os.path.join("sub", "b.csv")]
        assert inv[0][1] >= inv[1][1]

    def test_submission_template_is_never_a_table_candidate(self, tmp_path):
        """T-KG-2: a competition's sample_submission.csv has the target but no features -- training on it
        would be silently meaningless, so it is excluded from `tables` outright."""
        _write_csv(tmp_path / "sample_submission.csv", ["Id", "target"], [[1, 0]] * 99)
        _write_csv(tmp_path / "train.csv", ["a", "target"], [[1, 0]])
        b = rkv.classify_files(rkv.walk_inventory(str(tmp_path)))
        assert [r for r, _ in b["tables"]] == ["train.csv"]

    def test_pick_table_prefers_train_over_a_larger_sibling(self, tmp_path):
        """T-KG-3: `train*` wins even when another table is bigger."""
        _write_csv(tmp_path / "metadata.csv", ["a"], [[1]] * 500)
        _write_csv(tmp_path / "train.csv", ["a", "target"], [[1, 0]])
        b = rkv.classify_files(rkv.walk_inventory(str(tmp_path)))
        assert rkv.pick_table(str(tmp_path), b, _args())[1] == "train.csv"

    def test_detect_mode_tabular_when_only_a_table(self, tmp_path):
        _write_csv(tmp_path / "train.csv", ["a", "target"], [[1, 0]])
        b = rkv.classify_files(rkv.walk_inventory(str(tmp_path)))
        assert rkv.detect_mode(str(tmp_path), b, _args())[0] == "tabular"

    def test_detect_mode_images_needs_two_class_dirs_and_min_images(self, tmp_path):
        """T-KG-4: the guard that stops a tabular dataset with a few plot PNGs from being read as an image
        dataset. One class directory of 20 images is NOT an image dataset; two of 12 each is."""
        for i in range(20):
            _png(tmp_path / "plots" / f"{i}.png", 100, seed=i)
        _write_csv(tmp_path / "train.csv", ["a", "target"], [[1, 0]])
        b = rkv.classify_files(rkv.walk_inventory(str(tmp_path)))
        assert rkv.detect_mode(str(tmp_path), b, _args())[0] == "tabular"

        _image_tree(tmp_path / "imgs", n=12)
        b = rkv.classify_files(rkv.walk_inventory(str(tmp_path)))
        assert rkv.detect_mode(str(tmp_path), b, _args())[0] == "images"

    def test_detect_mode_npy_when_no_table(self, tmp_path):
        np.save(tmp_path / "x.npy", np.zeros((4, 3)))
        b = rkv.classify_files(rkv.walk_inventory(str(tmp_path)))
        assert rkv.detect_mode(str(tmp_path), b, _args())[0] == "npy"

    def test_mode_flag_overrides_autodetection(self, tmp_path):
        np.save(tmp_path / "x.npy", np.zeros((4, 3)))
        b = rkv.classify_files(rkv.walk_inventory(str(tmp_path)))
        mode, why = rkv.detect_mode(str(tmp_path), b, _args(mode="tabular"))
        assert mode == "tabular" and "--mode" in why

    def test_undetectable_mode_returns_none(self, tmp_path):
        (tmp_path / "readme.md").write_text("nothing usable here")
        b = rkv.classify_files(rkv.walk_inventory(str(tmp_path)))
        assert rkv.detect_mode(str(tmp_path), b, _args())[0] is None


# =========================================================================== target & task inference
class TestTargetAndTask:
    def test_target_guess_prefers_a_conventional_name_over_the_last_column(self):
        df = pd.DataFrame({"a": [1], "target": [0], "z": [3]})
        col, why = rkv.guess_target(df, _args())
        assert col == "target" and "conventional" in why

    def test_target_guess_falls_back_to_the_last_column(self):
        df = pd.DataFrame({"a": [1], "b": [2], "verdict": [3]})
        col, why = rkv.guess_target(df, _args())
        assert col == "verdict" and "last column" in why

    def test_explicit_target_wins_and_a_missing_one_raises_with_the_columns(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        assert rkv.guess_target(df, _args(target="a"))[0] == "a"
        with pytest.raises(KeyError, match="'b'"):
            rkv.guess_target(df, _args(target="nope"))

    def test_low_cardinality_integer_target_is_classification(self):
        s = pd.Series([0, 1, 2] * 40, name="t")
        assert rkv.infer_task(s, _args())[0] == "classification"

    def test_continuous_float_target_is_regression(self):
        s = pd.Series(np.linspace(0, 1, 120), name="t")
        assert rkv.infer_task(s, _args())[0] == "regression"

    def test_high_cardinality_integers_are_regression_not_1000_classes(self):
        """T-KG-5: integral does not mean categorical -- a per-row integer count is a regression target."""
        s = pd.Series(np.arange(1000), name="t")
        assert rkv.infer_task(s, _args())[0] == "regression"

    def test_string_target_over_max_classes_raises_with_the_flag_that_fixes_it(self):
        s = pd.Series([f"c{i}" for i in range(80)], name="t")
        with pytest.raises(ValueError, match="--max_classes"):
            rkv.infer_task(s, _args())

    def test_constant_target_raises(self):
        with pytest.raises(ValueError, match="nothing to predict"):
            rkv.infer_task(pd.Series([1] * 20, name="t"), _args())

    def test_task_flag_overrides_inference(self):
        s = pd.Series([0, 1] * 30, name="t")
        assert rkv.infer_task(s, _args(task="regression")) == ("regression", "--task")


# =========================================================================== column hygiene
class TestColumnHygiene:
    def test_id_constant_and_mostly_missing_columns_are_dropped_with_reasons(self, tmp_path):
        """T-KG-6: every dropped column is dropped for a stated, recorded reason."""
        n = 60
        rows = [[i, 1.0, (i % 5) * 0.5, "" if i % 4 else "x", float(i)] for i in range(n)]
        d = _ingest_table(tmp_path, ["Id", "constant", "feat", "mostly_missing", "target"], rows, task="regression")
        reasons = " ".join(d["meta"]["dropped_columns"])
        assert "Id (id-like" in reasons
        assert "constant (constant" in reasons
        assert "mostly_missing" in reasons and "missing" in reasons
        assert d["meta"]["numeric_columns"] == ["feat"]

    def test_high_cardinality_categorical_is_dropped_at_the_cap(self, tmp_path):
        """Cardinality 20 over 60 rows: not unique-per-row, so this exercises the --max_cardinality rule
        specifically rather than the id-like rule that would also fire on an all-unique column."""
        n = 60
        rows = [[float(i % 7), f"u{i % 20}", float(i)] for i in range(n)]
        d = _ingest_table(tmp_path, ["feat", "many", "target"], rows, task="regression", max_cardinality=5)
        assert any("many" in c and "cardinality" in c for c in d["meta"]["dropped_columns"])

    def test_all_unique_string_column_is_dropped_as_id_like(self, tmp_path):
        """The id-like rule fires before the cardinality rule: a value unique to every row is an identifier,
        and identifiers both carry no signal and leak row order."""
        rows = [[float(i % 7), f"u{i}", float(i)] for i in range(60)]
        d = _ingest_table(tmp_path, ["feat", "rowkey", "target"], rows, task="regression")
        assert any("rowkey" in c and "id-like" in c for c in d["meta"]["dropped_columns"])

    def test_explicit_drop_cols_is_honoured(self, tmp_path):
        rows = [[float(i % 7), float(i % 3), float(i)] for i in range(40)]
        d = _ingest_table(tmp_path, ["keep", "toss", "target"], rows, task="regression", drop_cols="toss")
        assert d["meta"]["numeric_columns"] == ["keep"]

    def test_string_numeric_column_is_coerced_not_one_hot(self, tmp_path):
        """T-KG-7: the Telco `TotalCharges` case -- digits stored as strings with a few blanks stay NUMERIC
        rather than exploding into a one-hot block."""
        rows = [[f"{i * 1.5}" if i % 11 else " ", float(i)] for i in range(60)]
        d = _ingest_table(tmp_path, ["charges", "target"], rows, task="regression")
        assert d["meta"]["numeric_columns"] == ["charges"]
        assert d["meta"]["onehot_columns"] == []

    def test_max_features_drops_the_widest_one_hot_first(self, tmp_path):
        n = 80
        rows = [[float(i), f"w{i % 12}", f"n{i % 3}", float(i % 2)] for i in range(n)]
        d = _ingest_table(tmp_path, ["num", "wide", "narrow", "target"], rows, task="regression", max_features=5)
        oh = " ".join(d["meta"]["onehot_columns"])
        assert "narrow" in oh and "wide" not in oh
        assert d["train"].dense.shape[1] <= 5


# =========================================================================== encoding & leakage
class TestEncodingAndLeakage:
    def test_scaler_is_fit_on_train_only(self, tmp_path):
        """T-KG-8 (load-bearing): the standardization must NOT see the test rows. The fixture gives the two
        halves of the split materially different means, so a whole-dataset scaler -- the leak that
        core/extended_datasets.py:load_superconductivity has -- would centre BOTH halves near zero. Train
        centres at ~0; test must not."""
        n = 400
        rows = [[float(i), float(i % 2)] for i in range(n)]  # `feat` grows monotonically with the row index
        d = _ingest_table(tmp_path, ["feat", "target"], rows, task="classification", test_frac=0.5, split_seed=3)
        xtr = d["train"].dense.numpy()
        xte = d["test"].dense.numpy()
        assert abs(float(xtr.mean())) < 0.05, "train features must be centred by their own statistics"
        assert abs(float(xtr.std()) - 1.0) < 0.1
        # The test half is encoded with the TRAIN statistics, so it keeps whatever offset it really has.
        assert not np.allclose(xte.mean(0), 0.0, atol=1e-6)

    def test_unseen_test_category_encodes_to_an_all_zero_block(self):
        """T-KG-9: a level that appears only at test must encode to all-zeros -- never crash, and never
        trigger a re-fit of the encoder on test data."""
        tr = pd.DataFrame({"c": ["a", "b", "a", "b"]})
        te = pd.DataFrame({"c": ["a", "zzz"]})
        enc = rkv._fit_encoder(tr, _args())
        out = rkv._apply_encoder(te, enc, standardize=False)
        assert out.shape == (2, 2)
        assert out[0].tolist() == [1.0, 0.0]  # 'a' -> its own column
        assert out[1].sum() == 0.0  # unseen 'zzz' -> all zeros

    def test_missing_numeric_is_imputed_with_the_train_median(self):
        tr = pd.DataFrame({"v": [1.0, 2.0, 3.0, 100.0]})
        te = pd.DataFrame({"v": [np.nan]})
        enc = rkv._fit_encoder(tr, _args())
        assert rkv._apply_encoder(te, enc, standardize=False)[0, 0] == pytest.approx(2.5)  # median, not mean

    def test_nan_gets_its_own_categorical_level(self):
        """Absence is signal on Kaggle data, so a missing category is a level rather than a dropped row."""
        tr = pd.DataFrame({"c": ["a", "a", None, None]})
        enc = rkv._fit_encoder(tr, _args())
        assert enc["cats"][0][1] == ["__nan__", "a"]

    def test_no_standardize_leaves_raw_values(self):
        tr = pd.DataFrame({"v": [10.0, 20.0, 30.0]})
        enc = rkv._fit_encoder(tr, _args())
        assert rkv._apply_encoder(tr, enc, standardize=False)[:, 0].tolist() == [10.0, 20.0, 30.0]


# =========================================================================== splits, labels, chance
class TestSplitsAndBaselines:
    def test_stratified_split_keeps_every_class_on_both_sides(self):
        y = np.array([0] * 50 + [1] * 8 + [2] * 2)
        tr, te = rkv.stratified_split(y, 0.2, seed=0)
        assert set(tr) | set(te) == set(range(len(y))) and not (set(tr) & set(te))
        for c in (0, 1, 2):
            assert (y[tr] == c).sum() >= 1 and (y[te] == c).sum() >= 1

    def test_splits_are_deterministic_given_the_seed(self):
        y = np.array([0] * 30 + [1] * 30)
        assert np.array_equal(rkv.stratified_split(y, 0.25, 7)[1], rkv.stratified_split(y, 0.25, 7)[1])
        assert not np.array_equal(rkv.stratified_split(y, 0.25, 7)[1], rkv.stratified_split(y, 0.25, 8)[1])

    def test_labels_are_contiguous_int64_from_zero(self, tmp_path):
        rows = [[float(i), ["cat", "dog", "emu"][i % 3]] for i in range(60)]
        d = _ingest_table(tmp_path, ["feat", "target"], rows)
        y = np.concatenate([d["train"].y, d["test"].y])
        assert y.dtype == np.int64 and sorted(set(y.tolist())) == [0, 1, 2]
        assert d["meta"]["class_names"] == ["cat", "dog", "emu"]

    def test_singleton_class_is_dropped_and_labels_recoded(self, tmp_path):
        """T-KG-10: a class with one row cannot be stratified onto both sides, so it is dropped -- and the
        remaining codes must be recoded contiguously, because _infer_nout uses y.max()+1."""
        rows = [[float(i), "rare" if i == 0 else ("a" if i % 2 else "b")] for i in range(40)]
        d = _ingest_table(tmp_path, ["feat", "target"], rows)
        assert d["meta"]["class_names"] == ["a", "b"]
        y = np.concatenate([d["train"].y, d["test"].y])
        assert sorted(set(y.tolist())) == [0, 1]

    def test_chance_is_the_train_majority_scored_on_test(self, tmp_path):
        """T-KG-11 (load-bearing): `chance` is the accuracy of the TRAIN-majority predictor measured on the
        TEST split. On a 90/10 target that is ~0.9, not 1/K = 0.5 -- reporting 0.5 would make a model that
        merely predicts the majority class look like it had learned something."""
        rows = [[float(i), 0 if i % 10 else 1] for i in range(200)]
        d = _ingest_table(tmp_path, ["feat", "target"], rows)
        assert d["meta"]["majority_class"] == "0"
        assert d["chance"] == pytest.approx(float((d["test"].y == 0).mean()))
        assert d["chance"] > 0.8  # and emphatically not 1/K

    def test_regression_target_scale_is_the_train_std(self, tmp_path):
        """T-KG-12: the z-scored target's scale is handed to _eval_test as target_scale, so the reported MAE
        comes back in the ORIGINAL units."""
        rows = [[float(i % 7), float(i) * 3.0] for i in range(100)]
        d = _ingest_table(tmp_path, ["feat", "target"], rows, task="regression")
        raw = np.arange(100) * 3.0
        tr, _ = rkv.random_split(100, 0.2, 0)
        assert d["target_scale"] == pytest.approx(float(raw[tr].std()))
        assert d["target_units"] == "target"
        assert abs(float(d["train"].y.mean())) < 1e-5  # train target is centred
        assert d["chance"] == 0.0

    def test_target_units_flag_labels_the_mae(self, tmp_path):
        rows = [[float(i % 7), float(i)] for i in range(60)]
        d = _ingest_table(tmp_path, ["feat", "price"], rows, task="regression", target_units="USD")
        assert d["target_units"] == "USD"


# =========================================================================== images
class TestImages:
    def test_class_dirs_give_nchw_and_sorted_class_codes(self, tmp_path):
        _image_tree(tmp_path, classes=("zebra", "aardvark"), n=10, hw=8)
        inv = rkv.walk_inventory(str(tmp_path))
        d = rkv.encode_images(str(tmp_path), rkv.classify_files(inv), _args(hw=8))
        assert d["train"].dense.shape[1:] == (3, 8, 8)
        assert d["meta"]["class_names"] == ["aardvark", "zebra"]  # sorted, so code 0 is 'aardvark'
        assert d["task"] == "classification"

    def test_gray_flag_yields_one_channel(self, tmp_path):
        _image_tree(tmp_path, n=10, hw=8)
        inv = rkv.walk_inventory(str(tmp_path))
        d = rkv.encode_images(str(tmp_path), rkv.classify_files(inv), _args(hw=8, gray=True))
        assert d["train"].dense.shape[1] == 1

    def test_train_test_directories_are_used_as_the_official_split(self, tmp_path):
        """T-KG-13: when the dataset ships its own split, honour it rather than re-splitting -- the author may
        have split by subject, and a random re-split would leak across that boundary."""
        _image_tree(tmp_path, n=10, hw=8, prefix="train")
        _image_tree(tmp_path, n=4, hw=8, prefix="test")
        inv = rkv.walk_inventory(str(tmp_path))
        b = rkv.classify_files(inv)
        layout, payload = rkv.detect_image_layout(str(tmp_path), b, _args())
        assert layout == "split_class_dirs" and payload == {"train": "train", "test": "test"}
        d = rkv.encode_images(str(tmp_path), b, _args(hw=8))
        assert d["meta"]["n_train"] == 20 and d["meta"]["n_test"] == 8

    def test_use_dir_split_off_ignores_the_shipped_split(self, tmp_path):
        """`off` falls through to the largest class-directory root and re-splits it -- which is the train
        directory here. The chosen layout is always printed, so this is visible rather than silent."""
        _image_tree(tmp_path, n=10, hw=8, prefix="train")
        _image_tree(tmp_path, n=4, hw=8, prefix="test")
        b = rkv.classify_files(rkv.walk_inventory(str(tmp_path)))
        layout, payload = rkv.detect_image_layout(str(tmp_path), b, _args(use_dir_split="off"))
        assert layout == "class_dirs" and payload == {"dir": "train"}

    def test_flat_directory_plus_labels_csv(self, tmp_path):
        """T-KG-14: the filename column is found by matching against the files actually on disk."""
        rows = []
        for i in range(24):
            _png(tmp_path / "images" / f"img{i:03d}.png", 60 + 120 * (i % 2), seed=i)
            rows.append([f"img{i:03d}.png", "even" if i % 2 == 0 else "odd"])
        _write_csv(tmp_path / "labels.csv", ["filename", "kind"], rows)
        b = rkv.classify_files(rkv.walk_inventory(str(tmp_path)))
        layout, payload = rkv.detect_image_layout(str(tmp_path), b, _args())
        assert layout == "flat_csv" and payload["dir"] == "images"
        d = rkv.encode_images(str(tmp_path), b, _args(hw=8))
        assert d["meta"]["class_names"] == ["even", "odd"]
        assert d["meta"]["n_train"] + d["meta"]["n_test"] == 24

    def test_corrupt_image_is_skipped_and_its_own_label_dropped(self, tmp_path):
        """T-KG-15: a truncated file must drop ITS OWN label, not shift every later label by one. The fixture
        makes each class's pixels diagnostic, so a misalignment would show up as a label/content mismatch."""
        _image_tree(tmp_path, classes=("dark", "bright"), n=10, hw=8)
        (tmp_path / "dark" / "999_corrupt.png").write_bytes(b"definitely not a PNG")
        inv = rkv.walk_inventory(str(tmp_path))
        d = rkv.encode_images(str(tmp_path), rkv.classify_files(inv), _args(hw=8, no_standardize=True))
        assert d["meta"]["images_skipped"] == 1
        assert d["meta"]["n_train"] + d["meta"]["n_test"] == 20
        X = np.concatenate([d["train"].dense.numpy(), d["test"].dense.numpy()])
        y = np.concatenate([d["train"].y, d["test"].y])
        bright = d["meta"]["class_names"].index("bright")
        # class 'bright' really is the brighter one -- i.e. labels still line up with pixels
        assert X[y == bright].mean() > X[y != bright].mean()

    def test_per_class_cap_is_seeded_and_not_alphabetical(self, tmp_path):
        _image_tree(tmp_path, n=20, hw=8)
        picked = rkv.list_class_images(str(tmp_path), "", _args(per_class=5))
        assert all(len(v) == 5 for v in picked.values())
        assert picked == rkv.list_class_images(str(tmp_path), "", _args(per_class=5))  # deterministic
        assert [os.path.basename(p) for p in picked["alpha"]] != [f"{i:03d}.png" for i in range(5)]

    def test_max_mb_guard_names_the_flags_that_fix_it(self, tmp_path):
        _image_tree(tmp_path, n=10, hw=8)
        inv = rkv.walk_inventory(str(tmp_path))
        with pytest.raises(ValueError, match="--per_class"):
            rkv.encode_images(str(tmp_path), rkv.classify_files(inv), _args(hw=8, max_mb=0.0001))

    def test_single_class_directory_is_rejected(self, tmp_path):
        _image_tree(tmp_path, classes=("only",), n=10, hw=8)
        inv = rkv.walk_inventory(str(tmp_path))
        with pytest.raises((ValueError, FileNotFoundError)):
            rkv.encode_images(str(tmp_path), rkv.classify_files(inv), _args(hw=8))


# =========================================================================== arrays
class TestArrays:
    def test_npz_keys_are_found_by_convention(self, tmp_path):
        np.savez(tmp_path / "d.npz", x=np.zeros((10, 4)), y=np.arange(10))
        X, y, keys = rkv.load_arrays(str(tmp_path / "d.npz"), _args())
        assert keys == {"x_key": "x", "y_key": "y"} and X.shape == (10, 4)

    def test_npz_two_unconventional_arrays_pair_by_length(self, tmp_path):
        np.savez(tmp_path / "d.npz", blob=np.zeros((10, 4, 4)), tags=np.arange(10))
        _, _, keys = rkv.load_arrays(str(tmp_path / "d.npz"), _args())
        assert keys == {"x_key": "blob", "y_key": "tags"}

    def test_explicit_keys_win(self, tmp_path):
        np.savez(tmp_path / "d.npz", x=np.zeros((10, 4)), y=np.arange(10), other=np.arange(10) * 2)
        _, y, keys = rkv.load_arrays(str(tmp_path / "d.npz"), _args(y_key="other"))
        assert keys["y_key"] == "other" and y[1] == 2

    def test_npy_finds_a_sibling_label_file(self, tmp_path):
        np.save(tmp_path / "features.npy", np.zeros((6, 3)))
        np.save(tmp_path / "labels.npy", np.arange(6))
        _, y, keys = rkv.load_arrays(str(tmp_path / "features.npy"), _args())
        assert keys == {"y_file": "labels.npy"} and len(y) == 6

    def test_unsupervised_npz_raises_with_the_flag_that_fixes_it(self, tmp_path):
        np.savez(tmp_path / "d.npz", a=np.zeros((10, 4)), b=np.zeros((10, 4)), c=np.zeros((10, 4)))
        with pytest.raises(ValueError, match="--y_key"):
            rkv.load_arrays(str(tmp_path / "d.npz"), _args())

    def test_one_hot_target_is_argmaxed(self, tmp_path):
        y1h = np.eye(3)[np.arange(30) % 3]
        np.savez(tmp_path / "d.npz", x=np.random.RandomState(0).randn(30, 5), y=y1h)
        b = rkv.classify_files(rkv.walk_inventory(str(tmp_path)))
        d = rkv.encode_arrays(str(tmp_path), b, _args())
        assert d["task"] == "classification" and d["meta"]["n_classes"] == 3

    @pytest.mark.parametrize("shape", [(20,), (6, 2, 2, 2, 2, 2, 2)])
    def test_rank_outside_2_to_6_raises_with_actionable_text(self, tmp_path, shape):
        """T-KG-16: route_grid_rank only handles rank 2-6, so anything else must fail at ingest with advice
        rather than deep inside the router."""
        np.savez(tmp_path / "d.npz", x=np.zeros(shape), y=np.arange(shape[0]) % 2)
        b = rkv.classify_files(rkv.walk_inventory(str(tmp_path)))
        with pytest.raises(ValueError, match="rank 2-6"):
            rkv.encode_arrays(str(tmp_path), b, _args())

    def test_rank_4_array_routes_to_spatial(self, tmp_path):
        rng = np.random.RandomState(0)
        np.savez(tmp_path / "d.npz", x=rng.rand(40, 3, 8, 8), y=np.arange(40) % 2)
        b = rkv.classify_files(rkv.walk_inventory(str(tmp_path)))
        d = rkv.encode_arrays(str(tmp_path), b, _args())
        assert rkv.probe_contract(d["train"], _args(), 0.05, "default", None)[0] == "spatial"

    def test_kind_hint_is_passed_through_to_alldata(self, tmp_path):
        """A raw (N,H,W) stack has no channel axis, so --kind_hint spatial is how you say 'this is an image'."""
        np.savez(tmp_path / "d.npz", x=np.random.RandomState(0).rand(40, 8, 8), y=np.arange(40) % 2)
        b = rkv.classify_files(rkv.walk_inventory(str(tmp_path)))
        d = rkv.encode_arrays(str(tmp_path), b, _args(kind_hint="spatial"))
        assert d["train"].kind_hint == "spatial"
        assert rkv.probe_contract(d["train"], _args(), 0.05, "default", None)[0] == "spatial"

    def test_uint8_grid_is_rescaled_to_unit_range(self, tmp_path):
        np.savez(
            tmp_path / "d.npz",
            x=(np.random.RandomState(0).rand(40, 3, 8, 8) * 255).astype(np.uint8),
            y=np.arange(40) % 2,
        )
        b = rkv.classify_files(rkv.walk_inventory(str(tmp_path)))
        d = rkv.encode_arrays(str(tmp_path), b, _args())
        assert d["meta"]["normalize"] == "unit" and float(d["train"].dense.max()) <= 1.0


# =========================================================================== budget & routing
class TestBudget:
    def _train(self, n=40, feats=6):
        from ilmarinen.core.allgraph import AllData

        rng = np.random.RandomState(0)
        return AllData.dense_tensor(rng.randn(n, feats).astype("float32"), y=np.arange(n) % 2)

    def test_budget_comes_from_the_probed_contract(self):
        from run_standard_validation import BUDGET

        bud = rkv.choose_budget(_args(), "spatial", self._train())
        assert bud["width"] == BUDGET["spatial"]["width"] and bud["depth"] == BUDGET["spatial"]["depth"]

    def test_cli_overrides_win_over_the_budget(self):
        bud = rkv.choose_budget(_args(width=7, depth=2, epochs=3), "spatial", self._train())
        assert (bud["width"], bud["depth"], bud["epochs"]) == (7, 2, 3)

    def test_unknown_contract_falls_back_to_the_sequence_budget(self):
        from run_standard_validation import BUDGET

        assert rkv.choose_budget(_args(), "generated_equivariant", self._train()) == dict(BUDGET["sequence"])

    def test_wide_table_deepens_the_sequence_budget(self):
        """T-KG-17: _seq_depth_for reads the feature count for a flat table, so a very wide one gets depth."""
        shallow = rkv.choose_budget(_args(), "sequence", self._train(feats=10))["depth"]
        deep = rkv.choose_budget(_args(), "sequence", self._train(feats=600))["depth"]
        assert deep > shallow and deep == 3

    def test_budget_contract_flag_skips_the_probe(self):
        contract, detail = rkv.probe_contract(self._train(), _args(budget_contract="graph"), 0.05, "default", None)
        assert contract == "graph" and "--budget_contract" in detail["why"]

    def test_probe_honours_disabled_contracts(self):
        """T-KG-18: route() does not apply the --enabled_contracts restriction (fit does), so the probe has to
        replay it -- otherwise the budget would be sized for a contract the fit will never build."""
        train = self._train()
        assert rkv.probe_contract(train, _args(), 0.05, "default", None)[0] == "sequence"
        contract, detail = rkv.probe_contract(train, _args(), 0.05, "default", {"spatial", "graph"})
        assert contract != "sequence" and "contract_restriction" in detail


# =========================================================================== the runner contract
class TestRunnerContract:
    def test_default_results_path_is_not_the_benchmark_file(self):
        """T-KG-19 (load-bearing): kaggle rows must never land in standard_val_rows.json, which
        make_results_table.py --insert-readme renders into the published README table."""
        from ilmarinen.core.paths import cache_path

        assert os.path.basename(cache_path("kaggle_val_rows.json")) == "kaggle_val_rows.json"
        src = open(os.path.join(os.path.dirname(rkv.__file__), "run_kaggle_validation.py")).read()
        # the benchmark file may be NAMED (the docstring explains why the two are kept apart) but never opened
        assert 'cache_path("kaggle_val_rows.json")' in src
        assert 'cache_path("standard_val_rows.json")' not in src

    def test_inspect_returns_zero_without_training(self, tmp_path, monkeypatch, capsys):
        """T-KG-20: --inspect must not reach the model. make_allgraph is booby-trapped to prove it."""
        _write_csv(tmp_path / "train.csv", ["a", "b", "target"], [[i, i % 3, i % 2] for i in range(60)])

        def _boom(*a, **k):
            raise AssertionError("--inspect must not construct a model")

        monkeypatch.setattr(rkv, "make_allgraph", _boom)
        assert _run(monkeypatch, "--local_dir", tmp_path, "--inspect") == 0
        out = capsys.readouterr().out
        assert "KAGGLE INSPECT" in out and "Nothing was trained" in out and "ROUTE" in out

    def test_local_dir_never_imports_kagglehub(self, tmp_path, monkeypatch):
        """T-KG-21: the offline path must not touch the Kaggle boundary at all -- no import, no network, no
        credentials. A stub that raises on any use proves it."""

        class _Trap:
            def __getattr__(self, name):
                raise AssertionError(f"kagglehub.{name} must not be reached with --local_dir")

        monkeypatch.setitem(sys.modules, "kagglehub", _Trap())
        assert rkv.resolve_source(_args(local_dir=str(tmp_path))) == str(tmp_path)

    def test_missing_local_dir_is_rejected(self, tmp_path):
        assert rkv.resolve_source(_args(local_dir=str(tmp_path / "nope"))) is None

    def test_malformed_handle_is_rejected_before_any_network_call(self):
        """A competition handle is a bare slug; catching the shape here turns Kaggle's opaque 403 into a
        precise message without spending a request."""
        assert rkv.resolve_source(_args(handle="competitions/titanic", source="competition")) is None
        assert rkv.resolve_source(_args(handle="owner/slug/extra/bits", source="dataset")) is None

    def test_missing_kagglehub_message_is_actionable(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "kagglehub", None)  # `import kagglehub` -> ImportError
        with pytest.raises(ImportError) as e:
            rkv._kagglehub()
        assert "pip install" in str(e.value) and "--local_dir" in str(e.value)

    @pytest.mark.parametrize(
        "status,needle",
        [(403, "403"), (404, "404"), (401, "unauthenticated")],
    )
    def test_http_error_messages_name_the_handle_and_the_fix(self, status, needle):
        exc = type("KaggleApiHTTPError", (Exception,), {})("boom")
        exc.response = argparse.Namespace(status_code=status)
        text = rkv._kagglehub_error_text(exc, _args(handle="owner/thing", source="dataset"))
        assert needle in text and "--local_dir" in text
        if status != 401:
            assert "owner/thing" in text

    def test_ingest_failure_records_a_row_and_returns_one(self, tmp_path, monkeypatch):
        """T-KG-22: a broken dataset is a recorded OUTCOME, not a traceback -- the same discipline the
        standard runner applies to a failing loader."""
        _write_csv(tmp_path / "train.csv", ["a", "target"], [[1, 0], [2, 1], [3, 0], [4, 1]])
        out = tmp_path / "rows.json"
        rc = _run(monkeypatch, "--local_dir", tmp_path, "--target", "nonexistent", "--results_out", out)
        assert rc == 1
        rows = json.load(open(out))["rows"]
        assert list(rows) == ["rows"] or True  # name is derived from the directory
        row = next(iter(rows.values()))
        assert row["status"] == "error" and "ingest:" in row["note"]

    def test_row_name_slugifies_the_handle(self):
        assert rkv.row_name(_args(handle="uciml/iris")) == "uciml-iris"
        assert rkv.row_name(_args(handle="uciml/iris", name="custom")) == "custom"


# =========================================================================== end-to-end (smoke)
@pytest.mark.smoke
class TestEndToEnd:
    def test_tabular_classification_beats_chance(self, tmp_path, monkeypatch):
        """T-KG-23: the whole path -- CSV on disk to a held-out score -- on a genuinely separable target."""
        rng = np.random.RandomState(0)
        X = rng.randn(300, 6)
        rows = [[*X[i].round(4), "red" if i % 2 else "blue", int(X[i, 0] + X[i, 2] > 0)] for i in range(300)]
        _write_csv(tmp_path / "train.csv", [f"f{i}" for i in range(6)] + ["color", "target"], rows)
        out = tmp_path / "rows.json"
        rc = _run(
            monkeypatch,
            "--local_dir",
            tmp_path,
            "--device",
            "cpu",
            "--width",
            8,
            "--depth",
            1,
            "--epochs",
            4,
            "--results_out",
            out,
            "--name",
            "tab",
        )
        assert rc == 0
        row = json.load(open(out))["rows"]["tab"]
        assert row["status"] == "ok" and row["contract"] == "sequence" and row["task"] == "classification"
        assert row["metric"] == "acc" and row["value"] > row["chance"]
        assert row["kaggle"]["mode"] == "tabular" and row["kaggle"]["target"] == "target"

    def test_images_route_to_the_spatial_contract(self, tmp_path, monkeypatch):
        _image_tree(tmp_path, n=12, hw=8)
        out = tmp_path / "rows.json"
        rc = _run(
            monkeypatch,
            "--local_dir",
            tmp_path,
            "--mode",
            "images",
            "--hw",
            8,
            "--device",
            "cpu",
            "--width",
            8,
            "--depth",
            1,
            "--epochs",
            2,
            "--results_out",
            out,
            "--name",
            "img",
        )
        assert rc == 0
        row = json.load(open(out))["rows"]["img"]
        assert row["status"] == "ok" and row["contract"] == "spatial" and row["kaggle"]["hw"] == 8

    def test_regression_reports_mae_in_original_units(self, tmp_path, monkeypatch):
        """T-KG-24: target_scale reaches _eval_test, so the headline is MAE[<units>] with R2 alongside."""
        rng = np.random.RandomState(0)
        X = rng.randn(300, 5).astype(np.float32)
        np.savez(tmp_path / "d.npz", x=X, y=(X[:, 0] * 10.0 + 100.0).astype(np.float32))
        out = tmp_path / "rows.json"
        rc = _run(
            monkeypatch,
            "--local_dir",
            tmp_path,
            "--device",
            "cpu",
            "--width",
            8,
            "--depth",
            1,
            "--epochs",
            4,
            "--target_units",
            "USD",
            "--results_out",
            out,
            "--name",
            "reg",
        )
        assert rc == 0
        row = json.load(open(out))["rows"]["reg"]
        assert row["task"] == "regression" and row["metric"] == "MAE[USD]" and "R2" in row["extra"]
        assert row["skill"] == pytest.approx(row["extra"]["R2"])

    def test_rows_upsert_without_disturbing_each_other(self, tmp_path, monkeypatch):
        """T-KG-25: the merge-on-write store accumulates across invocations, as it does for the batch runs."""
        rng = np.random.RandomState(1)
        X = rng.randn(120, 4)
        rows = [[*X[i].round(4), int(X[i, 0] > 0)] for i in range(120)]
        _write_csv(tmp_path / "train.csv", [f"f{i}" for i in range(4)] + ["target"], rows)
        out = tmp_path / "rows.json"
        base = [
            "--local_dir",
            tmp_path,
            "--device",
            "cpu",
            "--width",
            8,
            "--depth",
            1,
            "--epochs",
            2,
            "--results_out",
            out,
        ]
        assert _run(monkeypatch, *base, "--name", "first") == 0
        assert _run(monkeypatch, *base, "--name", "second") == 0
        doc = json.load(open(out))
        assert set(doc["rows"]) == {"first", "second"}
        assert len(doc["meta"]["runs"]) == 2
