"""Tests for the ported data layer (barcode registry) and core gems
(leakage-controlled splits, bootstrap-Spearman drivers, split-conformal)."""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from kalos import bootstrap_spearman, rank_drivers, split_conformal
from kalos.data.barcode_registry import BarcodeRegistry
from kalos.core.splits import row_hash_groups, make_splits, assert_no_group_leakage


def _sheet(n=12, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "pH": rng.uniform(5, 7, n).round(2),
        "Glycerol": rng.uniform(0, 40, n).round(1),
        "titer": rng.uniform(0, 1, n).round(3),
        "strain": "X33",              # identity -> must be dropped
        "campaign_id": "C-2026",      # pseudonymous -> must be hashed
    })


def test_barcode_registry_register_query_persist():
    df = _sheet()
    reg = BarcodeRegistry()
    info = reg.register_dataset("anagram", df, result_cols=["titer"], meta_cols=["campaign_id"])
    assert info["dataset_barcode"].startswith("KAL-DS-")
    assert len(info["barcodes"]) == len(df) and all(b.startswith("KAL-") for b in info["barcodes"])

    bc = info["barcodes"][0]
    rec = reg.get(bc)
    assert "pH" in rec.features and "titer" in rec.results
    assert "strain" not in rec.meta and "strain" not in rec.features   # identity dropped
    assert rec.meta.get("campaign_id") != "C-2026" and rec.meta["anonymized"]  # hashed
    assert set(reg.filter("anagram")) == set(info["barcodes"])

    flat = reg.to_dataframe(reg.filter("anagram"))
    assert "barcode" in flat.columns and "pH" in flat.columns and len(flat) == len(df)

    # registering the SAME data is idempotent (content-stable barcodes)
    reg.register_dataset("anagram", df, result_cols=["titer"], meta_cols=["campaign_id"])
    assert len(reg) == len(df)

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        path = fh.name
    reg.save(path)
    reloaded = BarcodeRegistry.load(path)
    assert len(reloaded) == len(reg) and reloaded.get(bc).results["titer"] == rec.results["titer"]
    os.unlink(path)


def test_splits_group_replicates_and_tripwire():
    # two recipes, each replicated 4x -> 2 groups; replicates must not straddle folds
    X = pd.DataFrame(np.repeat([[5.5, 10.0], [6.0, 20.0]], 4, axis=0), columns=["pH", "gly"])
    g = row_hash_groups(X)
    assert len(np.unique(g)) == 2
    y = np.arange(8) % 2
    splits = make_splits(X, y, g, n_splits=2)
    assert_no_group_leakage(splits, g)   # passes
    # a hand-built leaky split must raise
    bad = [(np.array([0, 1]), np.array([2, 3]))]  # group 0 on both sides
    with pytest.raises(AssertionError, match="(?i)leakage"):
        assert_no_group_leakage(bad, g)


def test_row_hash_groups_keeps_nan_distinct_from_zero():
    # A real 0.0 and a missing value in the same column must NOT be treated as one
    # replicate; collapsing NaN to 0.0 merged unrelated rows and leaked them.
    X = pd.DataFrame({"pH": [6.0, 6.0], "gly": [0.0, np.nan]})
    g = row_hash_groups(X)
    assert len(np.unique(g)) == 2


def test_row_hash_groups_handles_categorical_columns():
    # Mixed numeric + string (e.g. a medium name) must group without raising.
    X = pd.DataFrame({"medium": ["CD Fusion", "CD Fusion", "BalanCD"], "gly": [10.0, 10.0, 20.0]})
    g = row_hash_groups(X)
    assert len(g) == 3 and len(np.unique(g)) == 2   # first two are replicates


def test_make_splits_returns_no_dummy_when_too_few_groups():
    # One group cannot be split leakage-free: must yield NO evaluable split
    # (was: a train==validation dummy that scored the model against itself).
    X = pd.DataFrame({"pH": [6.0, 6.0, 6.0], "gly": [10.0, 10.0, 10.0]})
    g = row_hash_groups(X)              # all identical -> 1 group
    y = np.array([0.1, 0.2, 0.3])
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        splits = make_splits(X, y, g, n_splits=5)
    assert splits == []


def test_bootstrap_drivers_rank_real_driver_first():
    rng = np.random.default_rng(0)
    n = 200
    x0 = rng.uniform(0, 1, n)
    noise = rng.uniform(0, 1, n)
    y = 3 * x0 + rng.normal(0, 0.1, n)
    boot = bootstrap_spearman(np.column_stack([x0, noise]), y, B=100, feature_names=["x0", "noise"])
    assert rank_drivers(boot, top_k=1)[0][0] == "x0"
    j = list(boot["feature_names"]).index("x0")
    assert boot["lo"][j] > 0                                  # real driver CI excludes 0
    jn = list(boot["feature_names"]).index("noise")
    assert boot["lo"][jn] < 0 < boot["hi"][jn]                # noise CI straddles 0


def test_split_conformal_coverage():
    rng = np.random.default_rng(1)
    y = rng.normal(0, 1, 500)
    q = split_conformal(lambda X: np.zeros(len(X)), np.zeros((500, 1)), y, alpha=0.1)
    covered = np.mean(np.abs(y) <= q)
    assert covered >= 0.88                                    # ~>= 1 - alpha
