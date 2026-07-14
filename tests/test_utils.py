"""Unit tests for split utilities — group-stratified splitting and leakage checks."""
import numpy as np
import pandas as pd

import pytest

from bertuner.utils import (
    split_group_stratified,
    check_group_leakage,
    train_val_test_split,
)


def make_grouped_df(n_groups=20, rows_per_group=5, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_groups):
        label = g % 2  # group-level label keeps stratification feasible
        for i in range(rows_per_group):
            rows.append({"group": f"question_{g}", "target": label, "text": f"row {g}-{i}"})
    return pd.DataFrame(rows)


class TestSplitGroupStratified:
    def test_no_group_overlap(self):
        df = make_grouped_df()
        train, val, test = split_group_stratified(df, "group", "target", seed=42)
        train_g, val_g, test_g = set(train["group"]), set(val["group"]), set(test["group"])
        assert not train_g & val_g
        assert not train_g & test_g
        assert not val_g & test_g

    def test_all_rows_covered_exactly_once(self):
        df = make_grouped_df()
        train, val, test = split_group_stratified(df, "group", "target", seed=42)
        assert len(train) + len(val) + len(test) == len(df)

    def test_both_classes_in_every_split(self):
        df = make_grouped_df(n_groups=40)
        train, val, test = split_group_stratified(df, "group", "target", seed=42)
        for split in (train, val, test):
            assert set(split["target"]) == {0, 1}

    def test_reproducible_with_same_seed(self):
        df = make_grouped_df()
        a = split_group_stratified(df, "group", "target", seed=7)
        b = split_group_stratified(df, "group", "target", seed=7)
        for x, y in zip(a, b):
            pd.testing.assert_frame_equal(x, y)


class TestCheckGroupLeakage:
    def test_clean_split_reports_verified(self, capsys):
        df = make_grouped_df()
        train, val, test = split_group_stratified(df, "group", "target", seed=42)
        check_group_leakage(train, val, test, "group")
        assert "No group leakage" in capsys.readouterr().out

    def test_leaky_split_reports_warning(self, capsys):
        df = make_grouped_df()
        check_group_leakage(df, df, df, "group")
        assert "Leakage detected" in capsys.readouterr().out


class TestTrainValTestSplit:
    def make_xy(self, n=100):
        X = pd.DataFrame({"text": [f"row {i}" for i in range(n)]})
        y = pd.DataFrame({"target": np.arange(n) % 2})
        return X, y

    def test_split_sizes(self):
        X, y = self.make_xy()
        X_tr, X_val, X_te, y_tr, y_val, y_te = train_val_test_split(X, y)
        assert len(X_tr) == 70
        assert len(X_val) == 15
        assert len(X_te) == 15
        assert len(X_tr) == len(y_tr)

    def test_stratification_preserved(self):
        X, y = self.make_xy()
        _, _, _, y_tr, y_val, y_te = train_val_test_split(X, y, stratify_y=True)
        for split in (y_tr, y_val, y_te):
            assert split["target"].mean() == pytest.approx(0.5, abs=0.05)

    def test_reproducible_with_same_seed(self):
        X, y = self.make_xy()
        a = train_val_test_split(X, y, random_state=7)
        b = train_val_test_split(X, y, random_state=7)
        for x, z in zip(a, b):
            pd.testing.assert_frame_equal(x, z)

    def test_rejects_sizes_not_summing_to_one(self):
        X, y = self.make_xy()
        with pytest.raises(ValueError):
            train_val_test_split(X, y, train_size=0.5, validation_size=0.2, test_size=0.2)
