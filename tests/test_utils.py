"""Unit tests for split utilities — group-stratified splitting and leakage checks."""
import numpy as np
import pandas as pd

from bertuner.utils import split_group_stratified, check_group_leakage


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
