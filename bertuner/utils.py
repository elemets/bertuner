import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold, train_test_split


def train_val_test_split(
    X,
    y,
    train_size=0.7,
    validation_size=0.15,
    test_size=0.15,
    random_state=42,
    stratify_y=True,
):
    """Two-stage stratified train/val/test split.

    Returns X_train, X_val, X_test, y_train, y_val, y_test in the same
    order as the model_tuner function this replaces.
    """
    total = train_size + validation_size + test_size
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"train/validation/test sizes must sum to 1, got {total}")

    holdout = validation_size + test_size
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X,
        y,
        test_size=holdout,
        random_state=random_state,
        stratify=y if stratify_y else None,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp,
        y_tmp,
        test_size=test_size / holdout,
        random_state=random_state,
        stratify=y_tmp if stratify_y else None,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def split_group_stratified(df, group_col, y_col, seed=42, test_size=0.15, val_size=0.15):
    groups = df[group_col].astype(str).str.strip().str.casefold().values
    y = df[y_col].values

    # First split: train vs temp (val+test)
    temp_size = test_size + val_size
    n_splits = int(round(1 / temp_size))
    n_splits = max(n_splits, 2)

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_idx, temp_idx = next(sgkf.split(df, y, groups=groups))

    temp_df = df.iloc[temp_idx].copy()
    temp_groups = temp_df[group_col].astype(str).str.strip().str.casefold().values
    temp_y = temp_df[y_col].values

    # Second split: val vs test (split temp)
    test_frac_of_temp = test_size / temp_size
    n_splits2 = int(round(1 / test_frac_of_temp))
    n_splits2 = max(n_splits2, 2)

    # StratifiedGroupKFold is heuristic and can produce a single-class fold;
    # retry with a bumped seed until val and test both contain every class.
    for attempt in range(10):
        sgkf2 = StratifiedGroupKFold(
            n_splits=n_splits2, shuffle=True, random_state=seed + 1 + attempt
        )
        val_rel_idx, test_rel_idx = next(sgkf2.split(temp_df, temp_y, groups=temp_groups))
        if len(set(temp_y[val_rel_idx])) > 1 and len(set(temp_y[test_rel_idx])) > 1:
            break

    val_idx = temp_df.index[val_rel_idx]
    test_idx = temp_df.index[test_rel_idx]

    train_df = df.iloc[train_idx].copy()
    val_df = df.loc[val_idx].copy()
    test_df = df.loc[test_idx].copy()

    return train_df, val_df, test_df


def check_group_leakage(train_df, val_df, test_df, group_col):
    train_q = set(train_df[group_col])
    val_q = set(val_df[group_col])
    test_q = set(test_df[group_col])

    overlap_tv = train_q & val_q
    overlap_tt = train_q & test_q
    overlap_vt = val_q & test_q

    print("Train ∩ Val:", len(overlap_tv))
    print("Train ∩ Test:", len(overlap_tt))
    print("Val ∩ Test:", len(overlap_vt))

    if overlap_tv or overlap_tt or overlap_vt:
        print("\nWARNING!! ___====Leakage detected!====___")
    else:
        print("\nVERIFIED: No group leakage across splits.")
