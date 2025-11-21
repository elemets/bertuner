import pandas as pd 

def enhanced_balance_data(
    df: pd.DataFrame, grouping_key: str, strategy: str = "adaptive", seed: int = 42
) -> pd.DataFrame:
    """Enhanced data balancing with multiple strategies."""
    question_counts = df[grouping_key].value_counts()

    if strategy == "adaptive":
        # Adaptive balancing based on distribution
        median_count = question_counts.median()
        min_samples = max(5, int(median_count * 0.5))
        max_samples = min(50, int(median_count * 2))
    elif strategy == "uniform":
        min_samples, max_samples = 10, 25
    elif strategy == "conservative":
        min_samples, max_samples = 3, 15
    else:
        min_samples, max_samples = 8, 30

    balanced_rows = []
    for question in question_counts.index:
        question_data = df[df[grouping_key] == question]
        current_count = len(question_data)

        if current_count < min_samples:
            needed = min_samples - current_count
            additional = question_data.sample(n=needed, replace=True, random_state=seed)
            balanced_rows.append(pd.concat([question_data, additional]))
        elif current_count > max_samples:
            sampled = question_data.sample(n=max_samples, random_state=seed)
            balanced_rows.append(sampled)
        else:
            balanced_rows.append(question_data)

    return pd.concat(balanced_rows, ignore_index=True)