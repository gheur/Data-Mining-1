def minMaxScaling(num_series):
    if not isinstance(num_series, list):
        return

    min_val = min(num_series)
    max_val = max(num_series)

    for i in range(len(num_series)):
        num_series[i] = float((num_series[i] - min_val)) / (max_val - min_val)

    return num_series