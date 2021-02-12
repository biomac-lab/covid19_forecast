import pandas as pd
import numpy as np

# Define functions for model
def confirmed_to_onset(confirmed, p_delay, col_name='num_cases', min_onset_date=None):
    min_onset_date = pd.to_datetime(min_onset_date)
    # Reverse cases so that we convolve into the past
    convolved = np.convolve(np.squeeze(confirmed.iloc[::-1].values), p_delay)

    # Calculate the new date range
    dr = pd.date_range(end=confirmed.index[-1],
                        periods=len(convolved))
    # Flip the values and assign the date range
    onset = pd.Series(np.flip(convolved), index=dr, name=col_name)
    if min_onset_date:
        onset = np.round(onset.loc[min_onset_date:])
    else:
        onset = np.round(onset.iloc[onset.values>=1])

    onset.index.name = 'date'
    return pd.DataFrame(onset)

# Smooths cases using a rolling window and gaussian sampling
def prepare_cases(daily_cases, col='num_cases', out_col=None, cutoff=0):
    if not out_col:
        out_col = 'smoothed_'+str(col)

    daily_cases[out_col] = daily_cases[col].rolling(7,
        win_type='gaussian',
        min_periods=1,
        center=True).mean(std=2).round()

    idx_start = np.searchsorted(daily_cases[out_col], cutoff)
    daily_cases[out_col] = daily_cases[out_col].iloc[idx_start:]
    return daily_cases

# Smooths cases using a rolling window and gaussian sampling
def smooth_1d(signal, col='num_cases', out_col=None, cutoff=0):
    if not out_col:
        out_col = 'smoothed_'+str(col)

    signal[out_col] = signal[col].rolling(7,
        win_type='gaussian',
        min_periods=1,
        center=True).mean(std=2)

    idx_start = np.searchsorted(signal[out_col], cutoff)
    signal[out_col] = signal[out_col].iloc[idx_start:]
    return signal