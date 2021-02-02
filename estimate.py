
from functions.adjust_cases_functions import prepare_cases
from seir_model import SEIRD

import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd
import numpy as np
import os

from global_config import config


data_dir            = config.get_property('data_dir_covid')
geo_dir             = config.get_property('geo_dir')
data_dir_mnps       = config.get_property('data_dir_col')
results_dir         = config.get_property('results_dir')
agglomeration_df    = pd.read_csv(os.path.join(data_dir_mnps, 'administrative_division_col_2018.csv')).set_index('poly_id')
agglomerated_folder = os.path.join(data_dir, 'data_stages', 'colombia', 'agglomerated', 'geometry' )

data =  pd.read_csv(os.path.join(agglomerated_folder, 'cases.csv'), parse_dates=['date_time'], dayfirst=True).set_index('poly_id').loc[11001].set_index('date_time')
data = data.resample('D').sum().fillna(0)[['num_cases','num_diseased']]
data  = prepare_cases(data, col='num_cases', cutoff=0)    # .rename({'smoothed_num_cases':'num_cases'})
data  = prepare_cases(data, col='num_diseased', cutoff=0) # .rename({'smoothed_num_cases':'num_cases'})


data = data.rename(columns={'num_cases': 'confirmed', 'num_diseased':'death'})[['confirmed', 'death']]
data = prepare_cases(data, col='confirmed')
data = prepare_cases(data, col='death')

data = data.iloc[:-11]

model = SEIRD(
    confirmed = data['confirmed'].cumsum(),
    death     = data['death'].cumsum(),
    T         = len(data),
    N         = 8181047,
    )


T_future = 27
path_to_save = os.path.join(results_dir, 'weekly_forecast' , 'bogota', pd.to_datetime(data.index.values[-1]).strftime('%Y-%m-%d'))
if not os.path.exists(path_to_save):
    os.makedirs(path_to_save)

samples = model.infer(num_warmup=400, num_samples=2000)

# In-sample posterior predictive samples (don't condition on observations)
print(" * collecting in-sample predictive samples")
post_pred_samples = model.predictive()
# Forecasting posterior predictive (do condition on observations)
print(" * collecting forecast samples")
forecast_samples = model.forecast(T_future=T_future)

save_fields=['beta0', 'beta', 'sigma', 'gamma', 'dy0', 'dy', 'mean_dy',  'dy_future', 'mean_dy_future',
                'dz0', 'dz', 'dz_future', 'mean_dz', 'mean_dz_future',
                'y0', 'y', 'y_future', 'z0', 'z', 'z_future' ]

def trim(d):
    if d is not None:
        d = {k : v for k, v in d.items() if k in save_fields}
    return d

np.savez_compressed(os.path.join(path_to_save, 'samples.npz'),
                        mcmc_samples = trim(samples),
                        post_pred_samples = trim(post_pred_samples),
                        forecast_samples = trim(forecast_samples))

