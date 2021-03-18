from functions.adjust_cases_functions import prepare_cases 
from functions.general_utils import  get_bool
from models.seird_model import SEIRD

import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd
import numpy as np
import os

from global_config import config

import sys


if len(sys.argv) < 2:
    raise NotImplementedError()
else:
    poly_run  = int(sys.argv[1])
    name_dir  = str(sys.argv[2])
    drop_last_weeks = get_bool(sys.argv[3])
    print("**** Running inference and forecast for {}".format(name_dir))

data_dir            = config.get_property('data_dir_covid')
geo_dir             = config.get_property('geo_dir')
data_dir_mnps       = config.get_property('data_dir_col')
results_dir         = config.get_property('results_dir')
agglomerated_folder = os.path.join(data_dir, 'data_stages', 'colombia', 'agglomerated', 'geometry' )

polygons = pd.read_csv(os.path.join(agglomerated_folder, 'polygons.csv')).set_index('poly_id')
polygons = polygons.loc[poly_run]

data  =  pd.read_csv(os.path.join(agglomerated_folder, 'cases.csv'), parse_dates=['date_time'],
                    dayfirst=True).set_index('poly_id').loc[poly_run].set_index('date_time')
data  = data.resample('D').sum().fillna(0)[['num_cases','num_diseased']]
data  = prepare_cases(data, col='num_cases', cutoff=0)    # .rename({'smoothed_num_cases':'num_cases'})
data  = prepare_cases(data, col='num_diseased', cutoff=0) # .rename({'smoothed_num_cases':'num_cases'})
data  = data.rename(columns={'smoothed_num_cases': 'confirmed', 'smoothed_num_diseased':'death'})[['confirmed', 'death']]

print("**** **** Last day uploaded {}".format(pd.to_datetime(data.index.values[-1]).strftime('%Y-%b-%d')))

if drop_last_weeks:
    print("**** **** *** Droping last 2wk")
    print(sys.argv)
    data = data.iloc[:-14]

model = SEIRD(
    confirmed = data['confirmed'].cumsum(),
    death     = data['death'].cumsum(),
    T         = len(data),
    N         = int(polygons["attr_population"]),
    )


T_future = 100
path_to_save = os.path.join(results_dir, 'weekly_forecast' , name_dir, pd.to_datetime(data.index.values[-1]).strftime('%Y-%m-%d'))
print("**** **** **** Fitting until {}".format(pd.to_datetime(data.index.values[-1]).strftime('%Y-%b-%d')))

if not os.path.exists(path_to_save):
    os.makedirs(path_to_save)

samples = model.infer(num_warmup=400, num_samples=2000, num_chains=1)

# In-sample posterior predictive samples (don't condition on observations)
print(" * collecting in-sample predictive samples")
post_pred_samples = model.predictive()

# Forecasting posterior predictive (do condition on observations)
print(" * collecting forecast samples")
forecast_samples = model.forecast(T_future=T_future)

save_fields=['beta0', 'beta', 'sigma', 'gamma', 'dy0', 'dy', 'mean_dy', 'mean_dy0',  'dy_future', 'mean_dy_future',
                'dz0', 'dz', 'dz_future', 'mean_dz', 'mean_dz0', 'mean_dz_future',
                'y0', 'y', 'y_future', 'z0', 'z', 'z_future' ]

def trim(d, fields):
    if d is not None:
        d = {k : v for k, v in d.items() if k in fields}
    return d

np.savez_compressed(os.path.join(path_to_save, 'samples.npz'),
                        mcmc_samples = samples,
                        post_pred_samples = post_pred_samples,
                        forecast_samples = forecast_samples)

