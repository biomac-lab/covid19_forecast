from functions.adjust_cases_functions import prepare_cases
from models.seird_model import SEIRD
from models.seirhd_model import SEIRHD


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
agglomerated_folder = os.path.join(data_dir, 'data_stages', 'colombia', 'agglomerated', 'geometry')

data =  pd.read_csv(os.path.join(agglomerated_folder, 'cases.csv'), parse_dates=['date_time'], dayfirst=True).set_index('poly_id').loc[11001].set_index('date_time')

fig, axes = plt.subplots(2,1)
data["num_diseased"].plot(ax=axes[0], color='red', linestyle='--')
data["num_cases"].plot(ax=axes[1], color='k', linestyle='-')
ax_tw = axes[1].twinx()
data["num_infected_in_hospital"].plot(ax=ax_tw, color='blue', linestyle='--')
plt.show()

data = data.resample('D').sum().fillna(0)[['num_cases','num_diseased']]
data  = prepare_cases(data, col='num_cases', cutoff=0)
data  = prepare_cases(data, col='num_diseased', cutoff=0)
data = data.rename(columns={'smoothed_num_cases': 'confirmed', 'smoothed_num_diseased':'death'})[['confirmed', 'death']]
data = data.iloc[:-14]

model = SEIRHD(
    hospitalized     = data['death'].cumsum(),
    confirmed        = data['confirmed'].cumsum(),
    death            = data['death'].cumsum(),
    T                = len(data),
    N                = 8181047,
    )

T_future = 27
path_to_save = os.path.join(results_dir, 'weekly_forecast' , 'bogota', pd.to_datetime(data.index.values[-1]).strftime('%Y-%m-%d'))

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

#np.savez_compressed(os.path.join(path_to_save, 'samples.npz'),
#                        mcmc_samples = trim(samples, save_fields),
#                        post_pred_samples = trim(post_pred_samples, save_fields),
#                        forecast_samples = trim(forecast_samples, save_fields))

np.savez_compressed(os.path.join(path_to_save, 'samples.npz'),
                        mcmc_samples = samples,
                        post_pred_samples = post_pred_samples,
                        forecast_samples = forecast_samples)
