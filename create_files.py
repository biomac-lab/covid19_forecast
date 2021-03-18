from functions.adjust_cases_functions import prepare_cases
from global_config import config
from models.seird_model import SEIRModel
from models.seird_model import SEIRD
from functions.samples_utils import create_df_response
import pandas as pd
import numpy as np
import datetime
import os

import sys


if len(sys.argv) < 2:
    raise NotImplementedError()
else:
    poly_run  = int(sys.argv[1])
    name_dir  = str(sys.argv[2])
    drop_last_weeks = bool(sys.argv[3])

def load_samples(filename):

    x = np.load(filename, allow_pickle=True)

    mcmc_samples = x['mcmc_samples'].item()
    post_pred_samples = x['post_pred_samples'].item()
    forecast_samples = x['forecast_samples'].item()

    return mcmc_samples, post_pred_samples, forecast_samples

print("** Creating Files")
data_dir            = config.get_property('data_dir_covid')
geo_dir             = config.get_property('geo_dir')
data_dir_mnps       = config.get_property('data_dir_col')
results_dir         = config.get_property('results_dir')

agglomerated_folder = os.path.join(data_dir, 'data_stages', 'colombia', 'agglomerated', 'geometry' )
data =  pd.read_csv(os.path.join(agglomerated_folder, 'cases.csv'), parse_dates=['date_time'],
                    dayfirst=True).set_index('poly_id').loc[poly_run].set_index('date_time')
data = data.resample('D').sum().fillna(0)[['num_cases','num_diseased']]
data  = prepare_cases(data, col='num_cases', cutoff=0)    # .rename({'smoothed_num_cases':'num_cases'})
data  = prepare_cases(data, col='num_diseased', cutoff=0) # .rename({'smoothed_num_cases':'num_cases'})

polygons = pd.read_csv(os.path.join(agglomerated_folder, 'polygons.csv')).set_index('poly_id')
polygons = polygons.loc[poly_run]

data = data.rename(columns={'num_cases': 'confirmed', 'num_diseased':'death'})[['confirmed', 'death']]
data = prepare_cases(data, col='confirmed')
data = prepare_cases(data, col='death')

if drop_last_weeks:
    data = data.iloc[:-14]



T_future = 100
path_to_save = os.path.join(results_dir, 'weekly_forecast' , name_dir, pd.to_datetime(data.index.values[-1]).strftime('%Y-%m-%d'))

mcmc_samples, post_pred_samples, forecast_samples = load_samples(os.path.join(path_to_save,'samples.npz'))


model = SEIRD(
    confirmed = data['confirmed'].cumsum(),
    death     = data['death'].cumsum(),
    T         = len(data),
    N         = int(polygons["attr_population"]),
    samples   = mcmc_samples
    )

#model.dynamics(params=post_pred_samples, T         = len(data), x0=mcmc_samples["x0"])

forecast_samples['mean_dz0'] = forecast_samples["dz0"]
forecast_samples['mean_dy0'] = forecast_samples["dy0"]

deaths_fitted = model.combine_samples(forecast_samples, f='mean_dz', use_future=True)
cases_fitted  = model.combine_samples(forecast_samples, f='mean_dy', use_future=True)

df_deaths = create_df_response(deaths_fitted, time=len(data), date_init = pd.to_datetime(data.index.values[0]).strftime('%Y-%m-%d'),  forecast_horizon=100, use_future=True)
df_cases  = create_df_response(cases_fitted, time=len(data), date_init  = pd.to_datetime(data.index.values[0]).strftime('%Y-%m-%d'),  forecast_horizon=100, use_future=True)

df_deaths.to_csv(os.path.join(path_to_save, 'deaths_df.csv'))
df_cases.to_csv(os.path.join(path_to_save, 'cases_df.csv'))