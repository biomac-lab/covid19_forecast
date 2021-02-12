from functions.adjust_cases_functions import prepare_cases
from global_config import config
from models.seird_model import SEIRModel
from models.seird_model import SEIRD

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

def load_samples(filename):

    x = np.load(filename, allow_pickle=True)

    mcmc_samples = x['mcmc_samples'].item()
    post_pred_samples = x['post_pred_samples'].item()
    forecast_samples = x['forecast_samples'].item()

    return mcmc_samples, post_pred_samples, forecast_samples


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


data = data.rename(columns={'num_cases': 'confirmed', 'num_diseased':'death'})[['confirmed', 'death']]
data = prepare_cases(data, col='confirmed')
data = prepare_cases(data, col='death')

data = data.iloc[:-14]



T_future = 27
path_to_save = os.path.join(results_dir, 'weekly_forecast' , name_dir, pd.to_datetime(data.index.values[-1]).strftime('%Y-%m-%d'))

mcmc_samples, post_pred_samples, forecast_samples = load_samples(os.path.join(path_to_save,'samples.npz'))


model = SEIRD(
    confirmed = data['confirmed'].cumsum(),
    death     = data['death'].cumsum(),
    T         = len(data),
    N         = 8181047,
    samples   = mcmc_samples
    )

#model.dynamics(params=post_pred_samples, T         = len(data), x0=mcmc_samples["x0"])

forecast_samples['mean_dz0'] = forecast_samples["dz0"]
forecast_samples['mean_dy0'] = forecast_samples["dy0"]

deaths_fitted = model.combine_samples(forecast_samples, f='mean_dz', use_future=True)
cases_fitted  = model.combine_samples(forecast_samples, f='mean_dy', use_future=True)

def create_df_response(samples, time, date_init ='2020-03-06',  forecast_horizon=27, use_future=False):

    dates_fitted   = pd.date_range(start=pd.to_datetime(date_init), periods=time)
    dates_forecast = pd.date_range(start=dates_fitted[-1]+datetime.timedelta(1), periods=forecast_horizon)
    dates = list(dates_fitted)
    types = ['estimate']*len(dates_fitted)
    if use_future:
        dates += list(dates_forecast)
        types  += ['forecast']*len(dates_forecast)

    results_df = pd.DataFrame(samples.T)
    df_response = pd.DataFrame(index=dates)
    # Calculate key statistics
    df_response['mean']        = results_df.mean(axis=1).values
    df_response['median']      = results_df.median(axis=1).values
    df_response['std']         = results_df.std(axis=1).values
    df_response['low_975']      = results_df.quantile(q=0.025, axis=1).values
    df_response['high_975']     = results_df.quantile(q=0.975, axis=1).values

    df_response['low_90']      = results_df.quantile(q=0.1, axis=1).values
    df_response['high_90']     = results_df.quantile(q=0.9, axis=1).values

    df_response['low_75']      = results_df.quantile(q=0.25, axis=1).values
    df_response['high_75']     = results_df.quantile(q=0.75, axis=1).values

    df_response['type']        =  types
    df_response.index.name = 'date'
    return df_response

df_deaths = create_df_response(deaths_fitted, time=len(data), date_init ='2020-03-06',  forecast_horizon=27, use_future=True)
df_cases  = create_df_response(cases_fitted, time=len(data), date_init ='2020-03-06',  forecast_horizon=27, use_future=True)

df_deaths.to_csv(os.path.join(path_to_save, 'deaths_df.csv'))
df_cases.to_csv(os.path.join(path_to_save, 'cases_df.csv'))


## Compute growth rate over time
#beta  = mcmc_samples['beta']
#sigma = mcmc_samples['sigma'][:,None]
#gamma = mcmc_samples['gamma'][:,None]
#
## compute Rt
#rt = SEIRModel.R0((beta, sigma, gamma))
#results_df = pd.DataFrame(rt.T)
#
#df_rt = pd.DataFrame(index=t)
#
## Calculate key statistics
#df_rt['mean']        = results_df.mean(axis=1).values
#df_rt['median']      = results_df.median(axis=1).values
#df_rt['sdv']         = results_df.std(axis=1).values
#df_rt['low_ci_05']   = results_df.quantile(q=0.05, axis=1).values
#df_rt['high_ci_95']  = results_df.quantile(q=0.95, axis=1).values
#df_rt['low_ci_025']  = results_df.quantile(q=0.025, axis=1).values
#df_rt['high_ci_975'] = results_df.quantile(q=0.975, axis=1).values