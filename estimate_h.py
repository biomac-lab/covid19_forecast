from functions.adjust_cases_functions import prepare_cases
from models.seirhd_model import SEIRHD
from models.seird_model import SEIRD


import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd
import numpy as np
import os

from global_config import config


data_dir            = config.get_property('data_dir_covid')
data_dir_mnps       = config.get_property('data_dir_col')
results_dir         = config.get_property('results_dir')
hosp_url            = config.get_property('hosp_url')
geo_dir             = config.get_property('geo_dir')

agglomerated_folder = os.path.join(data_dir, 'data_stages', 'colombia', 'agglomerated', 'geometry')

data = pd.read_csv(os.path.join(agglomerated_folder, 'cases.csv'), parse_dates=['date_time'], dayfirst=True).set_index('poly_id').loc[11001]#.set_index('date_time')
hosp = pd.read_csv(hosp_url, encoding='ISO-8859-1', sep=';', dtype=str, skiprows=4, skipfooter=2, engine='python'
                    ).rename(columns={'Fecha': 'date_time', 'Camas ocupadas COVID 19': 'hospitalized', 'Camas asignadas COVID 19':'total_beds'})
hosp['hospitalized'] = hosp["hospitalized"].apply(lambda x: int(x.replace('.', '')))
hosp['total_beds'] = hosp["total_beds"].apply(lambda x: int(x.replace('.', '')))
hosp["date_time"] =  pd.to_datetime(hosp["date_time"], format='%d/%m/%Y')


data_all = pd.merge(data, hosp, how='outer').set_index('date_time')
data_all_raw = data_all.copy()

# hospitalized cases only available since mid. may...
data_all = data_all.dropna()
# fig, axes = plt.subplots(2,1)
# data_all["num_diseased"].plot(ax=axes[0], color='red', linestyle='--', label='Deaths')
# data_all["num_cases"].plot(ax=axes[1], color='k', linestyle='-', label='Cases')
# ax_tw = axes[1].twinx()
# data_all["hospitalized"].plot(ax=axes[1], color='blue', linestyle='--', label='Hosp')
# axes[0].legend()
# axes[1].legend()
# ax_tw.legend()
# plt.show()



data_all = data_all.resample('D').sum().fillna(0)[['num_cases','num_diseased', 'hospitalized']]
data_all  = prepare_cases(data_all, col='num_cases', cutoff=0)
data_all  = prepare_cases(data_all, col='num_diseased', cutoff=0)
data_all = data_all.rename(columns={'smoothed_num_cases': 'confirmed', 'smoothed_num_diseased':'death'})[['confirmed', 'death', 'hospitalized']]
data_all = data_all.iloc[:-14]
data_all = data_all.iloc[:100]



model = SEIRHD(
    hospitalized     = data_all['hospitalized'].cumsum(),
    confirmed        = data_all['confirmed'].cumsum(),
    death            = data_all['death'].cumsum(),
    T                = len(data_all),
    N                = 8181047,
    )

T_future = 27
path_to_save = os.path.join(results_dir, 'weekly_forecast' , 'bogota', 'hosp_'+pd.to_datetime(data.index.values[-1]).strftime('%Y-%m-%d'))

if not os.path.exists(path_to_save):
    os.makedirs(path_to_save)

samples = model.infer(num_warmup=400, num_samples=2000, num_chains=1)

# In-sample posterior predictive samples (don't condition on observations)
print(" * collecting in-sample predictive samples")
post_pred_samples = model.predictive()
# Forecasting posterior predictive (do condition on observations)
print(" * collecting forecast samples")
forecast_samples = model.forecast(T_future=T_future)

forecast_samples['mean_dz0'] = forecast_samples["dz0"]
forecast_samples['mean_dy0'] = forecast_samples["dy0"]
forecast_samples['mean_dh0'] = forecast_samples["dh0"]

hosp_fitted   = model.combine_samples(forecast_samples, f='mean_dh', use_future=True)
deaths_fitted = model.combine_samples(forecast_samples, f='mean_dz', use_future=True)
cases_fitted  = model.combine_samples(forecast_samples, f='mean_dy', use_future=True)


from functions.samples_utils import create_df_response

df_hosp   = create_df_response(hosp_fitted, time=len(data_all), date_init ='2020-05-15',  forecast_horizon=27, use_future=True)
df_deaths = create_df_response(deaths_fitted, time=len(data_all), date_init ='2020-05-15',  forecast_horizon=27, use_future=True)
df_cases  = create_df_response(cases_fitted, time=len(data_all), date_init ='2020-05-15',  forecast_horizon=27, use_future=True)
from functions.plot_utils import plot_fit

data_all['type'] = 'fitted'
plot_fit(df_deaths, data_all, col_data='death',   y_lim_up = 300, y_label='Deaths', color='indianred', path_to_save='figures/mcmc_2/deaths.png')
plot_fit(df_cases,  data_all, col_data='confirmed',  y_lim_up = 7000,  y_label='Cases', color='darksalmon', path_to_save='figures/mcmc_2/cases.png')
plot_fit(df_hosp, data_all,   col_data='hospitalized',   y_lim_up = 5000, y_label='Hospitalization', color='blue', path_to_save='figures/mcmc_2/hosp.png')



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

