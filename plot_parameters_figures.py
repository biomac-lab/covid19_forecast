from matplotlib.dates import date2num, num2date
from matplotlib import dates as mdates
from matplotlib import ticker
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib import pyplot as plt
from global_config import config

from functions.adjust_cases_functions import prepare_cases
from seir_model import SEIRD

import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd
import numpy as np
import os

from global_config import config
from functions.samples_utils import create_df_response

data_dir            = config.get_property('data_dir_covid')
geo_dir             = config.get_property('geo_dir')
data_dir_mnps       = config.get_property('data_dir_col')
results_dir         = config.get_property('results_dir')
agglomerated_folder = os.path.join(data_dir, 'data_stages', 'colombia', 'agglomerated', 'geometry' )

data =  pd.read_csv(os.path.join(agglomerated_folder, 'cases.csv'), parse_dates=['date_time'], dayfirst=True).set_index('poly_id').loc[11001].set_index('date_time')
data = data.resample('D').sum().fillna(0)[['num_cases','num_diseased']]
data  = prepare_cases(data, col='num_cases', cutoff=0)    # .rename({'smoothed_num_cases':'num_cases'})
data  = prepare_cases(data, col='num_diseased', cutoff=0) # .rename({'smoothed_num_cases':'num_cases'})

data = data.rename(columns={'smoothed_num_cases': 'confirmed', 'smoothed_num_diseased':'death'})[['confirmed', 'death']]


data = data.iloc[:-14]

model = SEIRD(
    confirmed = data['confirmed'].cumsum(),
    death     = data['death'].cumsum(),
    T         = len(data),
    N         = 8181047,
    )


T_future = 27
path_to_save = os.path.join(results_dir, 'weekly_forecast' , 'bogota', pd.to_datetime(data.index.values[-1]).strftime('%Y-%m-%d'))


def load_samples(filename):

    x = np.load(filename, allow_pickle=True)
    mcmc_samples = x['mcmc_samples'].item()
    post_pred_samples = x['post_pred_samples'].item()
    forecast_samples = x['forecast_samples'].item()

    return mcmc_samples, post_pred_samples, forecast_samples


mcmc_samples, _, _ = load_samples(os.path.join(path_to_save, 'samples.npz'))

beta = mcmc_samples['beta']

beta_df = create_df_response(beta, beta.shape[-1], date_init ='2020-03-06',  forecast_horizon=27, use_future=False)


fig, ax = plt.subplots(1, 1, figsize=(15.5, 7))
ax.plot(beta_df.index.values, beta_df["median"], color='darkred', alpha=0.4, label='Median - Nowcast')
ax.fill_between(beta_df.index.values, beta_df["low_975"], beta_df["high_975"], color='darkred', alpha=0.4, label='95 CI - Nowcast')
ax.fill_between(beta_df.index.values, beta_df["low_75"], beta_df["high_75"], color='darkred', alpha=0.4, label='95 CI - Nowcast')

(y1_l, y2_l) = ax.get_ylim()

#ax.scatter(dates_forecast, median[num_times-1:num_times+num_forecast-1], edgecolor='k', facecolor='white')#, label='Deaths')
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
ax.xaxis.set_minor_locator(mdates.DayLocator())
ax.xaxis.set_major_locator(mdates.WeekdayLocator())
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.grid(which='major', axis='y', c='k', alpha=.1, zorder=-2)
ax.tick_params(axis='both', labelsize=15)
ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))
#ax.axvline(x = 37, linestyle='--',  label = '{}'.format(dates[-1].strftime('%b-%d')))
ax.set_ylabel(r'$\beta(t)$ - Contact Rate', size=15)
fig.savefig(os.path.join('figures', 'mcmc', 'contact_rate.png'), dpi=300, bbox_inches='tight', transparent=False)
plt.close()