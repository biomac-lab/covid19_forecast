
from matplotlib.dates import date2num, num2date
from matplotlib.colors import ListedColormap
from matplotlib import dates as mdates
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from matplotlib import ticker
from global_config import config

import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd
import numpy as np
import os

from global_config import config


from functions.adjust_cases_functions import prepare_cases
from functions.plot_utils import plot_fit
from global_config import config
from models.seird_model import SEIRModel
from models.seird_model import SEIRD

from datetime import date, timedelta
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

data['type']            = 'fitted'
data.iloc[-14:]['type'] = 'preliminary'


T_future = 28
path_to_checkpoints = os.path.join(results_dir, name_dir, 'checkpoints_agg')

import scipy.io as sio

x_post_forecast = sio.loadmat(os.path.join( path_to_checkpoints, 'forecast_xstates_bog'))['x_forecast']
para_post       = sio.loadmat(os.path.join( path_to_checkpoints, '300_para_post_mean.mat'))['para_post_mean']
x_post          = sio.loadmat(os.path.join( path_to_checkpoints, '300_x_post'))['x_post']


path_to_save = os.path.join(results_dir, 'weekly_forecast' , name_dir,
                            pd.to_datetime(data[data.type=='fitted'].index.values[-1]).strftime('%Y-%m-%d'))

pop = 8181047

parameters_csv =  pd.DataFrame(np.mean(para_post[[0,1,-1],:,:].T, axis=1), columns=['beta_i','beta_a', 'ifr'], index=pd.date_range(start=pd.to_datetime(data[data.type=='fitted'].index.values[0]).strftime('%Y-%m-%d'), periods=para_post.shape[-1]))
parameters_csv.index.name = 'date'
parameters_csv['beta_a'] = parameters_csv['beta_i']*parameters_csv['beta_a']
parameters_csv.to_csv(os.path.join(path_to_save, 'parameters.csv'))


variables_csv =  pd.DataFrame(np.maximum(np.mean(x_post[:7,:,:len(data)].T, axis=1),0), columns=['S','E', 'I', 'A', 'Id', 'cases','deaths'], index=pd.date_range(start=pd.to_datetime(data[data.type=='fitted'].index.values[0]).strftime('%Y-%m-%d'), periods=para_post.shape[-1]))
variables_csv.index.name = 'date'
variables_csv['R'] = pop-variables_csv['S']+variables_csv['E']+variables_csv['A']+variables_csv['I']+variables_csv['Id']#+variables_csv['deaths']
variables_csv = variables_csv[['S', 'E','A', 'I', 'Id', 'deaths','R']]
variables_csv['population'] = pop

variables_csv.to_csv(os.path.join(path_to_save, 'variables.csv'))
variables_csv = variables_csv/pop*100
variables_csv.to_csv(os.path.join(path_to_save, 'variables_percentage.csv'))


recovered = np.squeeze(sio.loadmat(os.path.join(path_to_checkpoints, 'recovered'))['recovered_all'])
recovered = recovered[:, :len(data[data.type=='fitted'])]

def create_df_response(samples, time, date_init ='2020-03-06',  forecast_horizon=27, use_future=False):

    dates_fitted   = pd.date_range(start=pd.to_datetime(date_init), periods=time)
    dates_forecast = pd.date_range(start=dates_fitted[-1]+datetime.timedelta(1), periods=forecast_horizon)
    dates = list(dates_fitted)
    types = ['estimate']*len(dates_fitted)
    if use_future:
        dates += list(dates_forecast)
        types  += ['forecast']*len(dates_forecast)

    results_df  = pd.DataFrame(samples.T)
    df_response = pd.DataFrame(index=dates)

    # Calculate key statistics
    df_response['mean']        = results_df.mean(axis=1).values
    df_response['median']      = results_df.median(axis=1).values
    df_response['std']         = results_df.std(axis=1).values
    df_response['low_975']     = results_df.quantile(q=0.025, axis=1).values
    df_response['high_975']    = results_df.quantile(q=0.975, axis=1).values
    df_response['low_90']      = results_df.quantile(q=0.1, axis=1).values
    df_response['high_90']     = results_df.quantile(q=0.9, axis=1).values
    df_response['low_75']      = results_df.quantile(q=0.25, axis=1).values
    df_response['high_75']     = results_df.quantile(q=0.75, axis=1).values
    df_response['type']        =  types
    df_response.index.name     = 'date'

    return df_response

df_response = create_df_response(np.cumsum(recovered, axis=1)/pop*10, recovered.shape[-1], date_init =pd.to_datetime(data[data.type=='fitted'].index.values[0]).strftime('%Y-%m-%d'))
df_response.to_csv(os.path.join(path_to_save, 'recovered_percentage.csv'))

fig, ax = plt.subplots(1, 1, figsize=(12.5, 7))
ax.plot(df_response.index.values, df_response["mean"], color='teal', alpha=0.4)
ax.fill_between(df_response.index.values, df_response["low_975"], df_response["high_975"], color='teal', alpha=0.6, label='95 % CI')

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
ax.grid(which='major', axis='x', c='k', alpha=.1, zorder=-2)
ax.tick_params(axis='both', labelsize=15)
ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f} %"))
ax.set_ylabel(r'Recovered Fraction $R(t)/N$', fontsize=15)
ax.legend(loc='upper left')
fig.savefig(os.path.join(path_to_save, 'parameters','recovered.png'),  dpi=300,  bbox_inches='tight', transparent=False)
plt.close()