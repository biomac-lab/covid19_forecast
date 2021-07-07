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


poly_run  = 11001
name_dir  = 'bogota'

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


x_post_frcst      = sio.loadmat(os.path.join( path_to_checkpoints, 'xpost_forecast'))['x_forecast']
obs_post_frcst    = sio.loadmat(os.path.join( path_to_checkpoints, 'obs_post_forecast'))['obs_temp']
deaths_post_frcst = sio.loadmat(os.path.join( path_to_checkpoints, 'deaths_post_forecast'))['obs_temp_H']
para_post         = sio.loadmat(os.path.join( path_to_checkpoints, '100_para_post_mean.mat'))['para_post_mean']
x_post            = sio.loadmat(os.path.join( path_to_checkpoints, '20_x_post'))['x_post']

deaths_forecast = x_post[6,:,:] #deaths_post_frcst
obs_forecast    = x_post[5,:,:] #obs_post_frcst

import seaborn as sns
import itertools
def create_sample_response(samples, num_times, min_date):
    df_samples_response = pd.DataFrame(columns=['date','ens_id','value', 'type'])
    df_samples_response['ens_id'] = list(itertools.chain(* [[id]*samples.shape[-1] for id in range(300)]))
    df_samples_response['date']   = list(pd.date_range(start=min_date, periods=samples.shape[-1]))*300
    df_samples_response['value']  = samples.reshape(-1)
    df_samples_response['type']   = ['forecast']*len(df_samples_response)
    df_samples_response['type'][ df_samples_response.date.isin(list(data[data.type=='fitted'].index.values))]   =  'fitted'
    df_samples_response['value_mod'] = np.nan
    df_samples_response['value_mod'][df_samples_response.type=='forecast'] = df_samples_response['value'][df_samples_response.type=='forecast']
    df_samples_response['value_mod'][df_samples_response.type=='fitted']   = df_samples_response['value'][df_samples_response.type=='fitted']

    return df_samples_response

min_date  = pd.to_datetime(data[data.type=='fitted'].index.values[0]).strftime('%Y-%m-%d')
num_times = x_post.shape[-1]

df_deaths = create_sample_response(deaths_forecast, num_times=num_times, min_date=min_date)
df_cases  = create_sample_response(obs_forecast, num_times=num_times, min_date=min_date)

fig, ax = plt.subplots(1, 1, figsize=(15.5, 7.2))
sns.lineplot(ax=ax, data=df_deaths[df_deaths.type=='fitted'], x="date", y="value_mod")
data_p = data[data.type=='fitted']
ax.scatter(data_p.index.values, data_p.smoothed_death, edgecolor='black', facecolor='red')
ax.set_ylim([0, 200])
plt.show()


fig, ax = plt.subplots(1, 1, figsize=(15.5, 7.2))
sns.lineplot(ax=ax, data=df_cases[df_cases.type=='fitted'], x="date", y="value_mod")
data_p = data[data.type=='fitted']
ax.scatter(data_p.index.values, data_p.smoothed_confirmed, edgecolor='black', facecolor='red')
ax.set_ylim([0, 6000])
plt.show()

