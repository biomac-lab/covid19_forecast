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
deaths_post_frcst = sio.loadmat(os.path.join( path_to_checkpoints, 'deaths_post_sim'))['obs_temp_H']
deaths_post_sim   = sio.loadmat(os.path.join( path_to_checkpoints, 'deaths_post_forecast'))['obs_temp_H']
para_post         = sio.loadmat(os.path.join( path_to_checkpoints, '400_para_post_mean.mat'))['para_post_mean']
x_post            = sio.loadmat(os.path.join( path_to_checkpoints, '400_x_post'))['x_post']

deaths_forecast = x_post_frcst[5,:,:]
import seaborn as sns
import itertools
df_deaths = pd.DataFrame(columns=['date','ens_id','value', 'type'])
df_deaths['ens_id'] = list(itertools.chain(* [[id]*deaths_forecast.shape[-1] for id in range(300)]))
df_deaths['date']   = list(pd.date_range(start=pd.to_datetime(data[data.type=='fitted'].index.values[0]).strftime('%Y-%m-%d'), periods=deaths_forecast.shape[-1]))*300
df_deaths['value']  = deaths_forecast.reshape(-1)
df_deaths['type']   = ['forecast']*len(df_deaths)
df_deaths['type'][ df_deaths.date.isin(list(data[data.type=='fitted'].index.values))]   =  'fitted'
df_deaths['value_mod'] = np.nan
df_deaths['value_mod'][df_deaths.type=='forecast'] = df_deaths['value'][df_deaths.type=='forecast']/10
df_deaths['value_mod'][df_deaths.type=='fitted']   = df_deaths['value'][df_deaths.type=='fitted']
sns.lineplot(data=df_deaths, x="date", y="value_mod")
plt.show()


deaths_forecast = x_post_frcst[6,:,:len(data[data.type=='fitted'])]
df_deaths = pd.DataFrame(columns=['date','ens_id','value', 'type'])
df_deaths['ens_id'] = list(itertools.chain(* [[id]*deaths_forecast.shape[-1] for id in range(300)]))
df_deaths['date']   = list(pd.date_range(start=pd.to_datetime(data[data.type=='fitted'].index.values[0]).strftime('%Y-%m-%d'), periods=deaths_forecast.shape[-1]))*300
df_deaths['value']  = deaths_forecast.reshape(-1)
df_deaths['type']   = ['forecast']*len(df_deaths)
df_deaths['type'][ df_deaths.date.isin(list(data[data.type=='fitted'].index.values))]   =  'fitted'
df_deaths['value_mod'] = np.nan
df_deaths['value_mod'][df_deaths.type=='forecast'] = df_deaths['value'][df_deaths.type=='forecast']/300
df_deaths['value_mod'][df_deaths.type=='fitted']   = df_deaths['value'][df_deaths.type=='fitted']

sns.lineplot(data=df_deaths, x="date", y="value_mod")
plt.show()
