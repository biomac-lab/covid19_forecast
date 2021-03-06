from global_config import config


from functions.adjust_cases_functions import prepare_cases
from functions.general_utils import  get_bool

from global_config import config
from models.seird_model import SEIRModel
from models.seird_model import SEIRD

from functions.plot_utils import plot_fit


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
    drop_last_weeks = get_bool(sys.argv[3])

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
if drop_last_weeks:
    print("*** Droping last 2wk ***")
    data.iloc[-14:]['type'] = 'preliminary'


T_future = 28
path_to_save = os.path.join(results_dir, 'weekly_forecast' , name_dir,
                            pd.to_datetime(data[data.type=='fitted'].index.values[-1]).strftime('%Y-%m-%d'))


df_deaths = pd.read_csv(os.path.join(path_to_save, 'deaths_df.csv'))
df_deaths['date'] = pd.to_datetime(df_deaths['date'])
df_deaths = df_deaths.set_index('date')
df_deaths = df_deaths.iloc[:len(data)+T_future]

df_cases  = pd.read_csv(os.path.join(path_to_save, 'cases_df.csv')) #.set_index('date')
df_cases['date'] = pd.to_datetime(df_cases['date'])
df_cases = df_cases.set_index('date')
df_cases = df_cases.iloc[:len(data)+T_future]

d_max = data.smoothed_death.max()      + 10**int(np.log10(data.smoothed_death.max()))
c_max = data.smoothed_confirmed.max()  + 10**int(np.log10(data.smoothed_confirmed.max()))

plot_fit(df_deaths, data, col_data='smoothed_death',   y_lim_up = d_max, y_label='Deaths', color='indianred', path_to_save='figures/mcmc/deaths.png')
plot_fit(df_cases, data, col_data='smoothed_confirmed', y_lim_up = c_max,  y_label='Cases', color='darksalmon', path_to_save='figures/mcmc/cases.png')

os.popen('cp figures/mcmc/deaths.png {}'.format(os.path.join(path_to_save, 'deaths.png')))
os.popen('cp figures/mcmc/cases.png {}'.format(os.path.join(path_to_save, 'cases.png')))

from PIL import Image

im1 = Image.open('figures/mcmc/deaths.png').convert('RGB')
im2 = Image.open('figures/mcmc/cases.png').convert('RGB')

im_list = [im2]

pdf1_filename = "./report_{}_{}.pdf".format(name_dir, pd.to_datetime(data[data.type=='fitted'].index.values[-1]).strftime('%Y-%m-%d'))
im1.save(pdf1_filename, "PDF" ,resolution=100.0, save_all=True, append_images=im_list)