from seir_model import SEIRD



import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd
import numpy as np
import os 
from matplotlib.dates import date2num, num2date
from matplotlib import dates as mdates
from matplotlib import ticker
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib import pyplot as plt
from global_config import config
from functions.adjust_cases_functions import prepare_cases


data_dir            = config.get_property('data_dir_covid')
geo_dir             = config.get_property('geo_dir')
data_dir_mnps       = config.get_property('data_dir_col')
results_dir         = config.get_property('results_dir')
agglomeration_df    = pd.read_csv(os.path.join(data_dir_mnps, 'administrative_division_col_2018.csv')).set_index('poly_id')
agglomerated_folder = os.path.join(data_dir, 'data_stages', 'colombia', 'agglomerated', 'geometry' )

data =  pd.read_csv(os.path.join(agglomerated_folder, 'cases.csv'), parse_dates=['date_time'], dayfirst=True).set_index('poly_id').loc[11001].set_index('date_time')
data = data.resample('D').sum().fillna(0)[['num_cases','num_diseased']]
data  = prepare_cases(data, col='num_cases', cutoff=0)    # .rename({'smoothed_num_cases':'num_cases'})
data  = prepare_cases(data, col='num_diseased', cutoff=0) # .rename({'smoothed_num_cases':'num_cases'})


data = data.rename(columns={'num_cases': 'confirmed', 'num_diseased':'death'})[['confirmed', 'death']]

model = SEIRD(
    confirmed = data['confirmed'].cumsum(),
    death = data['death'].cumsum(),
    T = len(data),
    N = 8181047,
    )

samples = model.infer(num_warmup=400, num_samples=2000)

# Compute growth rate over time
beta  = samples['beta']
sigma = samples['sigma'][:,None]
gamma = samples['gamma'][:,None]
t     = pd.date_range(start=data.index.values[0], periods=beta.shape[1], freq='D')

from models.SEIRD_incident import SEIRModel
import numpy as onp
# compute Rt
rt = SEIRModel.R0((beta, sigma, gamma))
results_df = pd.DataFrame(rt.T)

df_rt = pd.DataFrame(index=t)

# Calculate key statistics
df_rt['Rt_mean']     = results_df.mean(axis=1).values
df_rt['Rt_median']   = results_df.median(axis=1).values
df_rt['Rt_Std']      = results_df.std(axis=1).values
df_rt['low_ci_05']   = results_df.quantile(q=0.05, axis=1).values
df_rt['high_ci_95']  = results_df.quantile(q=0.95, axis=1).values
df_rt['low_ci_025']  = results_df.quantile(q=0.025, axis=1).values
df_rt['high_ci_975'] = results_df.quantile(q=0.975, axis=1).values

