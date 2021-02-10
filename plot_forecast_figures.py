from matplotlib.dates import date2num, num2date
from matplotlib import dates as mdates
from matplotlib import ticker
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib import pyplot as plt
from global_config import config




from functions.adjust_cases_functions import prepare_cases
from global_config import config
from seir_model import SEIRModel
from seir_model import SEIRD

import pandas as pd
import numpy as np
import datetime
import os


data_dir            = config.get_property('data_dir_covid')
geo_dir             = config.get_property('geo_dir')
data_dir_mnps       = config.get_property('data_dir_col')
results_dir         = config.get_property('results_dir')

agglomerated_folder = os.path.join(data_dir, 'data_stages', 'colombia', 'agglomerated', 'geometry' )


data =  pd.read_csv(os.path.join(agglomerated_folder, 'cases.csv'), parse_dates=['date_time'], dayfirst=True).set_index('poly_id').loc[11001].set_index('date_time')
data = data.resample('D').sum().fillna(0)[['num_cases','num_diseased']]
data  = prepare_cases(data, col='num_cases', cutoff=0)    # .rename({'smoothed_num_cases':'num_cases'})
data  = prepare_cases(data, col='num_diseased', cutoff=0) # .rename({'smoothed_num_cases':'num_cases'})


data = data.rename(columns={'num_cases': 'confirmed', 'num_diseased':'death'})[['confirmed', 'death']]
data = prepare_cases(data, col='confirmed')
data = prepare_cases(data, col='death')

data = data.iloc[:-14]


T_future = 27
path_to_save = os.path.join(results_dir, 'weekly_forecast' , 'bogota', pd.to_datetime(data.index.values[-1]).strftime('%Y-%m-%d'))


df_deaths = pd.read_csv(os.path.join(path_to_save, 'deaths_df.csv'))
df_deaths['date'] = pd.to_datetime(df_deaths['date'])
df_deaths = df_deaths.set_index('date')

df_cases  = pd.read_csv(os.path.join(path_to_save, 'cases_df.csv')) #.set_index('date')
df_cases['date'] = pd.to_datetime(df_cases['date'])
df_cases = df_cases.set_index('date')

df_deaths_fit = df_deaths.copy(); df_deaths_fit      = df_deaths[df_deaths.type=='estimate']
df_deaths_forecast = df_deaths.copy(); df_deaths_forecast = df_deaths[df_deaths.type=='forecast']

df_cases_fit = df_cases.copy(); df_cases_fit  = df_cases[df_cases.type=='estimate']
df_cases_forecast = df_cases.copy(); df_cases_forecast  = df_cases[df_cases.type=='forecast']


fig, ax = plt.subplots(1, 1, figsize=(15.5, 7))
ax.fill_between(df_deaths_fit.index.values, df_deaths_fit["low_975"], df_deaths_fit["high_975"], color='gray', alpha=0.4, label='95 CI - Nowcast')
ax.plot(df_deaths_fit.index.values, df_deaths_fit["median"], color='black', alpha=0.4, label='Median - Nowcast')
ax.scatter(df_deaths_fit.index.values, data.smoothed_death, facecolor='black', alpha=0.6, edgecolor='black', s=10)
(y1_l, y2_l) = ax.get_ylim()

ax.fill_between(df_deaths_forecast.index.values, df_deaths_forecast.low_90, df_deaths_forecast.high_90, color='green', alpha=0.6, label='4 week forecast')
ax.plot(df_deaths_forecast.index.values, df_deaths_forecast["median"], color='green', alpha=0.4, label='Forecast - Median')
ax.scatter(df_deaths_forecast.index.values, df_deaths_forecast["median"], edgecolor='k', facecolor='white', s=10)#, label='Deaths')

#ax.plot(dates_forecast, median[num_times-1:num_times+num_forecast-1], color='darksalmon', linestyle='--')
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
ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}"))
#ax.axvline(x = 37, linestyle='--',  label = '{}'.format(dates[-1].strftime('%b-%d')))
ax.set_ylabel('Deaths', size=15)

ax.set_ylim((y1_l, 200) )
ax.legend()
fig.savefig(os.path.join(path_to_save, 'death_aggregated_forecast_mcmc.png'), dpi=300, bbox_inches='tight', transparent=False)
fig.savefig(os.path.join('figures', 'mcmc', 'death_aggregated_forecast_mcmc.png'), dpi=300, bbox_inches='tight', transparent=False)
plt.close()


################################################  CASES ################################################

fig, ax = plt.subplots(1, 1, figsize=(15.5, 7))
ax.fill_between(df_cases_fit.index.values, df_cases_fit["low_975"], df_cases_fit["high_975"], color='gray', alpha=0.4, label='95 CI - Nowcast')
ax.plot(df_cases_fit.index.values, df_cases_fit["median"], color='black', alpha=0.4, label='Median - Nowcast')
ax.scatter(df_cases_fit.index.values, data.smoothed_confirmed, facecolor='black', alpha=0.6, edgecolor='black', s=10)
(y1_l, y2_l) = ax.get_ylim()

ax.fill_between(df_cases_forecast.index.values, df_cases_forecast.low_90, df_cases_forecast.high_90, color='darksalmon', alpha=0.6, label='4 week forecast')
ax.plot(df_cases_forecast.index.values, df_cases_forecast["median"], color='darksalmon', alpha=0.4, label='Forecast - Median')
ax.scatter(df_cases_forecast.index.values, df_cases_forecast["median"], edgecolor='k', facecolor='white', s=10)#, label='Deaths')

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
ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}"))
#ax.axvline(x = 37, linestyle='--',  label = '{}'.format(dates[-1].strftime('%b-%d')))
ax.set_ylabel('Cases', size=15)
ax.legend()
ax.set_ylim((y1_l, 9000) )

fig.savefig(os.path.join(path_to_save, 'cases_aggregated_forecast_mcmc.png'), dpi=300, bbox_inches='tight', transparent=False)
fig.savefig(os.path.join('figures', 'mcmc', 'cases_aggregated_forecast_mcmc.png'), dpi=300, bbox_inches='tight', transparent=False)

plt.close()