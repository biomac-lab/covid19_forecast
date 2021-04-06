from matplotlib.dates import date2num, num2date
from matplotlib.colors import ListedColormap
from matplotlib import dates as mdates
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from matplotlib import ticker

from functions.adjust_cases_functions import prepare_cases
from functions.general_utils import  get_bool
from models.seird_model import SEIRD

import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd
import numpy as np
import os

from global_config import config

import sys

dict_map = {'0-39': (0, 39), '40-49': (40,49),
            '50-59': (50,59), '60-69': (60,69), '70-90+': (70,200) }

NGroups  = len(dict_map)

def age_group(val, dict_map):
    for ag in dict_map:
        if dict_map[ag][0] <= val <= dict_map[ag][1]:
            return ag
    return 'NaN'

poly_run  = 13001

data_dir            = config.get_property('data_dir_covid')
geo_dir             = config.get_property('geo_dir')
data_dir_mnps       = config.get_property('data_dir_col')
results_dir         = config.get_property('results_dir')

agglomerated_folder = os.path.join(data_dir, 'data_stages', 'colombia', 'agglomerated', 'geometry' )

polygons = pd.read_csv(os.path.join(agglomerated_folder, 'polygons.csv')).set_index('poly_id')
polygons = polygons.loc[poly_run]

data  =  pd.read_csv(os.path.join(agglomerated_folder, 'cases.csv'), parse_dates=['date_time'],
                    dayfirst=True).set_index('poly_id').loc[poly_run].set_index('date_time')

data  = data.resample('D').sum().fillna(0)[['num_cases','num_diseased']]
data  = prepare_cases(data, col='num_cases', cutoff=0)    # .rename({'smoothed_num_cases':'num_cases'})
data  = prepare_cases(data, col='num_diseased', cutoff=0) # .rename({'smoothed_num_cases':'num_cases'})
data  = data.rename(columns={'smoothed_num_cases': 'confirmed', 'smoothed_num_diseased':'death'})[['confirmed', 'death']]


raw_folder = os.path.join(data_dir, 'data_stages', 'colombia', 'raw', 'cases' )
cases_raw_df = pd.read_csv(os.path.join(raw_folder, 'cases_raw.csv'), parse_dates =['Fecha de inicio de síntomas', 'Fecha de diagnóstico', 'Fecha de muerte'], dayfirst=True) #.set_index('poly_id')
cases_raw_df['age_group'] = cases_raw_df['Edad'].apply(lambda x: age_group( x, dict_map) )
cases_raw_df = cases_raw_df[['Código DIVIPOLA municipio', 'Nombre municipio', 'Nombre departamento',  'age_group', 'Sexo' ,'Fecha de inicio de síntomas', 'Fecha de diagnóstico', 'Fecha de muerte']]
cases_raw_df = cases_raw_df.rename(columns={'Código DIVIPOLA municipio': 'poly_id'})
list_df_ages = []
for age_g in dict_map.keys():
    cases_agei = cases_raw_df[cases_raw_df.age_group==age_g].copy()
    cases_agei['num_cases']    = 1
    cases_agei['num_diseased'] = 1

    cases_agei_num_cases  = cases_agei.copy().groupby(['Fecha de diagnóstico','poly_id']).sum().reset_index().rename(columns={'Fecha de diagnóstico': 'date_time'})[['date_time','poly_id','num_cases']]
    cases_agei_num_deaths = cases_agei.copy()[['Fecha de muerte','poly_id','num_diseased']].dropna().groupby(['Fecha de muerte','poly_id']).sum().reset_index().rename(columns={'Fecha de muerte': 'date_time'})
    new_df = pd.merge(cases_agei_num_cases, cases_agei_num_deaths,  how='outer').fillna(0)
    new_df = new_df.groupby(['date_time','poly_id']).sum().reset_index().set_index('poly_id')

    new_df  = prepare_cases(new_df, col='num_cases', cutoff=0)    # .rename({'smoothed_num_cases':'num_cases'})
    new_df  = prepare_cases(new_df, col='num_diseased', cutoff=0) # .rename({'smoothed_num_cases':'num_cases'})
    new_df  = new_df.rename(columns={'smoothed_num_cases': 'confirmed', 'smoothed_num_diseased':'death'})[['date_time','confirmed', 'death']]
    new_df['age_group'] = age_g
    list_df_ages.append(new_df)

cases_df  =  pd.read_csv(os.path.join(agglomerated_folder, 'cases.csv'), parse_dates=['date_time'],
                    dayfirst=True)
cases_df_agg = cases_df.reset_index()[['poly_id','date_time', 'num_cases', 'num_diseased']].rename(columns={'num_cases': 'confirmed', 'num_diseased':'death'})
cases_df_agg['age_group'] = 'agg'
list_df_ages.append(cases_df_agg.set_index('poly_id'))
df_cases_ages = pd.concat(list_df_ages)



df_cases_ages = df_cases_ages.loc[poly_run]
import seaborn as sns
fig, ax = plt.subplots(1, 1, figsize=(12.5, 7))

sns.lineplot(ax=ax, data=df_cases_ages[df_cases_ages.age_group!='agg'], x='date_time', y='confirmed', hue='age_group', palette='flare')
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
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(12.5, 7))
sns.lineplot(ax=ax, data=df_cases_ages[df_cases_ages.age_group!='agg'], x='date_time', y='death', hue='age_group', palette='crest')
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
plt.show()

movement_df = pd.read_csv( os.path.join(agglomerated_folder, 'movement_range.csv' ), parse_dates=['date_time'])
movement_df = movement_df[movement_df.poly_id==poly_run]
movement_df['movement_change'] = 100*movement_df['movement_change']
fig, ax = plt.subplots(1, 1, figsize=(12.5, 7))
sns.lineplot(ax=ax, data=movement_df, x='date_time', y='movement_change', color='k')
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%y'))
ax.xaxis.set_minor_locator(mdates.DayLocator())
ax.xaxis.set_major_locator(mdates.WeekdayLocator())
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.grid(which='major', axis='y', c='k', alpha=.1, zorder=-2)
ax.tick_params(axis='both', labelsize=15)
ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f} %"))
plt.show()
