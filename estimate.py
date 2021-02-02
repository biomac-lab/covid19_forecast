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
data = prepare_cases(data, col='confirmed')
data = prepare_cases(data, col='death')

data = data.iloc[:-11]

model = SEIRD(
    confirmed = data['confirmed'].cumsum(),
    death     = data['death'].cumsum(),
    T         = len(data),
    N         = 8181047,
    )


T_future = 27
path_to_save = os.path.join(results_dir, 'weekly_forecast' , 'bogota', pd.to_datetime(data.index.values[-1]).strftime('%Y-%m-%d'))
if not os.path.exists(path_to_save):
    os.makedirs(path_to_save)

samples = model.infer(num_warmup=400, num_samples=2000)

# In-sample posterior predictive samples (don't condition on observations)
print(" * collecting in-sample predictive samples")
post_pred_samples = model.predictive()
# Forecasting posterior predictive (do condition on observations)
print(" * collecting forecast samples")
forecast_samples = model.forecast(T_future=T_future)

save_fields=['beta0', 'beta', 'sigma', 'gamma', 'dy0', 'dy', 'mean_dy',  'dy_future', 'mean_dy_future',
                'dz0', 'dz', 'dz_future', 'mean_dz', 'mean_dz_future',
                'y0', 'y', 'y_future', 'z0', 'z', 'z_future' ],

def trim(d):
    if d is not None:
        d = {k : v for k, v in d.items() if k in save_fields}
    return d

np.savez_compressed(os.path.join(path_to_save, 'samples.npz'),
                        mcmc_samples = trim(samples),
                        post_pred_samples = trim(post_pred_samples),
                        forecast_samples = trim(forecast_samples))


# Compute growth rate over time
beta  = samples['beta']
sigma = samples['sigma'][:,None]
gamma = samples['gamma'][:,None]
t     = pd.date_range(start=data.index.values[0], periods=beta.shape[1], freq='D')

from seir_model import SEIRModel
import numpy as onp
import datetime

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


t_forecast     = pd.date_range(start=pd.to_datetime(df_rt.index.values[-1])+datetime.timedelta(days=1), periods=T_future, freq='D')

results_df = pd.DataFrame(samples['dz'].T)
df_deaths = pd.DataFrame(index=t)
# Calculate key statistics
df_deaths['deaths_mean']     = results_df.mean(axis=1).values
df_deaths['deaths_median']    = results_df.median(axis=1).values
df_deaths['deaths_Std']      = results_df.std(axis=1).values
df_deaths['low_ci_05']       = results_df.quantile(q=0.05, axis=1).values
df_deaths['high_ci_95']      = results_df.quantile(q=0.95, axis=1).values
df_deaths['low_ci_025']      = results_df.quantile(q=0.025, axis=1).values
df_deaths['high_ci_975']     = results_df.quantile(q=0.975, axis=1).values
df_deaths['type']            = 'estimate'


results_df = pd.DataFrame(forecast_samples['dz_future'].T)
df_deaths_forecast = pd.DataFrame(index=t_forecast)
# Calculate key statistics
df_deaths_forecast['deaths_mean']     = results_df.mean(axis=1).values
df_deaths_forecast['deaths_median']    = results_df.median(axis=1).values
df_deaths_forecast['deaths_Std']      = results_df.std(axis=1).values
df_deaths_forecast['low_ci_05']       = results_df.quantile(q=0.05, axis=1).values
df_deaths_forecast['high_ci_95']      = results_df.quantile(q=0.95, axis=1).values
df_deaths_forecast['low_ci_025']      = results_df.quantile(q=0.025, axis=1).values
df_deaths_forecast['high_ci_975']     = results_df.quantile(q=0.975, axis=1).values
df_deaths_forecast['type']            = 'forecast'


fig, ax = plt.subplots(1, 1, figsize=(15.5, 7))

#ax.plot(pd.date_range(start='2020-03-06', periods=deaths_loc.shape[-1]), np.squeeze(np.median(deaths_loc,1)), color='k')
ax.scatter(data.index.values, data.smoothed_death, facecolor='black', alpha=0.6, edgecolor='black')


ax.fill_between(df_deaths.index.values, df_deaths.low_ci_05, df_deaths.high_ci_95, color='gray', alpha=0.4, label='95 CI - Nowcast')
ax.plot(df_deaths.index.values, df_deaths.deaths_median, color='black', alpha=0.4, label='Median - Nowcast')

ax.fill_between(df_deaths_forecast.index.values, df_deaths_forecast.low_ci_025, df_deaths_forecast.high_ci_975, color='blue', alpha=0.6, label='4 week forecast')
ax.plot(df_deaths_forecast.index.values, df_deaths_forecast.deaths_mean, color='blue', alpha=0.4, label='Forecast - Median')
ax.scatter(df_deaths_forecast.index.values, df_deaths_forecast.deaths_mean, edgecolor='k', facecolor='white')#, label='Deaths')

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
ax.legend()

fig.savefig(os.path.join(path_to_save, 'death_aggregated_forecast_mcmc.png'), dpi=300, bbox_inches='tight', transparent=False)
df_deaths_all = pd.concat([df_deaths, df_deaths_forecast])
df_deaths_all.to_csv(os.path.join(path_to_save, 'deaths_forecast.csv'))
plt.close()


################################################  CASES ################################################
results_df = pd.DataFrame(samples['mean_dy'].T)
df_deaths = pd.DataFrame(index=t)
# Calculate key statistics
df_deaths['cases_mean']     = results_df.mean(axis=1).values
df_deaths['cases_median']    = results_df.median(axis=1).values
df_deaths['cases_Std']      = results_df.std(axis=1).values
df_deaths['low_ci_05']       = results_df.quantile(q=0.05, axis=1).values
df_deaths['high_ci_95']      = results_df.quantile(q=0.95, axis=1).values
df_deaths['low_ci_025']      = results_df.quantile(q=0.025, axis=1).values
df_deaths['high_ci_975']     = results_df.quantile(q=0.975, axis=1).values
df_deaths['type']            = 'estimate'


results_df = pd.DataFrame(forecast_samples['mean_dy_future'].T)
df_deaths_forecast = pd.DataFrame(index=t_forecast)
# Calculate key statistics
df_deaths_forecast['cases_mean']     = results_df.mean(axis=1).values
df_deaths_forecast['cases_median']    = results_df.median(axis=1).values
df_deaths_forecast['cases_Std']      = results_df.std(axis=1).values
df_deaths_forecast['low_ci_05']       = results_df.quantile(q=0.05, axis=1).values
df_deaths_forecast['high_ci_95']      = results_df.quantile(q=0.95, axis=1).values
df_deaths_forecast['low_ci_025']      = results_df.quantile(q=0.025, axis=1).values
df_deaths_forecast['high_ci_975']     = results_df.quantile(q=0.975, axis=1).values
df_deaths_forecast['type']            = 'forecast'

fig, ax = plt.subplots(1, 1, figsize=(15.5, 7))

#ax.plot(pd.date_range(start='2020-03-06', periods=deaths_loc.shape[-1]), np.squeeze(np.median(deaths_loc,1)), color='k')
ax.scatter(data.index.values, data.smoothed_confirmed, facecolor='black', alpha=0.6, edgecolor='black')


ax.fill_between(df_deaths.index.values, df_deaths.low_ci_05, df_deaths.high_ci_95, color='gray', alpha=0.4, label='95 CI - Nowcast')
ax.plot(df_deaths.index.values, df_deaths.cases_median, color='black', alpha=0.4, label='Median - Nowcast')

ax.fill_between(df_deaths_forecast.index.values, df_deaths_forecast.low_ci_025, df_deaths_forecast.high_ci_975, color='green', alpha=0.4, label='4 week forecast')
ax.plot(df_deaths_forecast.index.values, df_deaths_forecast.cases_mean, color='green', alpha=0.4, label='Forecast - Median')
ax.scatter(df_deaths_forecast.index.values, df_deaths_forecast.cases_mean, edgecolor='k', facecolor='white')#, label='Deaths')

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
fig.savefig(os.path.join(path_to_save, 'cases_aggregated_forecast_mcmc.png'), dpi=300, bbox_inches='tight', transparent=False)
df_deaths_all = pd.concat([df_deaths, df_deaths_forecast])
df_deaths_all.to_csv(os.path.join(path_to_save, 'cases_forecast.csv'))
plt.close()
