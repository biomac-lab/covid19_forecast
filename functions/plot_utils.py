from matplotlib.dates import date2num, num2date
from matplotlib.colors import ListedColormap
from matplotlib import dates as mdates
from matplotlib.patches import Patch
from matplotlib import pyplot as plt
from matplotlib import ticker

import os
def plot_fit(df_fit, df_data, y_label='Deaths', y_lim_up = 200, color='blue', col_data='smoothed_death', col_up='high_95', col_down='low_95', col_point='median', ax=None,   forecast=True, path_to_save=None):
    """ df_fit with columns:
            'mean', 'median', 'std', 'low_95', 'high_95', 'low_80', 'high_80', 'low_50', 'high_50', 'type'
            type in ['estimate', 'forecast']

        df_data with columns:
                    'confirmed', 'death', 'smoothed_confirmed', 'smoothed_death', 'type'
                    type in ['fitted', 'preliminary']
    """

    df_estimate = df_fit.copy(); df_estimate = df_estimate[df_estimate.type=='estimate']
    df_forecast = df_fit.copy(); df_forecast = df_forecast[df_forecast.type=='forecast']

    df_data_fitted = df_data.copy(); df_data_fitted = df_data_fitted[df_data_fitted.type=='fitted']
    df_data_preliminary = df_data.copy(); df_data_preliminary = df_data_preliminary[df_data_preliminary.type=='preliminary']

    fig, axes = plt.subplots(1, 2, figsize=(20, 7), sharey=True)
    axes[0].fill_between(df_estimate.index.values, df_estimate[col_down], df_estimate[col_up], color='gray', alpha=0.4, label='95 CI - Nowcast')
    axes[0].plot(df_estimate.index.values, df_estimate[col_point], color='black', alpha=0.4, label='Median - Nowcast')

    axes[0].scatter(df_data_fitted.index.values, df_data_fitted[col_data], facecolor='black', alpha=0.6, edgecolor='black', s=30)
    (y1_l, y2_l) = axes[0].get_ylim()

    axes[0].fill_between(df_forecast.index.values, df_forecast[col_down], df_forecast[col_up], color=color, alpha=0.6, label='95% CI')
    axes[0].fill_between(df_forecast.index.values, df_forecast['low_80'], df_forecast['high_80'], color=color, alpha=0.4, label='80% CI')
    axes[0].fill_between(df_forecast.index.values, df_forecast['low_50'], df_forecast['high_50'], color=color, alpha=0.4, label='50% CI')

    axes[0].plot(df_forecast.index.values, df_forecast[col_point], color=color, alpha=0.4, label='Forecast - Median')
    axes[0].scatter(df_forecast.index.values, df_forecast[col_point], edgecolor='k', facecolor='white', s=10)
    axes[0].tick_params(axis='both', labelsize=15)

    axes[1].fill_between(df_estimate.iloc[-10:].index.values, df_estimate.iloc[-10:][col_up], df_estimate.iloc[-10:][col_down], color='gray', alpha=0.4)
    axes[1].plot(df_estimate.iloc[-10:].index.values, df_estimate.iloc[-10:][col_point], color='black', alpha=0.4)
    axes[1].fill_between(df_forecast.index.values, df_forecast[col_down], df_forecast[col_up], color=color, alpha=0.2, label='90% CI')
    axes[1].fill_between(df_forecast.index.values, df_forecast['low_80'], df_forecast['high_80'], color=color, alpha=0.4, label='80% CI')
    axes[1].fill_between(df_forecast.index.values, df_forecast['low_50'], df_forecast['high_50'], color=color, alpha=0.6, label='50% CI')

    axes[1].plot(df_forecast.index.values, df_forecast[col_point], color='black', alpha=0.4)
    axes[1].scatter(df_estimate.iloc[-10:].index.values, df_data_fitted.iloc[-10:][col_data], facecolor='black', alpha=0.6, edgecolor='black', s=50)
    axes[1].scatter(df_data_preliminary.index.values, df_data_preliminary[col_data], edgecolor='k', facecolor='red', s=50, label='Preliminary data')


    for ax in axes:
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
        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}"))
        ax.set_ylabel(y_label, size=15)
        ax.set_ylim( (y1_l, y_lim_up) )
        ax.legend(loc='upper left')

    axes[1].xaxis.set_major_locator(mdates.MonthLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
    axes[1].xaxis.set_minor_locator(mdates.DayLocator())
    axes[1].xaxis.set_minor_formatter(mdates.DateFormatter('%d'))

    axes[1].xaxis.set_major_locator(mdates.WeekdayLocator())
    axes[1].xaxis.set_major_locator(mdates.MonthLocator())
    axes[1].tick_params(which='both', axis='x', labelrotation=90, labelsize=15)
    axes[1].grid(which='both', axis='x', c='k', alpha=.1, zorder=-2)
    axes[0].grid(which='major', axis='x', c='k', alpha=.1, zorder=-2)
    plt.tight_layout()

    if path_to_save:
        fig.savefig(path_to_save, dpi=300, bbox_inches='tight', transparent=False)
    plt.close()