import pandas as pd
import datetime

def create_df_response(samples, time, date_init ='2020-03-06',  forecast_horizon=27, use_future=False):

    dates_fitted   = pd.date_range(start=pd.to_datetime(date_init), periods=time)
    dates_forecast = pd.date_range(start=dates_fitted[-1]+datetime.timedelta(1), periods=forecast_horizon)
    dates = list(dates_fitted)
    types = ['estimate']*len(dates_fitted)
    if use_future:
        dates += list(dates_forecast)
        types  += ['forecast']*len(dates_forecast)

    results_df = pd.DataFrame(samples.T)
    df_response = pd.DataFrame(index=dates)
    # Calculate key statistics
    df_response['mean']        = results_df.mean(axis=1).values
    df_response['median']      = results_df.median(axis=1).values
    df_response['std']         = results_df.std(axis=1).values
    df_response['low_975']      = results_df.quantile(q=0.025, axis=1).values
    df_response['high_975']     = results_df.quantile(q=0.975, axis=1).values

    df_response['low_90']      = results_df.quantile(q=0.1, axis=1).values
    df_response['high_90']     = results_df.quantile(q=0.9, axis=1).values

    df_response['low_75']      = results_df.quantile(q=0.25, axis=1).values
    df_response['high_75']     = results_df.quantile(q=0.75, axis=1).values

    df_response['type']        =  types
    df_response.index.name = 'date'
    return df_response
