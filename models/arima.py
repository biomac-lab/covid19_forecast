from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima.model import ARIMA

from tqdm import tqdm
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

class ARIMA_model(object):
    @classmethod
    def fit_arima(cls, data, order):

        model = ARIMA(data, order=order).fit()
        aic   = model.aic
        return (model, aic)

    @classmethod
    def optimize_arima(cls, data, d_values=[1]):
        log_data = data.copy()
        log_data[log_data==0]+=1
        log_data = np.log(log_data)

        acf_d    = acf(log_data.diff().dropna(), nlags=30, alpha=0.05)
        p_values = np.where((acf_d[0]<= acf_d[1][:,0]-acf_d[0]) + (acf_d[0]>= acf_d[1][:,1]-acf_d[0]))[0]

        pacf_d   = pacf(log_data.diff().dropna(), nlags=30, alpha=0.05)
        q_values = np.where((pacf_d[0]<= pacf_d[1][:,0]-pacf_d[0]) + (pacf_d[0]>= pacf_d[1][:,1]-pacf_d[0]))[0]

        best_aic   = 0
        best_model = None
        best_cfg   = None

        for p in tqdm(p_values):
            for d in d_values:
                for q in q_values:
                    order = (p,d,q)
                    model, aic = cls.fit_arima(log_data, order=order)

                    if aic < best_aic:
                        best_aic, best_cfg = aic, order
                        best_model = model

        return (best_model, best_cfg, best_aic)

    def forecast_arima(cls, data=None, T_future=28, quantiles=[50, 80, 95]):
        if data is not None:
            (best_model, best_cfg, best_aic) = cls.optimize_arima(data)

            PredictionResultsWrapper = best_model.get_forecast(steps=T_future, dynamic=True)
            df_result = np.exp(PredictionResultsWrapper.conf_int(alpha=1))
            df_result.columns    = ['mean', 'median']
            df_result.index.name = 'date'

            for quant in quantiles:
                df_ci = np.exp(PredictionResultsWrapper.conf_int(alpha=1-quant/100))
                df_ci.columns = [f'low_{quant}', f'high_{quant}']
                df_ci.index.name = 'date'
                df_result = pd.merge(df_result, df_ci, left_index=True, right_index=True)
        return df_result