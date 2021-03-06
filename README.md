# Forecasting SARS-CoV-2 dynamics for Bogotá D.C.
Contributors from the BIOMAC lab at Uniandes (listed alphabetically): Carlos Bravo, Jaime Cascante, Juan Cordovez and Mauricio Santos.

## Data Availability
We fork incidence data from the [DataLama](http://datalama.polyglot.site/) lab at URosario. For any other info about data please contact them. 

## Models Card
All the mathematical models we use infers past SARS-CoV-2 transmission rates (different assumptions and methods on the inference) and can be used to forecast community spread, and mortality.

## Forecast
We present forecasts in 4 week-format daily cases and deaths horizon.

![Cases Forecast](/figures/mcmc/cases.png "Cases Forecast")
![Deaths Forecast](/figures/mcmc/deaths.png "Deaths Forecasts")

## Recovered Population
We fit a mechanistic model with an Ensemble Adjustment Kalman Filter (EAKF) in the iterative filtering (IF) framework for estimate past symptomatic/reported and asymptomatic/unreported transmission rates and current state variables. Below we present the estimated fraction of recovered individuals. We compare (not fit) our model to the estimated total number of past infected individuals as of November 15th of the [seroprevalence study](https://www.ins.gov.co/estudio-nacional-de-seroprevalencia/reporte.html) from the national health institute (INS).

![Estimated fraction of recovered individuals](/figures/eakf/recovered.png "Contact rate")

## Parameter Estimates
1. Time variable contact rate $\beta (t)$
    ![Time Variable Contact Rate](/figures/mcmc/contact_rate.png "Contact rate")



## References
[1] Ray, E. L., Wattanachit, N., Niemi, J., Kanji, A. H., House, K., Cramer, E. Y., Bracher, J., Zheng, A., Yamana, T. K., Xiong, X., Woody, S., Wang, Y., Wang, L., Walraven, R. L., Tomar, V., Sherratt, K., Sheldon, D., Reiner, R. C., Prakash, B. A., … Reich, N. G. (2020). Ensemble Forecasts of Coronavirus Disease 2019 (COVID-19) in the U.S. MedRxiv, 2020.08.19.20177493. https://doi.org/10.1101/2020.08.19.20177493

[2] Gibson, G. C., Reich, N. G., & Sheldon, D. (2020). Real-time mechanistic bayesian forecasts of covid-19 mortality. 1–33.

[3] Anderson, J. L. (2001). An Ensemble Adjustment Kalman Filter for Data Assimilation. Australian Meteorological Magazine, 129(2), 2884–2887.

[4] Ionides, E. L., Bretó, C., & King, A. A. (2006). Inference for nonlinear dynamical systems. Proceedings of the National Academy of Sciences of the United States of America, 103(49), 18438–18443. https://doi.org/10.1073/pnas.0603181103

[5] King AA, Nguyen D and Ionides EL (2015) Statistical inference for partially observed Markov processes via the R package pomp. arXiv preprint arXiv:1509.00503.
