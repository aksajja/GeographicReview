Experiments being performed to prove the hypothesis "*Covid is a patchwork pandemic in the US*" for the **Geographic Review**.


>1) Random forest regression on Arizona data. The regression is followed by a prediction over test data for all the states. A series of R^2 values are stored.
>2) Random forest regression on each state over a set of time lags. The regression is followed by a prediction over test data for each instance. R^2 values are tabled.
>3) Vector Auto Regression (VAR) over the 'change in deaths' time series.  Predictions are made over the test data using the fitted model. R^2 values are tabled.
>>   Note - This experiment is run with and without exogenous data ('change in m50_index').
>4) Vector Error Correction Model (VECM) over the 'change in deaths' time series.  Predictions are made for <number of prediction steps(k)> using the fitted model (hence the model has to be fitted repeatedly upto nth day,for predicting k days after nth day ). R^2 values are tabled.
>>    Note - This experiment is run with and without exogenous data ('change in m50_index').
>5) Forecasting change in m50_index using the VAR modeling as in exp3.
>6) Granger causality between the change in mobility and change in deaths over a series of lags for a chosen window.
 
### Steps -

> 1) Manually copy file from [drive](https://drive.google.com/file/d/1vXaYl6PYYkgvOjFPS9bpdr7OoZeoBplC/view?usp=sharing) and save as 'data/United_States_COVID-19_Cases_and_Deaths_by_State_over_Time.csv'.
> 2) Manually copy file from [drive](https://drive.google.com/file/d/1kozpyXDrRZzHhmwNukFqmslu714uOwRf/view?usp=sharing) and save as 'data/Mobility_upto 28-09-2020_DL.xlsx'.
