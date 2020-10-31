import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import pickle
from typing import List
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests

from utils import merge_county_data, extract_max_lag_scores

def get_processed_input(filename: str, lag: int) -> List[pd.DataFrame]:
    data=pd.read_csv(filename,index_col=0)
    data=data.interpolate(method='linear', limit_direction='forward', axis=0)   # Interpolation to handle missing values.
    data=data.drop(columns=['state'])
    Label=data['tot_death'].to_frame()
    m50_df=pd.DataFrame(data['m50'])
    m50_index_df=pd.DataFrame(data['m50_index'])
    
    m50_df=m50_df.diff().iloc[1:]   # Take diff and drop first row.
    m50_index_df=m50_index_df.diff().iloc[1:]   # Take diff and drop first row.
    Y=Label-Label.shift(lag)      # Eg - This is the difference b/w labels, tot_deaths, of day1 and day18.
    Y.drop(Y.head(lag).index,inplace=True)    # Drop head elements = lag_order. 
    data.drop(data.tail(lag).index,inplace=True)  # Drop tail elements = lag_order.
    m50_df.drop(m50_df.tail(lag-1).index,inplace=True)
    m50_index_df.drop(m50_index_df.tail(lag-1).index,inplace=True)
    data=data.drop(columns=['m50','m50_index'])
    data['m50']=m50_df.values.tolist()
    data['m50'] = data['m50'].str.get(0)
    data['m50_index']=m50_df.values.tolist()
    data['m50_index'] = data['m50_index'].str.get(0)
    data=data.drop(columns=['tot_death'])
    
    return data,Y

def get_data(state_data_dir: str, chosen_data='tot_death'):
    if Path(f'data/{chosen_data}_diff_combined_state_level.csv').exists():
        with open(f'data/{chosen_data}_diff_combined_state_level.csv','r') as combined_state_level:
            return pd.read_csv(combined_state_level,index_col=0)
    else:
        state_files = os.listdir(state_data_dir)
        cols = [_state_file.split('.')[0] for _state_file in state_files]
        endog_df = pd.DataFrame()
        for _state_file in state_files:   # state_files contains state filenames.
            filename = state_data_dir+'/'+_state_file
            data=pd.read_csv(filename,index_col=0)
            data=data.interpolate(method='linear', limit_direction='forward', axis=0)   # Interpolation to handle missing values.
            if len(endog_df.index)<1:   # Insert index.
                endog_df = pd.DataFrame(index=data.index,columns=cols)
            endog_df[_state_file.split('.')[0]]=data[chosen_data].values
        # Use change in tot_deaths. Drop first row with NaN.
        endog_df = endog_df.diff().iloc[1:]
        with open(f'data/{chosen_data}_diff_combined_state_level.csv','w') as combined_state_level:
            endog_df.to_csv(combined_state_level)
    return endog_df

def fit_ar_model():
    pass

def store_exp_results(fit_score: dict, has_exog: bool, exp_name: str) -> None:
    print(f'storing {exp_name} results ....')
    if 'granger_' in exp_name: 
        with open(f'data/results/{exp_name}.csv','w') as results_file:
            pd.DataFrame.from_dict(fit_score,orient='index').to_csv(results_file)
    elif has_exog:
        with open(f'data/results/{exp_name}_w_exog.csv','w') as results_file:
            pd.DataFrame.from_dict(fit_score).to_csv(results_file)
    else:
        with open(f'data/results/{exp_name}_wo_exog.csv','w') as results_file:
            pd.DataFrame.from_dict(fit_score).to_csv(results_file)

def fit_var_model(endog_data: pd.DataFrame, exp_name: str, max_lags=None, exog_data=None):
    from statsmodels.tsa.vector_ar.var_model import VAR
    # Hardcoded values to have consistency in models.
    train_pct = 0.9
    steps = 1
    total_samples = len(endog_data.index)
    training_sample_size = int(train_pct*total_samples)
    test_samples = total_samples - training_sample_size
    
    endog_train_data = endog_data.iloc[:training_sample_size]
    exog_train_data = None
    if exog_data is not None:
        exog_train_data = exog_data.iloc[:training_sample_size]

    test_data = endog_data.iloc[training_sample_size:]
    # print('total samples: ',total_samples,' train samples: ',endog_train_data.shape)
    var_model = VAR(endog_train_data,exog=exog_train_data)
    var_model_fit = var_model.fit(maxlags=max_lags,ic='hqic')
    # print(f'MaxLags:{max_lags}, Fitted lag order:{var_model_fit.k_ar}, Non-zero coefs:{var_model_fit.coefs}')
    lag_order = var_model_fit.k_ar
    num_endog_vars = len(endog_data.columns)
    predictions = np.empty((0,num_endog_vars), float)
    for i in range(test_samples):
        if exog_data is not None:
            exog_start_index = training_sample_size+(i-lag_order)
            exog_end_index = exog_start_index+1
            exog_input = exog_data.iloc[exog_start_index:exog_end_index]
        else:
            exog_input = None
        if i<lag_order:
            _forecast_input = np.concatenate((endog_train_data.iloc[i-lag_order:],test_data.iloc[:i,:]),axis=0)
        else:
            _forecast_input = np.asarray(test_data.iloc[i-lag_order:i])
        predictions = np.append(predictions,var_model_fit.forecast(_forecast_input,steps=steps,exog_future=exog_input),axis=0)
    
    predictions_df = pd.DataFrame(predictions,columns=test_data.columns)
    fit_score = {'states':[],'r2':[]}
    for _state in endog_data.columns:
        fit_score['states'].append(_state)
        fit_score['r2'].append(r2_score(test_data[_state],predictions_df[_state]))
    
    store_exp_results(fit_score, exog_data is not None, exp_name)
    return var_model_fit, predictions, fit_score

def fit_vecm_model(endog_data: pd.DataFrame, exp_name: str, exog_data=None):
    from statsmodels.tsa.vector_ar.vecm import VECM
    
    # Hardcoded values to have consistency in models.
    train_pct = 0.9
    prediction_steps = 1
    total_samples = len(endog_data.index)
    training_sample_size = int(train_pct*total_samples)
    test_sample_size = total_samples - training_sample_size

    test_data = endog_data.iloc[training_sample_size:]
    num_endog_vars = len(endog_data.columns)
    predictions = np.empty((0,num_endog_vars), float)

    # Can predict <prediction_steps> into the future after fitting. 
    for _test_idx in range(test_sample_size):
        endog_train_data = endog_data.iloc[:training_sample_size+_test_idx]
        exog_train_data = None
        exog_test_data = None
        if exog_data is not None:
            exog_train_data = exog_data.iloc[:training_sample_size+_test_idx]
            exog_test_data = test_data[:_test_idx+1]

        vecm_model = VECM(endog_train_data,exog=exog_train_data)
        vecm_model_fit = vecm_model.fit()
        lag_order = vecm_model_fit.k_ar     # Unused
        predictions = np.append(predictions,vecm_model_fit.predict(prediction_steps,exog_fc=exog_test_data),axis=0)     # Confidence interval with <alpha>=0.05.
    fit_score = {'states':[],'r2':[]}
    for _state in endog_data.columns:
        y_obs = test_data[:test_sample_size][_state].to_numpy()
        y_pred = predictions[:,list(endog_data.columns).index(_state)]
        fit_score['states'].append(_state)
        fit_score['r2'].append(r2_score(y_obs,y_pred))

    store_exp_results(fit_score, exog_data is not None, exp_name)
    return vecm_model_fit, predictions, fit_score

def fit_rf_model(state_data_dir: str, model_state: str, filename: str, lag: int):
    X, Y= get_processed_input(state_data_dir+f'/{model_state}.csv', lag)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=42)
    regr_model = RandomForestRegressor(max_depth=2, random_state=0)
    regr_model.fit(X_train, y_train)
    # Compute r2 error on test data.
    y_pred=regr_model.predict(X_test)
    print(f'Test error on {model_state} fitted model & lag {lag}: ',r2_score(y_test, y_pred))
    # Store regression model.
    pickle.dump(regr_model, open(filename, 'wb'))
    
    return regr_model

## Following function to be implemented to call regression (VAR/VECM/RF) based on cmd line argument.
# def fit_model(model_type: str):
#     if regr_model_type=='random_forest':
#         return fit_rf_model()
#     elif regr_model_type=='var':
#         return fit_var_model(data, 8)  # ToDo: Remove hardcoded values.
#     elif regr_model_type=='vecm':
#         return fit_vecm_model(data)


def load_model(model_dir: str,state_data_dir: str, model_state: str, regr_model_type: str, lag: int, exog_series=None):
    filename = f'{model_dir}/{regr_model_type}/{model_state}/{model_state}_lag_{lag}.sav'
    model_dir_struct = filename.split('/')
    for depth in range(1,len(model_dir_struct)):  # List contains only folders.
        _dir = '/'.join(model_dir_struct[:depth])
        if not os.path.exists(_dir):
            os.makedirs(_dir)
    if Path(filename).exists():
        return pickle.load(open(filename, 'rb'))
    elif regr_model_type=='random_forest':
        regr_model = fit_rf_model(state_data_dir,model_state,filename,lag)
        return regr_model
    elif regr_model_type=='arima':
        endog_data = get_data(state_data_dir)[model_state]
        if exog_series is not None:
            exog_data = get_data(state_data_dir,chosen_data=exog_series)[model_state]
        else:
            exog_data = None
        regr_model = fit_arima_model(endog_data,f'one_to_all_{regr_model_type}',exog_data=exog_data)    # Fitted to Initial training data.
        return regr_model

def fit_arima_model(model_state: str, endog_data: pd.DataFrame, exp_name: str, exog_data=None):
    from statsmodels.tsa.arima.model import ARIMA
    # Hardcoded values to have consistency in models.
    train_pct = 0.9
    steps = 1
    all_states = endog_data.columns
    total_samples = len(endog_data.index)
    training_sample_size = int(train_pct*total_samples)
    test_samples = total_samples - training_sample_size
    
    num_endog_vars = len(endog_data.columns)
    endog_train_data = endog_data.iloc[:training_sample_size]
    exog_train_data = None
    if exog_data is not None:
        exog_train_data = exog_data.iloc[:training_sample_size]

    test_data = endog_data.iloc[training_sample_size:]
    history = endog_train_data.copy()
    for _col in history.columns:
        history[_col].values[:training_sample_size]=history[model_state].values[:training_sample_size]
    history.index = pd.DatetimeIndex(history.index).to_period('D')
    predictions = pd.DataFrame(columns=endog_train_data.columns)
    for t in range(len(test_data)):
        each_day_predictions = []
        obs = []
        for _state in all_states:   # state_files contains state filenames.
            model = ARIMA(history[_state],exog=exog_train_data)
            model_fit = model.fit()
            output = model_fit.forecast(steps=steps)
            yhat = output.iloc[0]
            each_day_predictions.append(yhat)
        history.loc[history.index[-1]+pd.offsets.Day(1)]=test_data.iloc[t]
        predictions.loc[history.index[-1]+pd.offsets.Day(1)] = each_day_predictions
        
    return predictions,test_data

def fit_one_predict_all(state_data_dir: str, model_state: str, chosen_metric, lag=17, regr_model_type = 'random_forest',exog_series=None):
    r2_dict = {}
    if regr_model_type=='random_forest':
        loaded_model = load_model(model_dir,state_data_dir,model_state,regr_model_type,lag)
        # Use the loaded model to predict for all states.
        state_files = os.listdir(state_data_dir)
        for _state_file in state_files:   # state_files contains state filenames.
            X, Y= get_processed_input(state_data_dir+'/'+_state_file, lag)
            result = loaded_model.predict(X)
            if chosen_metric=='r2':
                corr_metric_vals = r2_score(Y, result)
            elif chosen_metric=='pearsonr':
                corr_metric_vals = pearsonr(Y['tot_death'], result)[0]
            r2_dict[_state_file.split('.')[0]]=corr_metric_vals
    elif regr_model_type=='arima':
        endog_data = get_data(state_data_dir) # ToDo: Don't use default selection of endog_data.
        exog_data = None
        if exog_series is not None:
            exog_data = get_data(state_data_dir,chosen_data=exog_series)
        predictions,Y = fit_arima_model(model_state, endog_data,f'fit_one_to_all_{regr_model_type}',exog_data)
        
        for _state in endog_data.columns:
            if chosen_metric=='r2':
                corr_metric_vals = r2_score(Y[_state], predictions[_state])
            elif chosen_metric=='pearsonr':
                corr_metric_vals = pearsonr(Y[_state], predictions[_state])[0]
            r2_dict[_state]=corr_metric_vals
    
    Final=pd.DataFrame.from_dict({'State':list(r2_dict.keys()),f'{chosen_metric}_error':list(r2_dict.values())})
    result_location = f'data/results/fit_one_to_all_{regr_model_type}_{chosen_metric}.csv'
    Final.to_csv(result_location)
    print(f'Stored results to {result_location}')

def fit_each_w_lags(state_data_dir: str, chosen_metric, lags_list: List[int]):
    regr_model_type = 'random_forest'
    state_files = os.listdir(state_data_dir)
    cols = [f'Lag_{_lag}_{chosen_metric}' for _lag in lags_list]
    final_df = pd.DataFrame(columns=cols)
    for _state_file in state_files:   # state_files contains state filenames.
        model_state = _state_file.split('.')[0]
        r2_dict = {}
        for _lag in lags_list:
            X, Y= get_processed_input(state_data_dir+'/'+_state_file,_lag)
            loaded_model = load_model(model_dir,state_data_dir,model_state,regr_model_type,_lag)
            result = loaded_model.predict(X)
            if chosen_metric=='r2':
                corr_metric_vals = r2_score(Y, result)
            elif chosen_metric=='pearsonr':
                corr_metric_vals = pearsonr(Y['tot_death'], result)[0]
            r2_dict[_lag]=corr_metric_vals
        final_df.loc[_state_file.split('.')[0]]=list(r2_dict.values())
    final_df.to_csv(f'data/results/exp2_{chosen_metric}.csv')
    extract_max_lag_scores(chosen_metric)

def setup_data_dir():
    # Setup results directory.
    results_dir='data/results/'
    dir_struct = results_dir.split('/')
    for depth in range(1,len(dir_struct)):  # List contains only folders.
        _dir = '/'.join(dir_struct[:depth])
        if not os.path.exists(_dir):
            os.makedirs(_dir)

    # Setup state directory.
    state_data_dir = 'data/state_level'
    location = state_data_dir
    model_dir_struct = location.split('/')
    for depth in range(1,len(model_dir_struct)+1):  # List contains only folders.
        _dir = '/'.join(model_dir_struct[:depth])
        if not os.path.exists(_dir):
            os.makedirs(_dir)
    merge_county_data(state_data_dir)

model_dir = 'data/regr_models'
state_data_dir = 'data/state_level'

setup_data_dir()

## Uncomment to run an experiment.
chosen_metric = 'pearsonr'
chosen_model_state = 'Arizona'
# chosen_metric = 'r2'
lags_list = [_lag for _lag in range(1,21)]
# fit_one_predict_all(state_data_dir, 'Arizona', chosen_metric)   # Experiment 1
# fit_each_w_lags(state_data_dir,chosen_metric,lags_list)   # Experiment 2

# Experiment 3 endog_data is change in tot_death.
def run_exp3(endog_series: str, exog_series=None,chosen_states=None):
    endog_data = get_data(state_data_dir = 'data/state_level',chosen_data=endog_series)
    if exog_series is not None:
        exog_data = get_data(state_data_dir = 'data/state_level',chosen_data=exog_series)
    else:
        exog_data = None
    if chosen_states is not None:
        endog_data = endog_data[chosen_states]
        if exog_series is not None:
            exog_data = exog_data[chosen_states]

    lags_list = [_lag for _lag in range(21,1,-1)]
    var_model_fit = None
    for _lag in lags_list:
        try:
            var_model_fit, predictions, fit_score = fit_var_model(endog_data,f'var_{endog_series}',_lag,exog_data)     
            break
        except:
            continue

    if var_model_fit is not None:
        print('Var model fits with lag: ',var_model_fit.k_ar)
    else:
        print('Var model does not fit.')

# Experiment 4 endog_data is change in tot_death.
def run_exp4(endog_series: str, exog_series=None,chosen_states=None):
    endog_data = get_data(state_data_dir = 'data/state_level',chosen_data=endog_series)
    if exog_series is not None:
        exog_data = get_data(state_data_dir = 'data/state_level',chosen_data=exog_series)
    else:
        exog_data = None
    if chosen_states is not None:
        endog_data = endog_data[chosen_states]
        if exog_series is not None:
            exog_data = exog_data[chosen_states]

    vecm_model_fit = None
    try:
        vecm_model_fit, predictions, fit_score = fit_vecm_model(endog_data,f'vecm_{endog_series}',exog_data)     
    except:
        pass

    if vecm_model_fit is not None:
        print('vecm model fits with lag: ',vecm_model_fit.k_ar)
    else:
        print('vecm model does not fit.')

chosen_states = ['Arizona','California','Colorado','Nevada','New Mexico','Texas','Utah']
# run_exp3('tot_death', exog_series='m50_index', chosen_states=chosen_states)
# run_exp4('tot_death', exog_series='m50_index', chosen_states=chosen_states)

# Experiment 5 endog_data is mobility.
# run_exp3('m50_index', chosen_states=chosen_states)

# Experiment 6 Granger causality between the chg in mobility and chg in deaths for a given state.
def run_exp6(window=['2020-06-02','2020-08-02'], granger_test='ssr_chi2test', maxlag = 19):

    date_list = pd.date_range(start=window[0],end=window[1]).strftime("%Y-%m-%d").tolist()
    tot_death = get_data(state_data_dir,chosen_data='tot_death').loc[date_list]
    m50_index = get_data(state_data_dir,chosen_data='m50_index').loc[date_list]

    causal_values = {}
    for _state in tot_death.columns:
        combined_df = pd.concat([tot_death[_state],m50_index[_state]],axis=1)
        _result = grangercausalitytests(combined_df, maxlag= maxlag, verbose=False)
        causal_values[_state]= [round(val[0][granger_test][1],4) for key,val in _result.items()]

    store_exp_results(causal_values,False,exp_name=f'granger_{granger_test}')

# run_exp6()

# fit_one_predict_all(state_data_dir,chosen_model_state,chosen_metric='r2',regr_model_type='arima')   