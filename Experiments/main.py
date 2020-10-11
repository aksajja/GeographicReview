import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import pickle
from typing import List
import numpy as np

from utils import merge_county_data

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

def get_endog_data(state_data_dir: str):
    if Path('data/endog_combined_state_level.csv').exists():
        with open('data/endog_combined_state_level.csv','r') as endog_combined_state_level:
            return pd.read_csv(endog_combined_state_level,index_col=0)
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
            endog_df[_state_file.split('.')[0]]=data['tot_death'].values
        # Use change in tot_deaths. Drop first row with NaN.
        endog_df = endog_df.diff().iloc[1:]
        with open('data/endog_combined_state_level.csv','w') as endog_combined_state_level:
            endog_df.to_csv(endog_combined_state_level)
    return endog_df

def get_exog_data(state_data_dir: str):
    if Path('data/exog_combined_state_level.csv').exists():
        with open('data/exog_combined_state_level.csv','r') as exog_combined_state_level:
            return pd.read_csv(exog_combined_state_level,index_col=0)
    else:
        state_files = os.listdir(state_data_dir)
        cols = [_state_file.split('.')[0] for _state_file in state_files]
        exog_df = pd.DataFrame()
        for _state_file in state_files:   # state_files contains state filenames.
            filename = state_data_dir+'/'+_state_file
            data=pd.read_csv(filename,index_col=0)
            data=data.interpolate(method='linear', limit_direction='forward', axis=0)   # Interpolation to handle missing values.
            if len(exog_df.index)<1:   # Insert index.
                exog_df = pd.DataFrame(index=data.index,columns=cols)
            exog_df[_state_file.split('.')[0]]=data['m50_index'].values
        # Use change in m50_index. Drop first row with NaN.
        exog_df = exog_df.diff().iloc[1:]
        with open('data/exog_combined_state_level.csv','w') as exog_combined_state_level:
            exog_df.to_csv(exog_combined_state_level)
    return exog_df

def fit_ar_model():
    pass

def store_exp_results(fit_score: dict, has_exog: bool, exp_name: str) -> None:
    if has_exog:
        with open(f'data/results/{exp_name}_w_exog.csv','w') as exp4:
            pd.DataFrame.from_dict(fit_score).to_csv(exp4)
    else:
        with open(f'data/results/{exp_name}_wo_exog.csv','w') as exp4:
            pd.DataFrame.from_dict(fit_score).to_csv(exp4)

def fit_var_model(endog_data: pd.DataFrame, max_lags=None, exog_data=None):
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
    print('total samples: ',total_samples,' train samples: ',endog_train_data.shape)
    var_model = VAR(endog_train_data,exog=exog_train_data)
    var_model_fit = var_model.fit(maxlags=max_lags,ic='hqic')

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
            _forecast_input = np.concatenate((endog_train_data.iloc[i-lag_order:],test_data[:i,:]),axis=0)
            predictions = np.append(predictions,var_model_fit.forecast(_forecast_input,steps=steps,exog_future=exog_input),axis=0)
        else:
            predictions = np.append(predictions,var_model_fit.forecast(test_data.iloc[i-lag_order:i],steps=1,exog_future=exog_input),axis=0)
            if i == test_data.shape[0]:
                break
    
    fit_score = {'states':[],'r2':[]}
    for _state in endog_data.columns:
        fit_score['states'].append(_state)
        fit_score['r2'].append(r2_score(test_data[_state],predictions[_state]))
    
    store_exp_results(fit_score, exog_data is not None, 'var_exp')
    return var_model_fit, predictions, fit_score

def fit_vecm_model(data: pd.DataFrame, exog_data=None):
    from statsmodels.tsa.vector_ar.vecm import VECM
    
    # Hardcoded values to have consistency in models.
    train_pct = 0.9
    prediction_steps = 1
    total_samples = len(endog_data.index)
    training_sample_size = int(train_pct*total_samples)
    test_sample_size = total_samples - training_sample_size

    test_data = endog_data.iloc[training_sample_size:]
    print('total samples: ',total_samples,' train samples: ',training_sample_size)

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

    store_exp_results(fit_score, exog_data is not None, 'vecm_exp')
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


def load_model(model_dir: str,state_data_dir: str, model_state: str, regr_model_type: str, lag: int):
    filename = f'{model_dir}/{regr_model_type}/{model_state}/{model_state}_lag_{lag}.sav'
    model_dir_struct = filename.split('/')
    for depth in range(1,len(model_dir_struct)):  # List contains only folders.
        _dir = '/'.join(model_dir_struct[:depth])
        if not os.path.exists(_dir):
            os.makedirs(_dir)
    if Path(filename).exists():
        return pickle.load(open(filename, 'rb'))
    else:
        regr_model = fit_rf_model(state_data_dir,model_state,filename,lag)
        return regr_model

def fit_one_predict_all(state_data_dir: str, model_state: str, lag=17):
    regr_model_type = 'random_forest'
    loaded_model = load_model(model_dir,state_data_dir,model_state,regr_model_type,lag)
    # Use the loaded model to predict for all states.
    r2_dict = {}
    state_files = os.listdir(state_data_dir)
    for _state_file in state_files:   # state_files contains state filenames.
        X, Y= get_processed_input(state_data_dir+'/'+_state_file, lag)
        result = loaded_model.predict(X)
        r2=r2_score(Y, result)
        r2_dict[_state_file]=r2
    Final=pd.DataFrame.from_dict({'State':list(r2_dict.keys()),'Rsquared_error':list(r2_dict.values())})
    Final.to_csv('data/results/exp1.csv')

def fit_each_w_lags(state_data_dir: str, lags_list: List[int]):
    regr_model_type = 'random_forest'
    state_files = os.listdir(state_data_dir)
    cols = [f'Lag_{_lag}_r2' for _lag in lags_list]
    final_df = pd.DataFrame(columns=cols)
    for _state_file in state_files:   # state_files contains state filenames.
        model_state = _state_file.split('.')[0]
        r2_dict = {}
        for _lag in lags_list:
            X, Y= get_processed_input(state_data_dir+'/'+_state_file,_lag)
            loaded_model = load_model(model_dir,state_data_dir,model_state,regr_model_type,_lag)
            result = loaded_model.predict(X)
            r2=r2_score(Y, result)
            r2_dict[_lag]=r2
        final_df.loc[_state_file]=list(r2_dict.values())
    final_df.to_csv('data/results/exp2.csv')

def setup_data_dir():
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

lags_list = [_lag for _lag in range(1,3)]
fit_one_predict_all(state_data_dir,'Arizona')   # Experiment 1
# fit_each_w_lags(state_data_dir,lags_list)   # Experiment 2

endog_data = get_endog_data(state_data_dir = 'data/state_level')
exog_data = get_exog_data(state_data_dir = 'data/state_level')

# var_model_fit, predictions, fit_score = fit_var_model(endog_data,2,exog_data)     # Experiment 3 with exog_data.
# var_model_fit, predictions, fit_score = fit_var_model(endog_data,2)     # Experiment 3 without exog_data.
# vecm_model_fit, predictions, fit_score = fit_vecm_model(endog_data, exog_data)     # Experiment 4 with exog_data.
# vecm_model_fit, predictions, fit_score = fit_vecm_model(endog_data)     # Experiment 4 without exog_data.

# print('R2 score for Var is : ', fit_score)