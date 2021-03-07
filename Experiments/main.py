import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr
import pickle
from typing import List
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests

from utils import merge_county_data, extract_max_lag_scores, extract_significant_chi2_statistic
from constants import model_dir,state_data_dir,results_dir


class EvaluationModel:
    # Class attributes--constructor
    def __init__(self,chosen_label,chosen_columns):
        self.chosen_label = chosen_label
        self.chosen_columns = chosen_columns

    # New_Method
    def feature_scale(self, filename):
        chosen_columns = self.chosen_columns
        
        #Getting original Data
        data=pd.read_csv(filename,index_col=0)
        data=data[chosen_columns]
        #Scaling chosen columns
        data = np.asarray(data)
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)
        return scaled_data
        
    # New_method; result will be stored as 'r2_result' in cwd
    def r_squared_compare(self, inputfilepath, tolerance=0.1):
        count = 0
        result = pd.DataFrame(columns = ['State','r2_error'])

        #Read in the file
        data=pd.read_csv(inputfilepath)
        #Find Arizona
        index = data[data['State']=='Arizona'].index.values
        # Get Arizona r2
        AZr2 = data['r2_error'].values[index]
        x=float(AZr2)
        #Normalize
        data['r2_error'] = data['r2_error'] +(tolerance-x)
        #Find all values within +- tolerance of Arizona
        for ind in data.index: 
            if(data['r2_error'][ind] > 0 and data['r2_error'][ind] <= (2*tolerance)):
                result.loc[data.index[ind]] = data.iloc[ind]
                count = count+1

        path = results_dir + "/r2_comparison_result.csv"
        result.to_csv(path)      
        print('State Count:',count)
        
    def normalize_with_population(self,state,X,Y):
        local_X = X.copy()
        local_Y = Y.copy()
        million = 1000000
        cols_to_normalize = ['new_case','new_death']
        
        df = pd.read_excel('data/population.xlsx', header=1)
        df['State'] = df['State'].map(lambda x: x.lstrip('.').rstrip('.'))
        population=np.array(df)
        idx=np.where(population==state)[0][0]
        norm=population[idx][1]/million     # Population in millions
        for _col in local_X.columns:
            if _col in cols_to_normalize:
                local_X[_col] = local_X[_col].div(norm).round(4)
        local_Y = local_Y.div(norm).round(4)
    
        combined_df = local_X.merge(local_Y, how='inner', left_index=True, right_index=True)
        combined_df.to_csv('data/normalized_state_level/'+state+'.csv')
        return local_X,local_Y

    def normalize_data(self,state,X,Y):
        normalized_X,normalized_Y = self.normalize_with_population(state,X,Y)
    
        return normalized_X,normalized_Y

    def get_processed_input(self,filename: str, lag: int) -> List[pd.DataFrame]:
        chosen_label = self.chosen_label
        chosen_columns = self.chosen_columns
        
        # Initialize data.
        data=pd.read_csv(filename,index_col=0)
        state=filename.split('.')[0].split('/')[-1]
        data=data.interpolate(method='linear', limit_direction='forward', axis=0)   # Interpolation to handle missing values.
        data=data.drop(columns=['state'])
        Label=data[chosen_label].to_frame()
        data=data[chosen_columns]
        
        # Difference.
        data_diff=data.diff()       #  Take diff of consecutive days.
        Label_diff=Label.diff()
        Label_diff=Label_diff.iloc[1:]   # Drop first row.
        data_diff=data_diff.iloc[1:]   
        Y=Label_diff-Label_diff.shift(lag)      # Eg - This is the difference b/w labels, i.e. tot_deaths, of day1 and day18.
        
        # Data alignment.
        Y.drop(Y.head(lag).index,inplace=True)    # Drop head elements = lag_order. 
        data_diff.drop(data_diff.tail(lag).index,inplace=True)  # Drop tail elements = lag_order.
        
        # Normalize data.
        data_diff_norm,Y_diff_norm = self.normalize_data(state,data_diff,Y)
        
        return data_diff_norm,Y_diff_norm

# subclasses:
class ArimaModel(EvaluationModel):
    def __init__(self, chosen_model_state, chosen_metric, chosen_label, chosen_columns, lags_list, regr_model_type, chosen_states):
        super().__init__(chosen_label,chosen_columns)
        self.chosen_model_state = chosen_model_state
        self.chosen_metric = chosen_metric
        self.chosen_label = chosen_label
        self.chosen_columns = chosen_columns
        self.lags_list = lags_list
        self.chosen_states = chosen_states

    def fit(self,model_state: str, endog_data: pd.DataFrame, exp_name: str, exog_data=None):
        from statsmodels.tsa.arima.model import ARIMA
        # Hardcoded values to have consistency in models.
        train_pct = 0.9
        steps = 1
        all_states = endog_data.columns
        total_samples = len(endog_data.index)
        training_sample_size = int(train_pct*total_samples)
        test_sample_size = total_samples - training_sample_size
        
        num_endog_vars = len(endog_data.columns)
        endog_train_data = endog_data.iloc[:training_sample_size]
        exog_train_data = None
        if exog_data is not None:
            exog_train_data = exog_data.iloc[:training_sample_size]

        test_data = endog_data.iloc[training_sample_size:]
        history = endog_train_data.copy()
        for _col in history.columns:
            history[_col].values[:training_sample_size]=history[model_state].values[:training_sample_size]
        history.index = pd.DatetimeIndex(history.index) # Converts datatype str to Datetime
        history.index = history.index.to_period('D')    # Converts Datetime to 0,1,2,...,n
        predictions = pd.DataFrame(columns=endog_train_data.columns)
        for t in range(len(test_data)):
            each_day_predictions = []
            obs = []
            for _state in all_states:   # state_files contains state filenames.
                print(f'{t} day for {_state}')
                model = ARIMA(history[_state],order=(7,0,1),exog=exog_train_data)
                model_fit = model.fit()
                output = model_fit.forecast(steps=steps)
                yhat = output.iloc[0]
                each_day_predictions.append(yhat)
            history.loc[history.index[-1]+pd.offsets.Day(1)]=test_data.iloc[t]
            predictions.loc[history.index[-1]+pd.offsets.Day(1)] = each_day_predictions
    
        fit_score = {'states':[],'r2':[]}
        for _state in endog_data.columns:
            y_obs = test_data[:test_sample_size][_state].to_numpy()
            y_pred = predictions[:,list(endog_data.columns).index(_state)]
            fit_score['states'].append(_state)
            fit_score['r2'].append(r2_score(y_obs,y_pred))

        store_exp_results(fit_score, exog_data is not None, exp_name)
        
        return predictions,test_data

    def load_model(self, model_state: str, regr_model_type: str, lag: int, exog_series=None, force_fit=False):
        chosen_label = self.chosen_label
        chosen_columns = self.chosen_columns
        
        filename = f'{model_dir}/{regr_model_type}/{model_state}/{model_state}_lag_{lag}.sav'
        model_dir_struct = filename.split('/')
        for depth in range(1,len(model_dir_struct)):  # List contains only folders.
            _dir = '/'.join(model_dir_struct[:depth])
            if not os.path.exists(_dir):
                os.makedirs(_dir)
        if not force_fit and Path(filename).exists():
            return pickle.load(open(filename, 'rb'))
        else:
            endog_data = get_data()[model_state]
            if exog_series is not None:
                exog_data = get_data(chosen_data=exog_series)[model_state]
            else:
                exog_data = None
            regr_model = self.fit(model_state, endog_data,f'one_to_all_{regr_model_type}',exog_data=exog_data)    # Fitted to Initial training data.
            return regr_model

class ArmaModel(EvaluationModel):
    def __init__(self, chosen_model_state, chosen_metric, chosen_label, chosen_columns, lags_list, regr_model_type, chosen_states):
        super().__init__(chosen_label,chosen_columns)
        self.chosen_model_state = chosen_model_state
        self.chosen_metric = chosen_metric
        self.chosen_label = chosen_label
        self.chosen_columns = chosen_columns
        self.lags_list = lags_list
        self.chosen_states = chosen_states

    def fit(self,model_state: str, endog_data: pd.DataFrame, exp_name: str, exog_data=None):
        import warnings
        warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARMA',
                                FutureWarning)
        from statsmodels.tsa.arima_model import ARMA
        # Hardcoded values to have consistency in models.
        train_pct = 0.9
        steps = 1
        all_states = endog_data.columns
        total_samples = len(endog_data.index)
        training_sample_size = int(train_pct*total_samples)
        test_sample_size = total_samples - training_sample_size
    
        num_endog_vars = len(endog_data.columns)
        endog_train_data = endog_data.iloc[:training_sample_size]
        exog_train_data = None
        if exog_data is not None:
            exog_train_data = exog_data.iloc[:training_sample_size]

        test_data = endog_data.iloc[training_sample_size:]
        history = endog_train_data.copy()
        for _col in history.columns:
            history[_col].values[:training_sample_size]=history[model_state].values[:training_sample_size]
        history.index = pd.DatetimeIndex(history.index)
        history.index = history.index.to_period('D')
        predictions = pd.DataFrame(columns=endog_train_data.columns)
        for t in range(len(test_data)):
            each_day_predictions = []
            obs = []
            for _state in all_states:   # state_files contains state filenames.
                model = ARMA(history[_state],order=(17,1),exog=exog_train_data)
                model_fit = model.fit(maxiter=5)
                output = model_fit.forecast(steps=steps)
                yhat = output[0][0]
                each_day_predictions.append(yhat)
            history.loc[history.index[-1]+pd.offsets.Day(1)]=test_data.iloc[t]
            predictions.loc[history.index[-1]+pd.offsets.Day(1)] = each_day_predictions
    
        fit_score = {'states':[],'r2':[]}
        for _state in endog_data.columns:
            y_obs = test_data[:test_sample_size][_state].to_numpy()
            y_pred = predictions[:,list(endog_data.columns).index(_state)]
            fit_score['states'].append(_state)
            fit_score['r2'].append(r2_score(y_obs,y_pred))

        store_exp_results(fit_score, exog_data is not None, exp_name)


        return predictions,test_data

    def load_model(self, model_state: str, regr_model_type: str, lag: int, exog_series=None, force_fit=False):
        chosen_label = self.chosen_label
        chosen_columns = self.chosen_columns
        
        filename = f'{model_dir}/{regr_model_type}/{model_state}/{model_state}_lag_{lag}.sav'
        model_dir_struct = filename.split('/')
        for depth in range(1,len(model_dir_struct)):  # List contains only folders.
            _dir = '/'.join(model_dir_struct[:depth])
            if not os.path.exists(_dir):
                os.makedirs(_dir)
        if not force_fit and Path(filename).exists():
            return pickle.load(open(filename, 'rb'))
        else:
            endog_data = get_data()[model_state]
            if exog_series is not None:
                exog_data = get_data(chosen_data=exog_series)[model_state]
            else:
                exog_data = None
            regr_model = self.fit(model_state, endog_data,f'one_to_all_{regr_model_type}',exog_data=exog_data)    # Fitted to Initial training data.
            return regr_model

class VecmModel(EvaluationModel):
    def __init__(self, chosen_model_state, chosen_metric, chosen_label, chosen_columns, lags_list, regr_model_type, chosen_states):
        super().__init__(chosen_label,chosen_columns)
        self.chosen_model_state = chosen_model_state
        self.chosen_metric = chosen_metric
        self.chosen_label = chosen_label
        self.chosen_columns = chosen_columns
        self.lags_list = lags_list
        self.chosen_states = chosen_states

    def fit(self,model_state,endog_data: pd.DataFrame, exp_name: str, exog_data=None):
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

    def load_model(self, model_state: str, regr_model_type: str, lag: int, exog_series=None, force_fit=False):
        chosen_label = self.chosen_label
        chosen_columns = self.chosen_columns
        
        filename = f'{model_dir}/{regr_model_type}/{model_state}/{model_state}_lag_{lag}.sav'
        model_dir_struct = filename.split('/')
        for depth in range(1,len(model_dir_struct)):  # List contains only folders.
            _dir = '/'.join(model_dir_struct[:depth])
            if not os.path.exists(_dir):
                os.makedirs(_dir)
        if not force_fit and Path(filename).exists():
            return pickle.load(open(filename, 'rb'))
        else:
            endog_data = get_data()[model_state]
            if exog_series is not None:
                exog_data = get_data(chosen_data=exog_series)[model_state]
            else:
                exog_data = None
            regr_model = self.fit(model_state, endog_data,f'one_to_all_{regr_model_type}',exog_data=exog_data)    # Fitted to Initial training data.
            return regr_model

class VarModel(EvaluationModel):
    def __init__(self, chosen_model_state, chosen_metric, chosen_label, chosen_columns, lags_list, regr_model_type, chosen_states):
        super().__init__(chosen_label,chosen_columns)
        self.chosen_model_state = chosen_model_state
        self.chosen_metric = chosen_metric
        self.chosen_label = chosen_label
        self.chosen_columns = chosen_columns
        self.lags_list = lags_list
        self.chosen_states = chosen_states

    def fit(self,model_state: str, endog_data: pd.DataFrame, exp_name: str, max_lags=None, exog_data=None):
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

    def load_model(self, model_state: str, regr_model_type: str, lag: int, exog_series=None, force_fit=False):
        chosen_label = self.chosen_label
        chosen_columns = self.chosen_columns
        
        filename = f'{model_dir}/{regr_model_type}/{model_state}/{model_state}_lag_{lag}.sav'
        model_dir_struct = filename.split('/')
        for depth in range(1,len(model_dir_struct)):  # List contains only folders.
            _dir = '/'.join(model_dir_struct[:depth])
            if not os.path.exists(_dir):
                os.makedirs(_dir)
        if not force_fit and Path(filename).exists():
            return pickle.load(open(filename, 'rb'))
        else:
            endog_data = get_data()[model_state]
            if exog_series is not None:
                exog_data = get_data(chosen_data=exog_series)[model_state]
            else:
                exog_data = None
            regr_model = self.fit(model_state, endog_data,f'one_to_all_{regr_model_type}',exog_data=exog_data)    # Fitted to Initial training data.
            return regr_model
    
class RandomForestModel(EvaluationModel):
    def __init__(self, chosen_model_state, chosen_metric, chosen_label, chosen_columns, lags_list, regr_model_type, chosen_states):
        super().__init__(chosen_label,chosen_columns)
        self.chosen_model_state = chosen_model_state
        self.chosen_metric = chosen_metric
        self.chosen_label = chosen_label
        self.chosen_columns = chosen_columns
        self.lags_list = lags_list
        self.chosen_states = chosen_states

    def fit(self, model_state: str, filename: str, lag: int):
        chosen_label = self.chosen_label
        chosen_columns = self.chosen_columns
        
        X, Y= self.get_processed_input(state_data_dir+f'/{model_state}.csv', lag)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=42)
        regr_model = RandomForestRegressor(max_depth=2, random_state=0)
        regr_model.fit(X_train, y_train)
        # Compute r2 error on test data.
        y_pred=regr_model.predict(X_test)
        print(f'Test error on {model_state} fitted model & lag {lag}: ',r2_score(y_test, y_pred))
        # Store regression model.
        pickle.dump(regr_model, open(filename, 'wb'))
        
        return regr_model

    def load_model(self, model_state: str, regr_model_type: str, lag: int, exog_series=None, force_fit=False):
        chosen_label = self.chosen_label
        chosen_columns = self.chosen_columns
        
        filename = f'{model_dir}/{regr_model_type}/{model_state}/{model_state}_lag_{lag}.sav'
        model_dir_struct = filename.split('/')
        for depth in range(1,len(model_dir_struct)):  # List contains only folders.
            _dir = '/'.join(model_dir_struct[:depth])
            if not os.path.exists(_dir):
                os.makedirs(_dir)
        if not force_fit and Path(filename).exists():
            return pickle.load(open(filename, 'rb'))
        else:
            regr_model = self.fit(model_state,filename,lag)
            return regr_model


def setup_data_dir():
    # Setup results directory.
    
    dir_struct = results_dir.split('/')
    for depth in range(1,len(dir_struct)):  # List contains only folders.
        _dir = '/'.join(dir_struct[:depth])
        if not os.path.exists(_dir):
            os.makedirs(_dir)
    
    # Setup state directory.
    location = state_data_dir
    model_dir_struct = location.split('/')
    for depth in range(1,len(model_dir_struct)+1):  # List contains only folders.
        _dir = '/'.join(model_dir_struct[:depth])
        if not os.path.exists(_dir):
            os.makedirs(_dir)
    merge_county_data()

def get_data(chosen_data='tot_death'):
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

def store_exp_results(fit_score: dict, has_exog: bool, exp_name: str) -> None:
    print(f'storing {exp_name} results ....')
    if 'granger_' in exp_name: 
        with open(f'data/results/{exp_name}.csv','w') as results_file:
            pd.DataFrame.from_dict(fit_score,orient='index',columns=['chi2_val','lag']).to_csv(results_file)
    elif has_exog:
        with open(f'data/results/{exp_name}_w_exog.csv','w') as results_file:
            pd.DataFrame.from_dict(fit_score).to_csv(results_file)
    else:
        with open(f'data/results/{exp_name}_wo_exog.csv','w') as results_file:
            pd.DataFrame.from_dict(fit_score).to_csv(results_file)

def fit_one_predict_all(chosen_label,chosen_columns,chosen_metric,model_state,regr_model_type='random_forest', lag=4,exog_series=None, force_fit=False):
    r2_dict = {}
    if regr_model_type=='random_forest':
        model_obj = RandomForestModel(chosen_model_state,chosen_metric,chosen_label,chosen_columns,lags_list,regr_model_type, chosen_states)
        loaded_model = model_obj.load_model(model_state,regr_model_type,lag=lag, force_fit=force_fit)
        # Use the loaded model to predict for all states.
        state_files = os.listdir(state_data_dir)
        for _state_file in state_files:   # state_files contains state filenames.
            X, Y= model_obj.get_processed_input(state_data_dir+'/'+_state_file, lag)
            result = loaded_model.predict(X)
            if chosen_metric=='r2':
                corr_metric_vals = r2_score(Y, result)
            elif chosen_metric=='pearsonr':
                corr_metric_vals = pearsonr(Y['tot_death'], result)[0]
            r2_dict[_state_file.split('.')[0]]=corr_metric_vals
    elif regr_model_type=='arima':
        model_obj = ArimaModel(chosen_model_state,chosen_metric,chosen_label,chosen_columns,lags_list,regr_model_type, chosen_states)
        endog_data = get_data() # ToDo: Don't use default selection of endog_data.
        exog_data = None
        if exog_series is not None:
            exog_data = get_data(chosen_data=exog_series)
        predictions,Y = model_obj.fit(model_state, endog_data,f'fit_one_to_all_{regr_model_type}',exog_data)
    
        for _state in endog_data.columns:
            if chosen_metric=='r2':
                corr_metric_vals = r2_score(Y[_state], predictions[_state])
            elif chosen_metric=='pearsonr':
                corr_metric_vals = pearsonr(Y[_state], predictions[_state])[0]
            r2_dict[_state]=corr_metric_vals
    elif regr_model_type=='arma':
        model_obj = ArmaModel(chosen_model_state,chosen_metric,chosen_label,chosen_columns,lags_list,regr_model_type, chosen_states)
        endog_data = get_data() # ToDo: Don't use default selection of endog_data.
        exog_data = None
        if exog_series is not None:
            exog_data = get_data(chosen_data=exog_series)
        predictions,Y = model_obj.fit(model_state, endog_data,f'fit_one_to_all_{regr_model_type}',exog_data)
    
        for _state in endog_data.columns:
            if chosen_metric=='r2':
                corr_metric_vals = r2_score(Y[_state], predictions[_state])
            elif chosen_metric=='pearsonr':
                corr_metric_vals = pearsonr(Y[_state], predictions[_state])[0]
            r2_dict[_state]=corr_metric_vals

    Final=pd.DataFrame.from_dict({'State':list(r2_dict.keys()),f'{chosen_metric}_error':list(r2_dict.values())})
    result_location = f'data/results/fit_one_to_all_{regr_model_type}_{chosen_metric}.csv'
    Final.to_csv(result_location)
    model_obj.r_squared_compare(result_location)
    print(f'Stored results to {result_location}')

def fit_each_w_lags(self,regr_model_type='random_forest'):
    chosen_label = self.chosen_label
    chosen_columns = self.chosen_columns
    lags_list = self.lags_list
    chosen_metric = self.chosen_metric
    state_files = os.listdir(state_data_dir)
    cols = [f'Lag_{_lag}_{chosen_metric}' for _lag in lags_list]
    final_df = pd.DataFrame(columns=cols)
    for _state_file in state_files:   # state_files contains state filenames.
        model_state = _state_file.split('.')[0]
        r2_dict = {}
        for _lag in lags_list:
            X, Y= self.get_processed_input(state_data_dir+'/'+_state_file,_lag)
            loaded_model = self.load_model(model_state,regr_model_type,_lag)
            result = loaded_model.predict(X)
            if chosen_metric=='r2':
                corr_metric_vals = r2_score(Y, result)
            elif chosen_metric=='pearsonr':
                corr_metric_vals = pearsonr(Y['tot_death'], result)[0]
            r2_dict[_lag]=corr_metric_vals
        final_df.loc[_state_file.split('.')[0]]=list(r2_dict.values())
    final_df.to_csv(f'{results_dir}/exp2_{chosen_metric}.csv')
    extract_max_lag_scores(chosen_metric)

setup_data_dir()

## Uncomment to run an experiment.
chosen_model_state = 'Arizona'
chosen_metric = 'r2'    # r2, pearsonr
chosen_label = 'tot_death'
chosen_columns = ['new_case','new_death']
lags_list = [_lag for _lag in range(1,21)]
regr_model_type = 'arma'   # random_forest, arima, arma, var, vecm.
chosen_states = ['Arizona','California','Colorado','Nevada','New Mexico','Texas','Utah']

fit_one_predict_all(chosen_label,chosen_columns,chosen_metric,chosen_model_state,regr_model_type, lag=4,exog_series=None, force_fit=True)
# fit_each_w_lags(chosen_metric,lags_list)   # Experiment 2

# Experiment 3 endog_data is change in tot_death.
def run_exp3(model_state: str,endog_series: str, exog_series=None,chosen_states=None):
    endog_data = get_data(chosen_data=endog_series)
    if exog_series is not None:
        exog_data = get_data(chosen_data=exog_series)
    else:
        exog_data = None
    if chosen_states is not None:
        endog_data = endog_data[chosen_states]
        if exog_series is not None:
            exog_data = exog_data[chosen_states]

    lags_list = [_lag for _lag in range(21,1,-1)]
    model_obj = VarModel(chosen_model_state,chosen_metric,chosen_label,chosen_columns,lags_list,regr_model_type, chosen_states)

    var_model_fit = None
    for _lag in lags_list:
        try:
            var_model_fit, predictions, fit_score = model_obj.fit(model_state,endog_data,f'var_{endog_series}',_lag,exog_data)     
            break
        except:
            continue

    if var_model_fit is not None:
        print('Var model fits with lag: ',var_model_fit.k_ar)
    else:
        print('Var model does not fit.')

# Experiment 4 endog_data is change in tot_death.
def run_exp4(model_state: str,endog_series: str, exog_series=None,chosen_states=None):
    endog_data = get_data(chosen_data=endog_series)
    if exog_series is not None:
        exog_data = get_data(chosen_data=exog_series)
    else:
        exog_data = None
    if chosen_states is not None:
        endog_data = endog_data[chosen_states]
        if exog_series is not None:
            exog_data = exog_data[chosen_states]

    model_obj = VecmModel(chosen_model_state,chosen_metric,chosen_label,chosen_columns,lags_list,regr_model_type, chosen_states)
    vecm_model_fit = None
    try:
        vecm_model_fit, predictions, fit_score = model_obj.fit(model_state,endog_data,f'vecm_{endog_series}',exog_data)     
    except:
        pass

    if vecm_model_fit is not None:
        print('vecm model fits with lag: ',vecm_model_fit.k_ar)
    else:
        print('vecm model does not fit.')

# run_exp3(model_state,'tot_death', exog_series='m50_index', chosen_states=chosen_states)
# run_exp4(model_state,'tot_death', exog_series='m50_index', chosen_states=chosen_states)

# Experiment 5 endog_data is mobility.
# run_exp3('m50_index', chosen_states=chosen_states)

# Experiment 6 Granger causality between the chg in mobility and chg in deaths for a given state.
def run_exp6(window=['2020-03-02','2020-08-02'], granger_test='ssr_chi2test', maxlag = 19):

    date_list = pd.date_range(start=window[0],end=window[1]).strftime("%Y-%m-%d").tolist()
    tot_death = get_data(chosen_data='tot_death')
    m50_index = get_data(chosen_data='m50_index')

    causal_values = {}
    causal_p_values = {}
    for _state in tot_death.columns:
        combined_df = pd.concat([tot_death[_state],m50_index[_state]],axis=1)
        _result = grangercausalitytests(combined_df, maxlag= maxlag, verbose=False)
        causal_values[_state]= [round(val[0][granger_test][0],4) for key,val in _result.items()]
        causal_p_values[_state]= [round(val[0][granger_test][1],4) for key,val in _result.items()]

    causal_values_df = pd.DataFrame.from_dict(causal_values, orient='index')
    causal_p_values_df = pd.DataFrame.from_dict(causal_p_values, orient='index')
    chi2_dict = extract_significant_chi2_statistic(causal_values_df, causal_p_values_df)
    
    store_exp_results(chi2_dict,False,exp_name=f'granger_{granger_test}')

# run_exp6()