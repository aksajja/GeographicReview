from constants import us_state_abbrev
import pandas as pd
import numpy as np
from pathlib import Path

from constants import state_data_dir,results_dir

def merge_county_data():
    # Getting data, rearranging for simplicity and indexing by date
    data1=pd.read_csv('data/United_States_COVID-19_Cases_and_Deaths_by_State_over_Time.csv')
    data1 = data1.set_index('submission_date')
    data1=data1.drop(columns=['prob_cases','pnew_case','prob_death','consent_cases','created_at','consent_deaths','pnew_death','tot_cases', 'conf_cases','conf_death'])

    data2=pd.read_excel('data/Mobility_upto 28-09-2020_DL.xlsx')
    data2 = data2.set_index('Date')
    data2=data2.drop(columns=['Country', 'Samples'])

    skipped=[]
    # Merging county level data to get state level data.
    for _state in us_state_abbrev.keys():
        filename = state_data_dir+f'/{_state}.csv'
        if not Path(filename).exists():
            try:
                sub2=data1[data1["state"]==us_state_abbrev[_state]]
                sub3=data2[data2['State']==_state]
            except:
                skipped.append(_state)
                continue
            
            mergedDf = sub2.merge(sub3, left_index=True, right_index=True)
            mergedDf=mergedDf.drop(columns=['State'])
            mergedDf.to_csv('data/state_level/'+_state+'.csv')

def extract_max_lag_scores(chosen_metric: str):
    with open(f'{results_dir}/exp2_{chosen_metric}.csv','r') as _results_file:
        r_df = pd.read_csv(_results_file,index_col=0)
        max_df = r_df.max(axis=1)
        max_df.to_csv(f'{results_dir}/exp2_{chosen_metric}_max.csv')

def extract_significant_chi2_statistic(causal_values_df, causal_p_values_df):
    causal_p_values = causal_p_values_df.to_numpy()
    mx = np.ma.masked_array(causal_p_values, mask=causal_p_values==0)
    min_indices = mx.argmin(axis=1)
    
    chi2_vals = causal_values_df.to_numpy()
    selected_dict = dict()
    states = causal_values_df.index
    for _row in range(chi2_vals.shape[0]):
        selected_dict[states[_row]]=chi2_vals[_row][min_indices[_row]],min_indices[_row]
    return selected_dict
