from constants import us_state_abbrev
import pandas as pd
from pathlib import Path

def merge_county_data(state_level_dir: str):
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
        filename = state_level_dir+f'/{_state}.csv'
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
    with open(f'data/results/exp2_{chosen_metric}.csv','r') as _results_file:
        r_df = pd.read_csv(_results_file,index_col=0)
        max_df = r_df.max(axis=1)
        max_df.to_csv(f'data/results/exp2_{chosen_metric}_max.csv')
