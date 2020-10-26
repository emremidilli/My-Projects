# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 20:55:10 2020

@author: yunus
"""

import pandas as pd
from Connect_to_Database import execute_sql

# PREASSUMPTIONS:
    # tbl_feature_values include row data where each time stamp is correctly sequenced.


def get_feature_values(model_id, feature_type_id, from_time_stamp = "DEFAULT", to_time_stamp = "DEFAULT" ):
    sql = "EXEC SP_GET_TIME_STEPS "+ str(model_id) +","+ str(feature_type_id)
    df_time_steps =  execute_sql(sql, "")
    

    if not from_time_stamp == "DEFAULT":
        from_time_stamp = "'" + str(from_time_stamp) + "'"
        
    if not to_time_stamp == "DEFAULT":
        to_time_stamp = "'" + str(to_time_stamp) + "'"     
    
    sql_feature_value = "EXEC  [SP_GET_MODEL_FEATURE_VALUES] "+str(model_id)+", "+ str(feature_type_id) +" ,"+from_time_stamp+", "+to_time_stamp

    df_all_feature_values = execute_sql(sql_feature_value)
    
    df_all_feature_values = df_all_feature_values.set_index("TIME_STAMP")

    df_feature_values = pd.DataFrame()
    for i_index, i_row in df_time_steps.iterrows():
        time_step_id = i_row["ID"]
        time_step = int(i_row["TIME_STEP"])
        feature_id = i_row["FEATURE_ID"]
        boundary= i_row["BOUNDARY"]
        
        df = df_all_feature_values[df_all_feature_values["FEATURE_ID"]==feature_id]
        df_values = df["VALUE"]
        df_values = pd.DataFrame(df_values)
        
        df_values = df_values["VALUE"].shift(-time_step)
        df_feature_values[time_step_id] = df_values
        
        if boundary == 0:
            df_feature_values[time_step_id] = 0
    
    df_feature_values.sort_index(ascending=False)
    df_feature_values = df_feature_values.dropna()
    df_time_steps = df_time_steps.transpose()
    
    df_time_steps.columns = df_feature_values.columns 
    
    return df_feature_values, df_time_steps


def get_dimension_size(df_time_steps):
    feature_size = df_time_steps.loc[["MODEL_FEATURE_ID"]].transpose().MODEL_FEATURE_ID.unique().size
    window_length = df_time_steps.loc[["TIME_STEP"]].transpose().TIME_STEP.unique().size
    return feature_size, window_length


def main(model_id):
    df_input, df_time_steps_input= get_feature_values(model_id, "1")
    df_target, df_time_steps_target = get_feature_values(model_id, "2")
    df_merged =pd.merge(df_input, df_target, left_index=True, right_index=True)
        
    df_input = df_merged[df_input.columns]
    df_target= df_merged[df_target.columns]
    
    return  df_input, df_target ,df_time_steps_input, df_time_steps_target
