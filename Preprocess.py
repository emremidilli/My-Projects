# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 20:55:10 2020

@author: yunus
"""

import pandas as pd
from Connect_to_Database import execute_sql


def get_feature_values(model_id, feature_type_id, return_value = True, is_index_time_stamp = True, from_time_stamp = "DEFAULT", to_time_stamp = "DEFAULT" ):
    sql = "EXEC SP_GET_TIME_STEPS "+ str(model_id) +","+ str(feature_type_id)
    df_time_steps =  execute_sql(sql, "")
        
    # from time stamp and to time stamp
    
    if return_value == True:
        sql_feature_value = "SELECT * FROM FN_GET_MODEL_FEATURE_VALUES("+str(model_id)+", "+from_time_stamp+", "+to_time_stamp+")"
    else:
        sql_feature_value = "SELECT TOP 1 * FN_GET_MODEL_FEATURE_VALUES("+str(model_id)+", "+from_time_stamp+", "+to_time_stamp+")"
    
    df_all_feature_values = execute_sql(sql_feature_value)
    
    if is_index_time_stamp == True:
        index_column = "TIME_STAMP"
    else:
        index_column = "ID"
    
    df_all_feature_values = df_all_feature_values.set_index(index_column)

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
