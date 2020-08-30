# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 06:20:42 2020

@author: Yunus Emre Midilli
"""

import Neural_Attention_Mechanism
import pandas as pd
import numpy as np
from sklearn import metrics
from Connect_to_Database import execute_sql


def get_feature_values(model_id, feature_type_id):
    sql = "SELECT * FROM FN_GET_MODEL_FEATURES("+str(model_id)+","+str(feature_type_id)+")"
    
    df_features = execute_sql(sql)
    df_window_length=pd.DataFrame()
    for index, row in df_features.iterrows():
        window_length_min = row["WINDOW_LENGTH_MIN"]
        window_length_max = row["WINDOW_LENGTH_MAX"]
        
        df_range = pd.DataFrame(np.array(range(window_length_min,window_length_max+1)))
    
        if index==0:
            df_window_length = df_range
        else:
            df_window_length = df_window_length.append(df_range)
        
    df_window_length = df_window_length.drop_duplicates()
    
    df_all_feature_values = execute_sql("SELECT * FROM FN_GET_MODEL_FEATURE_VALUES("+str(model_id)+")")
    df_all_feature_values = df_all_feature_values.set_index("TIME_STAMP")

    df_feature_values = pd.DataFrame()
    df_feature_ids=pd.DataFrame()
    for i_index, i_row in df_window_length.iterrows():
        time_step = i_row[0]
        
        for j_index, j_row in df_features.iterrows():
            feature_id = j_row["FEATURE_ID"]
            feature_name = j_row["FEATURE_SHORT_DESCRIPTION"]
            stream_name = j_row["STREAM_SHORT_DESCRIPTION"]
            window_length_min = j_row["WINDOW_LENGTH_MIN"]
            window_length_max = j_row["WINDOW_LENGTH_MAX"]
            converted_feature_name = feature_type_id+ "_" + stream_name+ "_" +feature_name+ "_" +str(time_step)
            
            df = df_all_feature_values[df_all_feature_values["FEATURE_ID"]==feature_id]
            df_values = df["VALUE"]
            df_ids = df["ID"]
            
            df_values = pd.DataFrame(df_values)
            
            if time_step >= window_length_min and time_step <= window_length_max:
                df_values = df_values["VALUE"].shift(-time_step)
            else:
                df_values = np.zeros(df_values.shape[0])

            df_feature_values[converted_feature_name] = df_values
            df_feature_ids[converted_feature_name] = df_ids
            
    df_feature_values.sort_index(ascending=False)
    df_feature_values = df_feature_values.dropna()

    df_feature_ids = df_feature_ids.loc[df_feature_values.index]
    df_feature_ids.sort_index(ascending=False)
    df_feature_ids = df_feature_ids.dropna()

    return df_feature_values, df_features, df_window_length, df_feature_ids

def preprocess(model_id):
    df_input, df_features_input, df_window_length_input , df_feature_ids_input = get_feature_values(model_id, "1")
    df_target, df_features_target, df_window_length_target , df_feature_ids_target = get_feature_values(model_id, "2")
    df_merged =pd.merge(df_input, df_target, left_index=True, right_index=True)
        
    df_input = df_merged[df_input.columns]
    df_target= df_merged[df_target.columns]
    
    return  df_input, df_target , df_features_input, df_window_length_input, df_features_target, df_window_length_target, df_feature_ids_input, df_feature_ids_target


def learn(model_id):
    
    df_input, df_target, df_features_input, df_window_length_input, df_features_target, df_window_length_target, df_feature_ids_input, df_feature_ids_target= preprocess(model_id)
    
    if df_input.shape[0] == 0:
        result = 4
    else:
        actual, prediction = Neural_Attention_Mechanism.learn(df_input, df_target, df_features_input, df_window_length_input, df_features_target, df_window_length_target)
        
        sql ="EXEC SP_DELETE_TESTS " + str(model_id) + "; "
        
        for i_index, i_row in df_window_length_target.iterrows():   
            for j_index, j_row in df_features_target.iterrows():
                time_step = i_row[0]
                window_length_min = j_row["WINDOW_LENGTH_MIN"]
                window_length_max = j_row["WINDOW_LENGTH_MAX"]
                if time_step >= window_length_min and time_step <= window_length_max:
                    model_feature_id = j_row["ID"]
                    feature_name = j_row["FEATURE_SHORT_DESCRIPTION"]
                    stream_name = j_row["STREAM_SHORT_DESCRIPTION"]
                    col_name = "2"+ "_" + stream_name+ "_" +feature_name+ "_" + str(time_step)
                                        
                    y_true = np.array(actual[col_name])
                    y_pred = np.array(prediction[col_name])
                    
                    mse = metrics.mean_squared_error(y_true, y_pred)
                    mae = metrics.mean_absolute_error(y_true, y_pred) 
                    r = np.corrcoef(y_true, y_pred)[0, 1]
                    r2 = r**2
                    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
                    count = len(y_true)
                    exp_var = metrics.explained_variance_score(y_true, y_pred)
                    
                    sql =sql + "EXEC SP_ADD_TEST "+str(model_feature_id)+", "+str(time_step)+", 1 ,"+ str(count) + "; "                    
                    sql =sql + "EXEC SP_ADD_TEST "+str(model_feature_id)+", "+str(time_step)+", 2 ,"+ str(r2)+ "; "                    
                    sql =sql + "EXEC SP_ADD_TEST "+str(model_feature_id)+", "+str(time_step)+", 3 ,"+ str(mae)+ "; "                         
                    sql =sql + "EXEC SP_ADD_TEST "+str(model_feature_id)+", "+str(time_step)+", 4 ,"+ str(mse)+ "; "                    
                    sql =sql + "EXEC SP_ADD_TEST "+str(model_feature_id)+", "+str(time_step)+", 5 ,"+ str(mape)+ "; "                    
                    sql =sql + "EXEC SP_ADD_TEST "+str(model_feature_id)+", "+str(time_step)+", 6 ,"+ str(exp_var)+ "; "
                    sql =sql + "EXEC SP_ADD_TEST "+str(model_feature_id)+", "+str(time_step)+", 7 ,"+ str(r)+ "; "

        
        execute_sql(sql,"none")
        result = 5
        
    return result


def main():
    sql ="SELECT * FROM VW_MODELS WHERE LATEST_STATUS_ID = 2"
    
    df_models = execute_sql(sql)
    for i_index, i_row in df_models.iterrows():
        model_id = i_row["ID"]
        execute_sql("EXEC SP_UPDATE_MODEL_STATUS '"+str(model_id)+"', 3, 1")
        result = learn(model_id)
        execute_sql("EXEC SP_UPDATE_MODEL_STATUS '"+str(model_id)+"', "+ str(result) +", 1")
        
    
main()
