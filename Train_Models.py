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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


training_ratio = 0.7
test_ratio = round(1 - training_ratio,2)


def get_feature_values(model_id, feature_type_id):

    sql = "EXEC SP_GET_TIME_STEPS "+ str(model_id) +","+ str(feature_type_id)
    df_time_steps =  execute_sql(sql, "")
    
    df_all_feature_values = execute_sql("SELECT * FROM FN_GET_MODEL_FEATURE_VALUES("+str(model_id)+")")
    df_all_feature_values = df_all_feature_values.set_index("TIME_STAMP")

    df_feature_values = pd.DataFrame()
    df_feature_ids=pd.DataFrame()
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

    return df_feature_values, df_time_steps

def preprocess(model_id):
    df_input, df_time_steps_input= get_feature_values(model_id, "1")
    df_target, df_time_steps_target = get_feature_values(model_id, "2")
    df_merged =pd.merge(df_input, df_target, left_index=True, right_index=True)
        
    df_input = df_merged[df_input.columns]
    df_target= df_merged[df_target.columns]
    
    return  df_input, df_target ,df_time_steps_input, df_time_steps_target


def scale(X,ref_min, ref_max):
    range_max = 1
    range_min = 0
    
    X_min = X.min(axis=0).to_frame().transpose()
    X_max = X.max(axis=0).to_frame().transpose()
    
    ref_min = pd.concat([ref_min, X_min], ignore_index=True).min(axis=0)
    ref_max = pd.concat([ref_max, X_max], ignore_index=True).max(axis=0)
    
    X_std = (X - ref_min) / (ref_max - ref_min)
    X_scaled = (X_std * (range_max - range_min)) + range_min
    X_scaled = X_scaled.fillna(0)
    return X_scaled


def inverse_scale(X_scaled,ref_min, ref_max):
    range_max = 1
    range_min = 0
    

    X = (((X_scaled - ref_min.values)/(range_max - range_min))*(ref_max.values-ref_min.values))+ref_min.values

    return X


def learn(model_id):
    
    df_input, df_target, df_time_steps_input, df_time_steps_target = preprocess(model_id)
    
    if df_input.shape[0] == 0:
        result = 4
    else:
        input_tensor_train, input_tensor_test, target_tensor_train, target_tensor_test = train_test_split(df_input, df_target, test_size=test_ratio, shuffle=True)
        
        for i in df_time_steps_input.loc["ID"]:
            min_val = input_tensor_train[i].min()
            max_val = input_tensor_train[i].max()
            mean_val = input_tensor_train[i].mean()
            std_val = input_tensor_train[i].std()
            
            sql = "EXEC SP_UPDATE_TIME_STEP "+str(i)+", '"+ str(min_val) +"', '"+ str(max_val) +"','"+ str(mean_val) +"','"+ str(std_val) +"' " 
            execute_sql(sql,"none")
    
    
        for i in df_time_steps_target.loc["ID"]:
            min_val = target_tensor_train[i].min()
            max_val = target_tensor_train[i].max()
            mean_val = target_tensor_train[i].mean()
            std_val = target_tensor_train[i].std()
            
            sql = "EXEC SP_UPDATE_TIME_STEP "+str(i)+", '"+ str(min_val) +"', '"+ str(max_val) +"','"+ str(mean_val) +"','"+ str(std_val) +"' " 
            execute_sql(sql,"none")
    
    
        sql = "EXEC SP_GET_TIME_STEPS "+ str(model_id) +", 1"
        df_time_steps_input = execute_sql(sql, "").transpose()
        
        sql = "EXEC SP_GET_TIME_STEPS "+ str(model_id) +", 2"
        df_time_steps_target =  execute_sql(sql, "").transpose()
        
        df_test_index = input_tensor_test.index
        df_time_steps_input.columns = df_input.columns
        df_time_steps_target.columns = df_target.columns
        ref_min_x = df_time_steps_input.loc[["REFERENCE_MINIMUM"]]
        ref_max_x = df_time_steps_input.loc[["REFERENCE_MAXIMUM"]]
        ref_min_y = df_time_steps_target.loc[["REFERENCE_MINIMUM"]]
        ref_max_y = df_time_steps_target.loc[["REFERENCE_MAXIMUM"]]
        
        feature_size_x = df_time_steps_input.loc[["MODEL_FEATURE_ID"]].transpose().MODEL_FEATURE_ID.unique().size
        feature_size_y = df_time_steps_target.loc[["MODEL_FEATURE_ID"]].transpose().MODEL_FEATURE_ID.unique().size
        
        
        scaled_input_train = scale(input_tensor_train, ref_min_x, ref_max_x)
        scaled_target_train = scale(target_tensor_train, ref_min_y, ref_max_y)
        scaled_input_test = scale(input_tensor_test, ref_min_x, ref_max_x)
        scaled_target_test = scale(target_tensor_test, ref_min_y, ref_max_y)
        
        prediction = Neural_Attention_Mechanism.main(scaled_input_train.values, scaled_target_train.values, scaled_input_test.values, feature_size_x, feature_size_y)
        
        prediction = inverse_scale(prediction, ref_min_y, ref_max_y)
        prediction = pd.DataFrame(prediction)
        prediction["INDEX"] = df_test_index
        prediction = prediction.set_index("INDEX")
        prediction.columns = target_tensor_test.columns
        
        actual = target_tensor_test
        
        sql ="EXEC SP_DELETE_TESTS " + str(model_id) + "; "
        
        # for i_index, i_row in df_window_length_target.iterrows():   
        #     for j_index, j_row in df_features_target.iterrows():
        #         time_step = i_row[0]
        #         window_length_min = j_row["WINDOW_LENGTH_MIN"]
        #         window_length_max = j_row["WINDOW_LENGTH_MAX"]
        #         if time_step >= window_length_min and time_step <= window_length_max:
        #             model_feature_id = j_row["ID"]
        #             feature_name = j_row["FEATURE_SHORT_DESCRIPTION"]
        #             stream_name = j_row["STREAM_SHORT_DESCRIPTION"]
        #             col_name = "2"+ "_" + stream_name+ "_" +feature_name+ "_" + str(time_step)
                                        
        #             y_true = np.array(actual[col_name])
        #             y_pred = np.array(prediction[col_name])
                    
        #             mse = metrics.mean_squared_error(y_true, y_pred)
        #             mae = metrics.mean_absolute_error(y_true, y_pred) 
        #             r = np.corrcoef(y_true, y_pred)[0, 1]
        #             r2 = r**2
        #             mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        #             count = len(y_true)
        #             exp_var = metrics.explained_variance_score(y_true, y_pred)
                    
        #             sql =sql + "EXEC SP_ADD_TEST "+str(model_feature_id)+", "+str(time_step)+", 1 ,"+ str(count) + "; "                    
        #             sql =sql + "EXEC SP_ADD_TEST "+str(model_feature_id)+", "+str(time_step)+", 2 ,"+ str(r2)+ "; "                    
        #             sql =sql + "EXEC SP_ADD_TEST "+str(model_feature_id)+", "+str(time_step)+", 3 ,"+ str(mae)+ "; "                         
        #             sql =sql + "EXEC SP_ADD_TEST "+str(model_feature_id)+", "+str(time_step)+", 4 ,"+ str(mse)+ "; "                    
        #             sql =sql + "EXEC SP_ADD_TEST "+str(model_feature_id)+", "+str(time_step)+", 5 ,"+ str(mape)+ "; "                    
        #             sql =sql + "EXEC SP_ADD_TEST "+str(model_feature_id)+", "+str(time_step)+", 6 ,"+ str(exp_var)+ "; "
        #             sql =sql + "EXEC SP_ADD_TEST "+str(model_feature_id)+", "+str(time_step)+", 7 ,"+ str(r)+ "; "

        # execute_sql(sql,"none")
        # result = 5
        
    return result


def main():
    sql ="SELECT * FROM VW_MODELS WHERE LATEST_STATUS_ID = 2"
    
    df_models = execute_sql(sql)
    for i_index, i_row in df_models.iterrows():
        model_id = i_row["ID"]
        execute_sql("EXEC SP_UPDATE_MODEL_STATUS '"+str(model_id)+"', 3, 1")
        result = learn(model_id)
        execute_sql("EXEC SP_UPDATE_MODEL_STATUS '"+str(model_id)+"', "+ str(result) +", 1")
        
    
# main()


model_id = 1068

        
        
