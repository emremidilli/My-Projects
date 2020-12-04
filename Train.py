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
import pickle
import Preprocess


training_ratio = 0.7
test_ratio = round(1 - training_ratio,2)
scalers_dir = './__scalers__/'


def main():
    sql ="SELECT * FROM VW_MODELS WHERE LATEST_STATUS_ID = 2"
    
    df_models = execute_sql(sql)
    for i_index, i_row in df_models.iterrows():
        model_id = i_row["ID"]
        execute_sql("EXEC SP_UPDATE_MODEL_STATUS '"+str(model_id)+"', 3, 1")
        df_input, df_target, df_time_steps_input, df_time_steps_target = Preprocess.main(model_id)
    
        if df_input.shape[0] == 0:
            result = 4
        else:
            tensor_input_train, tensor_input_test, tensor_target_train, tensor_target_test = train_test_split(df_input, df_target, test_size=test_ratio, shuffle=False)
        
            df_test_index = tensor_input_test.index
    
            feature_size_x, window_length_x = Preprocess.get_dimension_size(df_time_steps_input)
            feature_size_y, window_length_y = Preprocess.get_dimension_size(df_time_steps_target)
    
            scaler_input = MinMaxScaler()
            scaler_target = MinMaxScaler()
            
            scaler_input.fit(tensor_input_train)
            scaler_target.fit(tensor_target_train)
            
            scaled_input_train = scaler_input.transform(tensor_input_train)
            scaled_target_train = scaler_target.transform(tensor_target_train)
            
            scaler_input.partial_fit(tensor_input_test)
            scaler_target.partial_fit(tensor_target_test)
            scaled_input_test = scaler_input.transform(tensor_input_test)
            
            prediction = Neural_Attention_Mechanism.main(model_id, scaled_input_train, scaled_target_train, scaled_input_test, feature_size_x, feature_size_y, window_length_x, window_length_y)
            
            prediction = scaler_target.inverse_transform(prediction)
            prediction = pd.DataFrame(prediction)
            prediction["INDEX"] = df_test_index
            prediction = prediction.set_index("INDEX")
            prediction.columns = tensor_target_test.columns
            
            scaler_file_input = scalers_dir +  str(model_id) + ' input.sav'
            scaler_file_target = scalers_dir +  str(model_id) + ' target.sav'
            
            pickle.dump(scaler_input, open(scaler_file_input, 'wb'))
            pickle.dump(scaler_target, open(scaler_file_target, 'wb'))
            
            actual = tensor_target_test
            
            sql ="EXEC SP_DELETE_TESTS " + str(model_id) + "; "
            
            df_time_steps_target = df_time_steps_target.transpose()
            for i_index, i_row in df_time_steps_target.iterrows():   
                boundary = i_row["BOUNDARY"]
                
                if boundary == 1:
                    time_step_id = i_row["ID"]
                                        
                    y_true = np.array(actual[time_step_id])
                    y_pred = np.array(prediction[time_step_id])
                    
                    mse = metrics.mean_squared_error(y_true, y_pred)
                    mae = metrics.mean_absolute_error(y_true, y_pred) 
                    r = np.corrcoef(y_true, y_pred)[0, 1]
                    r2 = r**2
                    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
                    count = len(y_true)
                    exp_var = metrics.explained_variance_score(y_true, y_pred)
                        
                    sql =sql + "EXEC SP_ADD_TEST "+str(time_step_id)+", 1 ,"+ str(count) + "; "                    
                    sql =sql + "EXEC SP_ADD_TEST "+str(time_step_id)+", 2 ,"+ str(r2)+ "; "                    
                    sql =sql + "EXEC SP_ADD_TEST "+str(time_step_id)+", 3 ,"+ str(mae)+ "; "                         
                    sql =sql + "EXEC SP_ADD_TEST "+str(time_step_id)+", 4 ,"+ str(mse)+ "; "                    
                    sql =sql + "EXEC SP_ADD_TEST "+str(time_step_id)+", 5 ,"+ str(mape)+ "; "                    
                    sql =sql + "EXEC SP_ADD_TEST "+str(time_step_id)+", 6 ,"+ str(exp_var)+ "; "
                    sql =sql + "EXEC SP_ADD_TEST "+str(time_step_id)+", 7 ,"+ str(r)+ "; "
    
        execute_sql(sql,"none")
        result = 5
        execute_sql("EXEC SP_UPDATE_MODEL_STATUS '"+str(model_id)+"', "+ str(result) +", 1")
        
    
main()