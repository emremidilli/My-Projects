# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 06:20:42 2020

@author: Yunus Emre Midilli
"""

import Neural_Attention_Mechanism
import Multi_Layer_Perceptron
import pandas as pd
import numpy as np
from Connect_to_Database import execute_sql
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pickle
import Preprocess
import Calculate_Accuracy
import shutil


gc_dec_TRAINING_RATIO = 0.7
gc_dec_TEST_RATIO = round(1 - gc_dec_TRAINING_RATIO,2)
gc_s_SCALERS_PATH = './__scalers__/'
gc_dt_FROM_DATE = "DEFAULT"
gc_dt_TO_DATE = "DEFAULT"



def main():
    sql ="SELECT * FROM VW_MODELS --WHERE LATEST_STATUS_ID = 2"
    
    df_models = execute_sql(sql)
    for i_index, i_row in df_models.iterrows():
        model_id = str(i_row["ID"])
        
        execute_sql("EXEC SP_UPDATE_MODEL_STATUS '"+model_id+"', 3, 1")
        df_input, df_target, df_time_steps_input, df_time_steps_target = Preprocess.main(model_id, gc_dt_FROM_DATE, gc_dt_TO_DATE)
        
        if df_input.shape[0] == 0:
            result = 4
        else:
              
            try:
                shutil.rmtree(model_id )
            except OSError as e:
                 print("Error: %s - %s." % (e.filename, e.strerror))
            
            tensor_input_train, tensor_input_test, tensor_target_train, tensor_target_test = train_test_split(df_input, df_target, test_size=gc_dec_TEST_RATIO, shuffle=False)
                        
            df_test_index = tensor_input_test.index
            
            feature_size_x, window_length_x = Preprocess.dfGetDimensionSize(df_time_steps_input)
            feature_size_y, window_length_y = Preprocess.dfGetDimensionSize(df_time_steps_target)
            df_time_steps_target = df_time_steps_target.transpose()
    
            scaler_input = MinMaxScaler()
            scaler_target = MinMaxScaler()
            
            scaler_input.fit(tensor_input_train)
            scaler_target.fit(tensor_target_train)
            
            scaled_input_train = scaler_input.transform(tensor_input_train)
            scaled_target_train = scaler_target.transform(tensor_target_train)
            
            scaler_input.partial_fit(tensor_input_test)
            scaler_target.partial_fit(tensor_target_test)
            scaled_input_test = scaler_input.transform(tensor_input_test)
            scaled_target_test = scaler_target.transform(tensor_target_test)
            
            o_model_neural_attention = Neural_Attention_Mechanism.Neural_Attention_Mechanism(model_id, feature_size_x, feature_size_y, window_length_x, window_length_y)
            o_model_neural_attention.train(scaled_input_train, scaled_target_train)
            prediction = o_model_neural_attention.predict(scaled_input_test)

            # oMultiLayerPerceptron = Multi_Layer_Perceptron.Multi_Layer_Perceptron(model_id, feature_size_x, feature_size_y, window_length_x, window_length_y)
            # oMultiLayerPerceptron.set_hyperparameters(epoch_size = 10, batch_size=50)
            # oMultiLayerPerceptron.train(scaled_input_train, scaled_target_train)
            # prediction = oMultiLayerPerceptron.dfPredict(scaled_input_test)

            prediction = scaler_target.inverse_transform(prediction)
            prediction = pd.DataFrame(prediction)
            prediction["INDEX"] = df_test_index
            prediction = prediction.set_index("INDEX")
            prediction.columns = tensor_target_test.columns
            
            scaler_file_input = gc_s_SCALERS_PATH +  model_id + ' input.sav'
            scaler_file_target = gc_s_SCALERS_PATH + model_id + ' target.sav'
            
            pickle.dump(scaler_input, open(scaler_file_input, 'wb'))
            pickle.dump(scaler_target, open(scaler_file_target, 'wb'))
            
            actual = tensor_target_test
            
            sql ="EXEC SP_DELETE_TESTS " + model_id + "; "
            
            
            for i_index, i_row in df_time_steps_target.iterrows():   
                boundary = i_row["BOUNDARY"]
                time_step_id = i_row["ID"]
                
                if boundary == 1:

                    y_true = np.array(actual[time_step_id])
                    y_pred = np.array(prediction[time_step_id])
                    
                    mse, mae, r, r2, mape, count, exp_var = Calculate_Accuracy.main(y_true, y_pred)

                    sql =sql + "EXEC SP_ADD_TEST "+str(time_step_id)+", 1 ,"+ str(count) + "; "                    
                    sql =sql + "EXEC SP_ADD_TEST "+str(time_step_id)+", 2 ,"+ str(r2)+ "; "                    
                    sql =sql + "EXEC SP_ADD_TEST "+str(time_step_id)+", 3 ,"+ str(mae)+ "; "                         
                    sql =sql + "EXEC SP_ADD_TEST "+str(time_step_id)+", 4 ,"+ str(mse)+ "; "                    
                    sql =sql + "EXEC SP_ADD_TEST "+str(time_step_id)+", 5 ,"+ str(mape)+ "; "                    
                    sql =sql + "EXEC SP_ADD_TEST "+str(time_step_id)+", 6 ,"+ str(exp_var)+ "; "
                    sql =sql + "EXEC SP_ADD_TEST "+str(time_step_id)+", 7 ,"+ str(r)+ "; "
    
        execute_sql(sql,"none")
        result = 5
        execute_sql("EXEC SP_UPDATE_MODEL_STATUS '"+model_id+"', "+ str(result) +", 1")



