# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 02:42:17 2020

@author: yunus emre midilli
"""

import Train
import Preprocess
from Connect_to_Database import execute_sql
import Neural_Attention_Mechanism
import pickle
import pandas as pd


def main():
    sql = "SELECT * FROM VW_MODELS WHERE LATEST_STATUS_ID = 5"
    df_models = execute_sql(sql, "")
    for i_index, i_row in df_models.iterrows():
        model_id = i_row["ID"]

        sql = "SELECT * FROM FN_GET_MODEL_FEATURES("+ str(model_id) +", 2)"
        df_output_model_features = execute_sql(sql, "")
        for j_index, j_row in df_output_model_features.iterrows():
            model_feature_id = j_row["ID"]

            sql = "EXEC SP_GET_FEATURE_VALUES_TO_BE_PREDICTED " + str(model_feature_id)
            
            df_to_be_predicted = execute_sql(sql, "")
            df_to_be_predicted = df_to_be_predicted.set_index("TIME_STAMP")

            if df_to_be_predicted.shape[0] > 0:
                from_time_stamp = df_to_be_predicted.index.min()
                to_time_stamp = df_to_be_predicted.index.max()
                
                df_input, df_time_steps_input = Preprocess.dfGetFeatureValues(model_id, "1",from_time_stamp, to_time_stamp)
                df_output , df_time_steps_target = Preprocess.dfGetFeatureValues(model_id, "2",from_time_stamp, to_time_stamp)
                
                feature_size_x, window_length_x = Preprocess.dfGetDimensionSize(df_time_steps_input)
                feature_size_y , window_length_y = Preprocess.dfGetDimensionSize(df_time_steps_target)
                
                df_input_index = df_input.index            
                df_input = df_input[df_input.index.isin(df_to_be_predicted.index)] 
                
                if df_input.shape[0] > 0:
                                    
                    scaler_file_input = Train.gc_s_SCALERS_PATH +  str(model_id) + ' input.sav'
                    scaler_file_target = Train.gc_s_SCALERS_PATH +  str(model_id) + ' target.sav'
                    
                    scaler_input = pickle.load(open(scaler_file_input, 'rb'))
                    scaler_target = pickle.load(open(scaler_file_target, 'rb'))
                    
                    scaler_input.partial_fit(df_input)
                    
                    scaled_input = scaler_input.transform(df_input)
                    
                    
                    o_model_neural_attention = Neural_Attention_Mechanism.Neural_Attention_Mechanism(str(model_id), feature_size_x, feature_size_y, window_length_x, window_length_y)
                    prediction = o_model_neural_attention.predict(scaled_input)
                    
                    
                    prediction = scaler_target.inverse_transform(prediction)
                    prediction = pd.DataFrame(prediction)
                    prediction["INDEX"] = df_input_index
                    prediction = prediction.set_index("INDEX")
                    prediction.columns = df_output.columns
                    

                    for l_index, l_row in prediction.iterrows():
                        base_feature_value_id =df_to_be_predicted.loc[l_index]["ID"]
                        for time_step_id , predicted_value in l_row.iteritems():
                            sql_add_prediction = "EXEC SP_ADD_PREDICTION "+str(base_feature_value_id)+" ,"+str(time_step_id)+" , '"+str(predicted_value)+"' "
                            execute_sql(sql_add_prediction,"none")
    
    
                    pickle.dump(scaler_input, open(scaler_file_input, 'wb'))
                    pickle.dump(scaler_target, open(scaler_file_target, 'wb'))
                

main()
