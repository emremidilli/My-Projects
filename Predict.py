# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 02:42:17 2020

@author: yunus
"""

import Train_Models
from Connect_to_Database import execute_sql
import Neural_Attention_Mechanism
import pickle


def main():
    sql = "SELECT * FROM VW_MODELS WHERE LATEST_STATUS_ID = 5"
    df_models = execute_sql(sql, "")
    for i_index, i_row in df_models.iterrows():
        model_id = i_row["ID"]
        df_input, df_time_steps_input= Train_Models.get_feature_values(model_id, "1")
        
        sql = "SELECT * FROM FN_GET_MODEL_FEATURES("+ str(model_id) +", 2)"
        df_output_model_features = execute_sql(sql, "")
        for j_index, j_row in df_output_model_features.iterrows():
            model_feature_id = j_row["ID"]

            sql = "EXEC SP_GET_FEATURE_VALUES_TO_BE_PREDICTED " + str(model_feature_id)
            
            df_to_be_predicted = execute_sql(sql, "")
            df_input = df_input[~df_input.index.isin(df_to_be_predicted.index)]
            
            scaler_file_input = Train_Models.scalers_dir +  str(model_id) + ' input.sav'
            scaler_file_target =Train_Models.scalers_dir +  str(model_id) + ' target.sav'
            
            scaler_input = pickle.load(open(scaler_file_input, 'rb'))
            scaler_target = pickle.load(open(scaler_file_target, 'rb'))
            
            scaler_input.partial_fit(df_input)
            
            scaled_input = scaler_input.transform(df_input)
            
            # predicted_result = Neural_Attention_Mechanism.attention_predict(scaled_input)
            # pickle.dump(scaler_input, open(scaler_file_input, 'wb'))
            # pickle.dump(scaler_target, open(scaler_file_target, 'wb'))
            
            return scaled_input
            
            
scaled_input = main()


