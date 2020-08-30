# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 02:42:17 2020

@author: yunus
"""
from Train_Models import get_feature_values
from Connect_to_Database import execute_sql
from Neural_Attention_Mechanism import attention_predict


def main():
    sql = "SELECT * FROM VW_MODELS WHERE LATEST_STATUS_ID = 5"
    df_models = execute_sql(sql, "")
    for i_index, i_row in df_models.iterrows():
        model_id = i_row["ID"]
        df_input, df_features_input, df_window_length_input , df_feature_ids_input = get_feature_values(model_id, "1")
        
        sql = "SELECT * FROM FN_GET_MODEL_FEATURES("+ str(model_id) +", 2)"
        df_output_model_features = execute_sql(sql, "")
        for j_index, j_row in df_output_model_features.iterrows():
            model_feature_id = j_row["ID"]

            sql = "EXEC SP_GET_FEATURE_VALUES_TO_BE_PREDICTED " + str(model_feature_id)
            
            df_to_be_predicted = execute_sql(sql, "")
            df_input = df_input[~df_input.index.isin(df_to_be_predicted.index)]
            
            
            
df_input = main()
