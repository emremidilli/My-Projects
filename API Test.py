# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 01:21:21 2020

@author: Yunus Emre Midilli
"""

import flask
from flask import request
import numpy as np
from sklearn import metrics
import json
from Connect_to_Database import execute_sql

error_msg = "Error: Please enter all fields correctly."
app = flask.Flask(__name__)
app.config["DEBUG"] = True
prefix = '/api/v1/'
host_ip = '192.168.1.37'


@app.route(prefix + 'models', methods=['GET'])
def get_models():
    models= execute_sql("SELECT * FROM VW_MODELS","json")
    return models


# http://10.2.65.145:5000/api/v1/models?model_id=1
@app.route(prefix +'model_details', methods=['GET'])
def get_model_details():
    
    if 'model_id' in request.args:
        model_id = request.args['model_id']
    else:
        return error_msg  + 'model_id'
    
    model_details= execute_sql("SELECT * FROM VW_MODELS WHERE ID = "  + model_id,"json")
    return model_details
 


@app.route(prefix + 'streams', methods=['GET'])
def get_streams():
    streams= execute_sql("SELECT DISTINCT STREAM_ID, STREAM_SHORT_DESCRIPTION FROM VW_FEATURES","json")
    return streams



# http://10.2.65.145:5000/api/v1/features?stream_id=1
@app.route(prefix +'features', methods=['GET'])
def get_features():
    
    if 'stream_id' in request.args:
        stream_id = request.args['stream_id']
    else:
        return error_msg  + 'stream_id'
    
    features= execute_sql("SELECT * FROM TBL_FEATURES WHERE STREAM_ID = "  + stream_id,"json")
    return features


# http://10.2.65.145:5000/api/v1/model_features?model_id=1&model_feature_type_id=1
@app.route(prefix +'model_features', methods=['GET'])
def get_model_features():
    
    if 'model_id' in request.args:
        model_id = request.args['model_id']
    else:
        return error_msg  + 'model_id'
    
    
    if 'model_feature_type_id' in request.args:
        model_feature_type_id = request.args['model_feature_type_id']
    else:
        return error_msg  + 'model_feature_type_id'
    
    
    model_features= execute_sql("select * from FN_GET_MODEL_FEATURES( "+ model_id +" , "+ model_feature_type_id +")","json")
    return model_features


# http://10.2.65.145:5000/api/v1/model_features_time_steps?model_feature_id=1
@app.route(prefix +'model_features_time_steps', methods=['GET'])
def get_model_feature_time_steps():
    
    if 'model_feature_id' in request.args:
        model_feature_id = request.args['model_feature_id']
    else:
        return error_msg + 'model_feature_id'
    
    sql = "EXEC SP_GET_MODEL_FEATURE_TIME_STEPS "+ model_feature_id +" "
    
    
    return execute_sql(sql,"json")


# http://10.2.65.145:5000/api/v1/define_model_feature?model_feature_id=1&model_id=1&feature_id=1&filter=NULL&feature_type_id=1&window_length_min=-5&window_length_max=-1
@app.route(prefix +'define_model_feature', methods=['GET'])
def define_model_feature():
    
    if 'model_feature_id' in request.args:
        model_feature_id = request.args['model_feature_id']
    else:
        return error_msg + 'model_feature_id'
    
    if 'model_id' in request.args:
        model_id = request.args['model_id']
    else:
        return error_msg + 'model_id'
    
    if 'feature_id' in request.args:
        feature_id = request.args['feature_id']
    else:
        return error_msg + 'feature_id'
    
    if 'filter' in request.args:
        filter_val = request.args['filter']
    else:
        return error_msg + 'filter'
    
    if 'feature_type_id' in request.args:
        feature_type_id = request.args['feature_type_id']
    else:
        return error_msg + 'feature_type_id'
    
    if 'window_length_min' in request.args:
        window_length_min = request.args['window_length_min']
    else:
        return error_msg + 'window_length_min'
    
    if 'window_length_max' in request.args:
        window_length_max = request.args['window_length_max']
    else:
        return error_msg + 'window_length_max'
    
    sql = ""
    if model_feature_id == "" or model_feature_id == "null":
        sql = "EXEC SP_ADD_MODEL_FEATURE "+ model_id +" , "+ feature_id +", '"+ filter_val +"', "+ feature_type_id +", "+ window_length_min +"," + window_length_max   
    else:
        sql = "EXEC SP_UPDATE_MODEL_FEATURE "+model_feature_id +", "+ model_id +" , "+ feature_id +", '"+ filter_val +"', "+ feature_type_id +", "+ window_length_min +"," + window_length_max
        
    
    return execute_sql(sql,"json")


# http://10.2.65.145:5000/api/v1/model_feature_details?model_feature_id=1
@app.route(prefix +'model_feature_details', methods=['GET'])
def get_model_feature_details():
    if 'model_feature_id' in request.args:
        model_feature_id = request.args['model_feature_id']
    else:
        return error_msg + 'model_feature_id'
    
    sql = "SELECT * FROM FN_GET_MODEL_FEATURE_DETAILS ("+ model_feature_id +")"
    
    return execute_sql(sql,"json")



# http://10.2.65.145:5000/api/v1/delete_model_feature?model_feature_id=1"
@app.route(prefix +'delete_model_feature', methods=['GET'])
def delete_model_feature():
    if 'model_feature_id' in request.args:
        model_feature_id = request.args['model_feature_id']
    else:
        return error_msg + 'model_feature_id'
    
    sql = "EXEC SP_DELETE_MODEL_FEATURE '"+ model_feature_id +"' "

    
    return execute_sql(sql,"json")



# http://10.2.65.145:5000/api/v1/define_model?model_id=1&model_short_description=bla bla bla&model_long_description=bla bla bla,"
@app.route(prefix +'define_model', methods=['GET'])
def define_model():
    if 'model_id' in request.args:
        model_id = request.args['model_id']
    else:
        return error_msg + 'model_id'
    
    if 'model_short_description' in request.args:
        model_short_description = request.args['model_short_description']
    else:
        return error_msg + 'model_short_description'
    
    if 'model_long_description' in request.args:
        model_long_description = request.args['model_long_description']
    else:
        return error_msg + 'model_long_description'
    
    sql = ""
    if model_id == "":
        sql = "EXEC SP_ADD_MODEL '"+ model_short_description +"' , '"+ model_long_description +"'"
    else:
        sql = "EXEC SP_UPDATE_MODEL "+ model_id +" , '"+ model_short_description +"' , '"+ model_long_description +"'"
        
    
    return execute_sql(sql,"json")



# http://10.2.65.145:5000/api/v1/delete_model?model_id=1"
@app.route(prefix +'delete_model', methods=['GET'])
def delete_model():
    if 'model_id' in request.args:
        model_id = request.args['model_id']
    else:
        return error_msg + 'model_id'
    
    sql = "EXEC SP_DELETE_MODEL '"+ model_id +"' "

    
    return execute_sql(sql,"json")


# http://10.2.65.145:5000/api/v1/update_model_status?model_id=1&status_id=3&created_by=1"
@app.route(prefix +'update_model_status', methods=['GET'])
def update_model_status():
    if 'model_id' in request.args:
        model_id = request.args['model_id']
    else:
        return error_msg + 'model_id'
    
    
    if 'status_id' in request.args:
        status_id = request.args['status_id']
    else:
        return error_msg + 'status_id'
    
    
    if 'created_by' in request.args:
        created_by = request.args['created_by']
    else:
        return error_msg + 'created_by'
    
    sql = "EXEC SP_UPDATE_MODEL_STATUS '"+ model_id +"', '"+status_id+"', '"+created_by+"' "

    
    return execute_sql(sql,"json")



@app.route(prefix +'read_from_db', methods=['GET'])
def read_from_db():
    if 'sql' in request.args:
        sql = request.args['sql']
    else:
        return error_msg
    
    return execute_sql(sql,"json")

@app.route(prefix +'write_to_db', methods=['GET'])
def write_to_db():
    if 'sql' in request.args:
        sql = request.args['sql']
    else:
        return error_msg
    
    return execute_sql(sql, "json")




# http://10.2.65.145:5000/api/v1/prediction?model_feature_id=1&time_step=3&from_date=2019-08-19 00:00&to_date=2019-08-23 00:00"
@app.route(prefix +'prediction', methods=['GET'])
def get_predictions():
    if 'model_feature_id' in request.args:
        model_feature_id = request.args['model_feature_id']
    else:
        return error_msg
    
    if 'time_step' in request.args:
        time_step = request.args['time_step']
    else:
        return error_msg   
    
    if 'from_date' in request.args:
        from_date = request.args['from_date']
    else:
        return error_msg   
    
    if 'to_date' in request.args:
        to_date = request.args['to_date']
    else:
        return error_msg
    
    sql = "EXEC SP_GET_PREDICTION_VALUES "+ model_feature_id +", '"+ from_date +"', '"+ to_date +"', "+ time_step +""
    df_predictions = execute_sql(sql)
     
    if df_predictions.shape[0]>0:
        
        y_true, y_pred = np.array(df_predictions["ACTUAL_VALUE"]), np.array(df_predictions["PREDICTION_VALUE"])
        
        mse = metrics.mean_squared_error(y_true, y_pred)
        mae = metrics.mean_absolute_error(y_true, y_pred) 
        r = np.corrcoef(y_true, y_pred)[0, 1]
        r2 = r**2
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        count = len(y_true)
        exp_var = metrics.explained_variance_score(y_true, y_pred)
        
        df_predictions["COUNT"] =count    
        df_predictions["MSE"] =mse
        df_predictions["MAE"] =mae
        df_predictions["R"] =r
        df_predictions["R2"] =r2
        df_predictions["MAPE"] =mape
        df_predictions["EXP_VAR"] =exp_var

    return df_predictions.to_json(orient='records')
    


# http://10.2.65.145:5000/api/v1/test_statistics?model_feature_id=1"
@app.route(prefix +'test_statistics', methods=['GET'])
def get_test_statistics():
    
    if 'model_feature_id' in request.args:
        model_feature_id = request.args['model_feature_id']
    else:
        return error_msg
    
    sql = "select * from FN_GET_TEST_STATISTICS("+ str(model_feature_id) +")"
    df = execute_sql(sql)
    
    df = df.pivot(index='TIME_STEP', columns='METRIC_ID')['VALUE']
    df_json = json.loads(df.to_json(orient='table'))


    return  json.dumps(df_json['data'])

# app.run(ssl_context='adhoc')
app.run(host= host_ip)
