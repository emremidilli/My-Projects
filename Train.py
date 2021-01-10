# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 06:20:42 2020

@author: Yunus Emre Midilli
"""

import Neural_Attention_Mechanism
import pandas as pd
import numpy as np
from Connect_to_Database import execute_sql
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pickle
import Preprocess
import Calculate_Accuracy
import shutil

import Optimize
from deap import base
from deap import creator

training_ratio = 0.7
test_ratio = round(1 - training_ratio,2)
scalers_dir = './__scalers__/'

def main():
    sql ="SELECT * FROM VW_MODELS --WHERE LATEST_STATUS_ID = 2"
    
    df_models = execute_sql(sql)
    for i_index, i_row in df_models.iterrows():
        model_id = str(i_row["ID"])
        
        execute_sql("EXEC SP_UPDATE_MODEL_STATUS '"+model_id+"', 3, 1")
        df_input, df_target, df_time_steps_input, df_time_steps_target = Preprocess.main(model_id)
    
        if df_input.shape[0] == 0:
            result = 4
        else:
            
            create_directory(model_id)
            
            tensor_input_train, tensor_input_test, tensor_target_train, tensor_target_test = train_test_split(df_input, df_target, test_size=test_ratio, shuffle=False)
        
            df_test_index = tensor_input_test.index
    
            feature_size_x, window_length_x = Preprocess.get_dimension_size(df_time_steps_input)
            feature_size_y, window_length_y = Preprocess.get_dimension_size(df_time_steps_target)
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
            o_model_neural_attention = get_optimum_configuration(o_model_neural_attention,scaled_input_train, scaled_target_train, scaled_input_test,scaled_target_test)
            o_model_neural_attention.train(scaled_input_train, scaled_target_train)
            prediction = o_model_neural_attention.predict(scaled_input_test)
            
            prediction = scaler_target.inverse_transform(prediction)
            prediction = pd.DataFrame(prediction)
            prediction["INDEX"] = df_test_index
            prediction = prediction.set_index("INDEX")
            prediction.columns = tensor_target_test.columns
            
            scaler_file_input = scalers_dir +  model_id + ' input.sav'
            scaler_file_target = scalers_dir + model_id + ' target.sav'
            
            pickle.dump(scaler_input, open(scaler_file_input, 'wb'))
            pickle.dump(scaler_target, open(scaler_file_target, 'wb'))
            
            actual = tensor_target_test
            
            sql ="EXEC SP_DELETE_TESTS " + model_id + "; "
            
            
            for i_index, i_row in df_time_steps_target.iterrows():   
                boundary = i_row["BOUNDARY"]
                
                if boundary == 1:
                    time_step_id = i_row["ID"]

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


def create_directory(model_id):
    try:
        shutil.rmtree(model_id)
    except OSError as e:
         print("Error: %s - %s." % (e.filename, e.strerror))


def evaluate_fitness(individual, o_model_neural_attention, scaled_input_train,scaled_target_train,scaled_input_test, scaled_target_test):
    iEpochSize = 5
    iBatchSize = individual[0]
    iHiddenSize = individual[1]
    decDropoutRateEncoder = individual[2]
    decDropoutRateDecoder = individual[3]
    decRecurrentDropoutRateEncoder = individual[4]
    decRecurrentDropoutRateDecoder = individual[5]
    decLearningRate = individual[6]
    decMomentumRate = individual[7]
    
    actual = scaled_target_test

    iPenalty = 999999
    decMinimumRequiredR2= 0.90

    if iBatchSize > actual.shape[0]:
        return iPenalty
    else:        
        o_model_neural_attention_2 = Neural_Attention_Mechanism.Neural_Attention_Mechanism(o_model_neural_attention.model_id, o_model_neural_attention.feature_size_input, o_model_neural_attention.feature_size_target, o_model_neural_attention.backward_window_length, o_model_neural_attention.forward_window_length)
        o_model_neural_attention_2.set_hyperparameters(iEpochSize, iBatchSize, iHiddenSize, decDropoutRateEncoder, decDropoutRateDecoder, decRecurrentDropoutRateEncoder, decRecurrentDropoutRateDecoder, decLearningRate, decMomentumRate)
        o_model_neural_attention_2.train(scaled_input_train, scaled_target_train)
        prediction = o_model_neural_attention_2.predict(scaled_input_test)
        
        aOverallScores=[]
        for i in range(actual.shape[1]):
            mse, mae, r, r2, mape, count, exp_var = Calculate_Accuracy.main(actual[:,i], prediction[:,i])
            if r2<=decMinimumRequiredR2:
                return iPenalty
            else:
                aOverallScores.append(r2)
        
        print(aOverallScores)
        return np.std(aOverallScores)

    
def get_optimum_configuration(o_model_neural_attention,scaled_input_train, scaled_target_train, scaled_input_test, scaled_target_test):
    dicDecisionVariables = {
                        "label":["iBatchSize", "iHiddenSize","decDropoutRateEncoder","decDropoutRateDecoder", "decRecurrentDropoutRateEncoder", "decRecurrentDropoutRateDecoder", "decLearningRate", "decMomentumRate"],
                        "variable_type": [Optimize.variable_types.integer,Optimize.variable_types.integer,Optimize.variable_types.decimal,Optimize.variable_types.decimal,Optimize.variable_types.decimal,Optimize.variable_types.decimal,Optimize.variable_types.decimal,Optimize.variable_types.decimal ],
                        "lower_bound" : [128,10,0, 0, 0, 0, 0.0001, 0.1],
                        "upper_bound" : [512, 200, 0.3, 0.3, 0.3, 0.3, 0.01, 0.9],
                        "step_size_lower" : [5, 5, 0.05,0.05,0.05,0.05,0.0005,0.005],
                        "step_size_upper" : [20, 20, 0.20,0.20,0.20,0.20,0.0020,0.020]
                        }
    
    
    dicObjectiveFunctions =  {
                        "label":["Fitness"]
                        }
    
    
    dfDecisonVariables = pd.DataFrame(data=dicDecisionVariables)
    dfObjectiveFunctions = pd.DataFrame(data=dicObjectiveFunctions)

    creator.create("Fitness_Function", base.Fitness, weights=(-1.0,))
    
    toolbox = base.Toolbox()
    toolbox.register("evaluate", evaluate_fitness, o_model_neural_attention = o_model_neural_attention,scaled_input_train = scaled_input_train,scaled_target_train = scaled_target_train,scaled_input_test= scaled_input_test, scaled_target_test= scaled_target_test)
    
    particle_swarm_optimization = Optimize.particle_swarm_optimization(creator.Fitness_Function,dfObjectiveFunctions,dfDecisonVariables, 100,10, 0.75, 0.30, 0.40, 0.30)
    
    optimum_result = particle_swarm_optimization.optimize(toolbox)
    
    # genetic_algorithm = Optimize.genetic_algorithm(creator.Fitness_Function,dfObjectiveFunctions, dfDecisonVariables,100, 10, 0.75,0.5,0.05, 0.4)
    # optimum_result = genetic_algorithm.optimize(toolbox)

    iEpochSize = 5
    iBatchSize = optimum_result[0]
    iHiddenSize= optimum_result[1]
    decDropoutRateEncoder= optimum_result[2]
    decDropoutRateDecoder= optimum_result[3]
    decRecurrentDropoutRateEncoder= optimum_result[4]
    decRecurrentDropoutRateDecoder= optimum_result[5]
    decLearningRate= optimum_result[6]
    decMomentumRate= optimum_result[7]
    
    o_model_neural_attention.set_hyperparameters(iEpochSize, iBatchSize, iHiddenSize, decDropoutRateEncoder, decDropoutRateDecoder, decRecurrentDropoutRateEncoder, decRecurrentDropoutRateDecoder, decLearningRate, decMomentumRate )

    return o_model_neural_attention


main()
