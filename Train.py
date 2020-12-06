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
import random
from deap import base
from deap import creator
from deap import tools


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


def evaluate_fitness(individual, o_model_neural_attention,scaled_input_train,scaled_target_train,scaled_input_test, scaled_target_test):
    epoch_size = individual[0]
    batch_size = individual[1]
    number_of_hidden_neuron = individual[2]
    dropout_rate_encoder = individual[3]
    dropout_rate_decoder = individual[4]
    recurrent_dropout_rate_encoder = individual[5]
    recurrent_dropout_rate_decoder = individual[6]
    learning_rate = individual[7]
    momentum_rate = individual[8]

    o_model_neural_attention.set_hyperparameters(epoch_size, batch_size, number_of_hidden_neuron, dropout_rate_encoder, dropout_rate_decoder, recurrent_dropout_rate_encoder, recurrent_dropout_rate_decoder, learning_rate, momentum_rate)
    o_model_neural_attention.train(scaled_input_train, scaled_target_train)
    prediction = o_model_neural_attention.predict(scaled_input_test)
    
    actual = scaled_target_test
    
    zero_indices = np.argwhere(np.all(actual[..., :] == 0, axis=0))
    
    y_true = np.delete(actual, zero_indices, axis=1)
    y_pred = np.delete(prediction, zero_indices, axis=1)
    
    print(y_true)
    
    print(y_pred)
    
    mse, mae, r, r2, mape, count, exp_var = Calculate_Accuracy.main(y_true, y_pred)
    
    print(mae)
    
    return mae

    
def get_optimum_configuration(o_model_neural_attention,scaled_input_train, scaled_target_train, scaled_input_test, scaled_target_test):
    creator.create("Fitness_Function", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness = creator.Fitness_Function)
    
    toolbox = base.Toolbox()
    
    toolbox.register("rng_epoch_size", random.randint, 1, 10)
    toolbox.register("rng_batch_size", random.randint, 100,1000)
    toolbox.register("rng_number_of_hidden_neuron", random.randint, 10,1000)
    toolbox.register("rng_dropout_rate_encoder", random.uniform, 0.1, 0.7)
    toolbox.register("rng_dropout_rate_decoder", random.uniform, 0.1, 0.7)
    toolbox.register("rng_recurrent_dropout_rate_encoder", random.uniform, 0.1, 0.7)
    toolbox.register("rng_recurrent_dropout_rate_decoder", random.uniform, 0.1, 0.7)
    toolbox.register("rng_learning_rate", random.uniform, 0.001, 0.1)
    toolbox.register("rng_momentum_rate", random.uniform, 0.1, 0.9)
    
    toolbox.register("individual", tools.initCycle, creator.Individual,(toolbox.rng_epoch_size, toolbox.rng_batch_size, toolbox.rng_number_of_hidden_neuron,toolbox.rng_dropout_rate_encoder, toolbox.rng_dropout_rate_encoder, toolbox.rng_recurrent_dropout_rate_encoder, toolbox.rng_recurrent_dropout_rate_decoder, toolbox.rng_learning_rate, toolbox.rng_momentum_rate))
    toolbox.register("population", tools.initRepeat, list ,toolbox.individual)
    toolbox.register("evaluate", evaluate_fitness, o_model_neural_attention = o_model_neural_attention,scaled_input_train = scaled_input_train,scaled_target_train = scaled_target_train,scaled_input_test= scaled_input_test, scaled_target_test= scaled_target_test)
    
    
    genetic_algorithm = Optimize.genetic_algorithm()
    
    optimum_result = genetic_algorithm.optimize_with_genetic_algorithm(toolbox)

    epoch_size = optimum_result[0]
    batch_size = optimum_result[1]
    number_of_hidden_neuron= optimum_result[2]
    dropout_rate_encoder= optimum_result[3]
    dropout_rate_decoder= optimum_result[4]
    recurrent_dropout_rate_encoder= optimum_result[5]
    recurrent_dropout_rate_decoder= optimum_result[6]
    learning_rate= optimum_result[7]
    momentum_rate= optimum_result[8]
    
    o_model_neural_attention.set_hyperparameters(epoch_size, batch_size, number_of_hidden_neuron, dropout_rate_encoder, dropout_rate_decoder, recurrent_dropout_rate_encoder, recurrent_dropout_rate_decoder, learning_rate, momentum_rate )

    return o_model_neural_attention


main()
