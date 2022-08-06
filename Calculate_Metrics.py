# LIBRARIES
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import os

import tensorflow as tf


# CONVERT [return] OF COMBINATIONS
def dfGetCombinationsOfReturns(dfToConvert):
    iForwardTimeWindow = dfToConvert.shape[1]
    dfConverted = pd.DataFrame(index = dfToConvert.index)
    for i in range(0, dfToConvert.shape[1]):
        for j in range(i, dfToConvert.shape[1]): 
            dfConverted.loc[: , str(i) + '_' + str(j) ] = dfToConvert.iloc[:, i:j+1].sum(axis = 1)
    
    dfConverted = dfConverted.iloc[::iForwardTimeWindow]
    return dfConverted

## Define Custom Metric Function
def fCalculateCustomMetric(aActual,aPrediction, iAxis = None):
    aLossDueToError = tf.math.subtract(aActual,aPrediction)
    
    iMultiplier = aActual.shape[len(aActual.shape) - 1]
    fPenalty = tf.math.abs(tf.math.reduce_max(aLossDueToError))
    fPenalty = fPenalty * iMultiplier 
    
    aLossDueToError = tf.where(aLossDueToError < 0 , aLossDueToError, 0 )
    aLossDueToError = tf.math.abs(aLossDueToError)

    

    aLossDueToSignDiff = tf.math.abs(tf.math.subtract(tf.math.sign(aActual), tf.math.sign(aPrediction)) )
    aLossDueToSignDiff = tf.where(aLossDueToSignDiff == 0, aLossDueToSignDiff, fPenalty)

    aTotalLoss = aLossDueToError + aLossDueToSignDiff
    fAggLoss = tf.math.reduce_mean(aTotalLoss, iAxis).numpy()
    return fAggLoss


def __init__(sOutputSymbol, sModelType, sDesignType, iTrialId):
    
    # CONFIGURATION
    sFolderPath = 'Data/'+ sOutputSymbol +'//'+ sModelType + '//'+ sDesignType+ '//'
    
    sModelName = os.path.join(sFolderPath + str(iTrialId))
    
    dfActual = pd.read_csv(sModelName+'\dfActual.csv',header=[0, 1], index_col=0)
    dfPrediction = pd.read_csv(sModelName+'\dfPrediction.csv',header=[0, 1], index_col=0)
    dfPerformance = pd.read_csv(sModelName+'\dfPerformance.csv', index_col=0)

    
    dfActualReturnCombinations = dfGetCombinationsOfReturns(dfActual)
    dfPredictionReturnCombinations = dfGetCombinationsOfReturns(dfPrediction)
    
    # SAVE RESULTS

    ## Log Metrics
    dictMetrics = {
        'mean absolute error':mean_absolute_error(dfActualReturnCombinations, dfActualReturnCombinations),
        'mean squared error':mean_squared_error(dfActualReturnCombinations, dfPredictionReturnCombinations),
        'r2 score':r2_score(dfActualReturnCombinations, dfPredictionReturnCombinations),
        'custom_metric': fCalculateCustomMetric(dfActualReturnCombinations, dfPredictionReturnCombinations)
        
    }
    
    dfMetrics = pd.DataFrame.from_dict(data = dictMetrics, orient = 'index', columns = ['value'])
    dfPerformance.drop(dfMetrics.index, axis = 0, inplace = True, errors = 'ignore')
    dfPerformance.append(dfMetrics).to_csv(sModelName+'\dfPerformance.csv')
    
