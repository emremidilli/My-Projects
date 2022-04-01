import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import seaborn as sns

import matplotlib.pyplot as plt

import os

import tensorflow as tf


def __init__(sOutputSymbol, sModelType, sDesignType, iTrialId):

    sFolderPath = 'Data/'+ sOutputSymbol +'//'+ sModelType + '//'+ sDesignType+ '//'


    sModelName = os.path.join(sFolderPath + str(iTrialId))


    dfOhlc = pd.read_csv('Data\dfOhlc.csv', index_col = 0)
    dfActual = pd.read_csv(sModelName+'\dfActual.csv',header=[0, 1], index_col=0)
    dfPrediction = pd.read_csv(sModelName+'\dfPrediction.csv',header=[0, 1], index_col=0)
    dfPerformance = pd.read_csv(sModelName+'\dfPerformance.csv', index_col=0)


    def dfGetClosePricesFromReturns(dfToConvert):
        dfClose = dfToConvert.copy() 

        for sCol in dfClose.columns:
            iTimeStep = int(sCol[0])
            sSymbolFeature = sCol[1]
            aSymbolFeature = sSymbolFeature.split(':')
            sSymbol = aSymbolFeature[0]

            dfReturns = dfToConvert.loc[:,sCol ]

            if iTimeStep == 0:
                dfOpenPrices = dfOhlc.loc[dfClose.index, sSymbol+':open']
            else:
                dfOpenPrices =  dfClose.loc[:, (str(iTimeStep-1), sSymbolFeature)]

            dfClose.loc[:, sCol] = (dfOpenPrices * dfReturns) + dfOpenPrices

        dfClose.rename(columns = {sOutputSymbol+':return' :sOutputSymbol+':close'}, level=1, inplace = True)
        return dfClose

    dfActualClose = dfGetClosePricesFromReturns(dfActual)
    dfPredictionClose = dfGetClosePricesFromReturns(dfPrediction)

    def dfGetCombinationsOfReturns(dfToConvert):
        dfConverted = pd.DataFrame(index = dfToConvert.index)
        for i in range(0, dfToConvert.shape[1]):
            for j in range(i, dfToConvert.shape[1]): 
                dfConverted.loc[: , str(i) + '_' + str(j) ] = dfToConvert.iloc[:, i:j+1].sum(axis = 1)

        return dfConverted


    dfActualReturnCombinations = dfGetCombinationsOfReturns(dfActual)
    dfPredictionReturnCombinations = dfGetCombinationsOfReturns(dfPrediction)


    def fCalculateCustomMetric(aActual,aPrediction):
        aLossDueToError = tf.math.subtract(aActual,aPrediction)
        aLossDueToError = tf.where(aLossDueToError < 0 , aLossDueToError, 0 )
        aLossDueToError = tf.math.abs(aLossDueToError)

        fPenalty = tf.math.reduce_max(aLossDueToError)

        aLossDueToSignDiff = tf.math.abs(tf.math.subtract(tf.math.sign(aActual), tf.math.sign(aPrediction)) )
        aLossDueToSignDiff = tf.where(aLossDueToSignDiff == 0, aLossDueToSignDiff, fPenalty)

        aTotalLoss = aLossDueToError + aLossDueToSignDiff
        fAggLoss = tf.math.reduce_sum(aTotalLoss).numpy()
        return fAggLoss


    y_pred = dfPredictionClose.iloc[:, :]
    y_true = dfActualClose.iloc[:, :]

    dictMetrics = {
        'mean absolute error': mean_absolute_error(y_true, y_pred),
        'mean squared error': mean_squared_error(y_true, y_pred),
        'r2 score': r2_score(y_true, y_pred),
        'custom_metric' : fCalculateCustomMetric(y_true, y_pred)
    }

    dfMetrics = pd.DataFrame.from_dict(data = dictMetrics, orient = 'index', columns = ['value'])
    dfPerformance.drop(dfMetrics.index, axis = 0, inplace = True, errors = 'ignore')
    dfPerformance.append(dfMetrics).to_csv(sModelName+'\dfPerformance.csv')


    iComparisionTimeStep = 0
    dfTestComparision = pd.DataFrame(dfPredictionClose.iloc[:,iComparisionTimeStep])
    dfTestComparision = dfTestComparision.join(dfActualClose.iloc[:,iComparisionTimeStep], how = "inner", lsuffix="prediction")
    dfTestComparision.columns = ["Prediction", "Actual"]

    plt.figure(figsize = (20,10))
    oFig = sns.scatterplot(data = dfTestComparision, x = "Actual", y ="Prediction")
    oFig.get_figure().savefig(sModelName + '\closing price scatter.png')

