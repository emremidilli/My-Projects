import tensorflow as tf
import pickle
import numpy as np
import itertools
import pandas as pd
import os
import Predictive_Model


def __init__(sOutputSymbol, sModelType, ixToPredict):
    sModelFolderPath =  'Data/'+ sOutputSymbol +'//'+ sModelType + '//Optimum Design//0//'
    
    
    iBackwardTimeWindow = 3
    iForwardTimeWindow =3
    
    # LOAD DATA
    ## Crpytocurrency List
    dfCrpytocurrencies = pd.read_csv('Data\cryptocurrencies.csv')
    
    
    ## Market Data
    dfOhlc = pd.read_csv('Data\dfOhlc.csv')
    dfOhlc['timestamp'] = pd.DatetimeIndex(dfOhlc['timestamp'])
    dfOhlc.set_index('timestamp', inplace=True)
    
    ## Scale Data
    dfScaledOhlc = pd.DataFrame(index = ixToPredict, columns  = dfOhlc.columns)
    for sColumn in dfOhlc.columns:
        sScalerFilePath = os.path.join(sModelFolderPath , "__scalers__")
        sScalerFilePath = os.path.join(sScalerFilePath , sColumn + ".sav")
        
        oScaler = pickle.load(open(sScalerFilePath, 'rb'))
    
        dfToScale = pd.DataFrame(dfOhlc.loc[ixToPredict, sColumn])
    
        dfScaledOhlc.loc[ixToPredict, sColumn] = np.reshape(oScaler.transform(dfToScale), (-1))
    
    ## Create Input Dataset
    aInputSymbols = dfCrpytocurrencies['Symbol'].values
    aInputFeatures = ['weekday', 'hour', 'minute' ,'upper_shadow', 'lower_shadow' ,'return']
    aInputFeatures = list(map(":".join, itertools.product(aInputSymbols, aInputFeatures)))
    
    iNrInputFeatures = len(aInputFeatures)
    
    aBackwardTimeSteps = range(-iBackwardTimeWindow, 0)
    
    aTplInputColumns = list(itertools.product(aBackwardTimeSteps, aInputFeatures))
    aIxInputColumns = pd.MultiIndex.from_tuples(aTplInputColumns, names= ['time_step', 'feature'])
    
    dfInput = pd.DataFrame(columns = aIxInputColumns)
    
    for tplColumn in list(dfInput.columns):
        dfInput.loc[:, tplColumn] = dfScaledOhlc[(tplColumn[1])].shift(-tplColumn[0])
    
    
    ixNas = dfInput[dfInput.isna().any(axis=1)].index
    dfInput.drop(ixNas, inplace = True, errors = 'ignore') 
    ixToPredict= ixToPredict.drop(ixNas, errors = 'ignore') 
    
    
    ## Create Output Dataset
    aOutputFeatures = ['return']
    aOutputFeatures = list(map(":".join, itertools.product([sOutputSymbol], aOutputFeatures)))
    iNrOutputFeatures = len(aOutputFeatures)
    
    aForwardTimeSteps = range(0, iForwardTimeWindow)
    
    
    aTplOutputColumns = list(itertools.product(aForwardTimeSteps, aOutputFeatures))
    aIxOutputColumns = pd.MultiIndex.from_tuples(aTplOutputColumns, names= ['time_step', 'feature'])
    
    dfOutput = pd.DataFrame(columns = aIxOutputColumns)
    
    for tplColumn in list(dfOutput.columns):
        dfOutput.loc[:, tplColumn] =  dfOhlc.loc[ixToPredict][(tplColumn[1])].shift(-tplColumn[0])
    
    ixNas = dfOutput[dfOutput.isna().any(axis=1)].index
    dfOutput.drop(ixNas, inplace = True, errors = 'ignore') 
    ixToPredict= ixToPredict.drop(ixNas, errors = 'ignore') 
    
    
    ## Reshape Datasets
    axMerged = dfInput.index.join(dfOutput.index, how = 'inner')
    
    dfInput = dfInput.loc[axMerged]
    dfOutput = dfOutput.loc[axMerged]
    
    ixToPredict = ixToPredict.join(axMerged, how = "inner")
    
    
    dfInputToSimulate = dfInput.loc[ixToPredict]
    aInputToSimulate = np.reshape(dfInputToSimulate.values, (dfInputToSimulate.shape[0], iBackwardTimeWindow, iNrInputFeatures))
    
    dfOutputToSimulate = dfOutput.loc[ixToPredict]
    aOutputToSimulate = np.reshape(dfOutputToSimulate.values, (dfOutputToSimulate.shape[0], iForwardTimeWindow, iNrOutputFeatures))
    
    aInputToSimulate = np.asarray(aInputToSimulate, np.float32)
    aOutputToSimulate = np.asarray(aOutputToSimulate, np.float32)
    
    
    oPredictiveModel = tf.keras.models.load_model(sModelFolderPath+  '__model__', custom_objects={'fCalculateLoss': Predictive_Model.fCalculateLoss})
    
    
    ## Test Model
    aPrediction = oPredictiveModel.predict(aInputToSimulate)
    aPrediction = aPrediction.reshape((-1, iForwardTimeWindow * iNrOutputFeatures))
    dfPrediction = pd.DataFrame(data = aPrediction, index = ixToPredict, columns = aIxOutputColumns)
    
    aActual = aOutputToSimulate.reshape((-1, iForwardTimeWindow * iNrOutputFeatures))
    dfActual =  pd.DataFrame(data = aActual, index = ixToPredict, columns = aIxOutputColumns).copy()
    
    
    return dfPrediction, dfActual



