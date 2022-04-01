# IMPORT LIBRARIES
import pandas as pd

import numpy as np

import itertools

import time

import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import seaborn as sns

import matplotlib.pyplot as plt

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping



def __init__(sOutputSymbol, sModelType, sDesignType, iTrialId):
    
    # CONFIGURATION
    sOutputSymbol = sOutputSymbol
    sModelType = sModelType
    sDesignType = sDesignType
    iTrialId = iTrialId

    sFolderPath = 'Data/'+ sOutputSymbol +'//'+ sModelType + '//'+ sDesignType+ '//'
    dfDesign = pd.read_csv(sFolderPath + 'Design.csv', index_col = 'Run ID')
    iBatchSize = dfDesign.loc[iTrialId, 'Batch Size']
    iNrOfHiddenNeurons = dfDesign.loc[iTrialId, 'Number of Hidden Neurons']
    iBackwardTimeWindow = 3
    iForwardTimeWindow = 3

    sModelName = os.path.join(sFolderPath + str(iTrialId))


    # LOAD DATA
    ## Crpytocurrency List
    dfCrpytocurrencies = pd.read_csv('Data\cryptocurrencies.csv')


    ## Market Data
    dfOhlc = pd.read_csv('Data\dfOhlc.csv')
    dfOhlc['timestamp'] = pd.DatetimeIndex(dfOhlc['timestamp'])
    dfOhlc.set_index('timestamp', inplace=True)


    # PREPROCESSING
    
    ## Split Data
    fTrainingRatio = 0.7
    fValidationRatio = 0.15
    fTestRatio = 0.15

    ixTrain, ixTest = train_test_split(
        dfOhlc.index,
        test_size=1-fTrainingRatio,
        shuffle=False)


    ixValidation, ixTest = train_test_split(
        ixTest,
        test_size=fTestRatio/(fTestRatio + fValidationRatio),
        shuffle=False)


    ## Scale Data
    dfScaledOhlc = pd.DataFrame(index = dfOhlc.index, columns  = dfOhlc.columns)

    for sColumn in dfOhlc.columns:
        oScaler = StandardScaler()

        dfTrain = pd.DataFrame(dfOhlc.loc[ixTrain, sColumn])
        dfValidation = pd.DataFrame(dfOhlc.loc[ixValidation, sColumn])
        dfTest = pd.DataFrame(dfOhlc.loc[ixTest, sColumn])

        oScaler.fit(dfTrain.append(dfValidation))

        dfScaledOhlc.loc[ixTrain, sColumn] = np.reshape(oScaler.transform(dfTrain), (-1))
        dfScaledOhlc.loc[ixValidation, sColumn] = np.reshape(oScaler.transform(dfValidation), (-1))
        dfScaledOhlc.loc[ixTest, sColumn] = np.reshape(oScaler.transform(dfTest), (-1))

        sScalerFilePath = os.path.join(sModelName , "__scalers__")
        sScalerFilePath = os.path.join(sScalerFilePath , sColumn + ".sav")
        os.makedirs(os.path.dirname(sScalerFilePath), exist_ok=True)

        pickle.dump(oScaler, open(sScalerFilePath, 'wb'))


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
    ixTrain= ixTrain.drop(ixNas, errors = 'ignore') 
    ixValidation= ixValidation.drop(ixNas,   errors = 'ignore') 
    ixTest = ixTest.drop(ixNas,   errors = 'ignore')


    ## Create Output Dataset
    aOutputFeatures = ['return']
    aOutputFeatures = list(map(":".join, itertools.product([sOutputSymbol], aOutputFeatures)))
    iNrOutputFeatures = len(aOutputFeatures)

    aForwardTimeSteps = range(0, iForwardTimeWindow)


    aTplOutputColumns = list(itertools.product(aForwardTimeSteps, aOutputFeatures))
    aIxOutputColumns = pd.MultiIndex.from_tuples(aTplOutputColumns, names= ['time_step', 'feature'])

    dfOutput = pd.DataFrame(columns = aIxOutputColumns)

    for tplColumn in list(dfOutput.columns):
        dfOutput.loc[:, tplColumn] =  dfOhlc[(tplColumn[1])].shift(-tplColumn[0])

    ixNas = dfOutput[dfOutput.isna().any(axis=1)].index
    dfOutput.drop(ixNas, inplace = True, errors = 'ignore') 
    ixTrain= ixTrain.drop(ixNas, errors = 'ignore') 
    ixValidation= ixValidation.drop(ixNas,   errors = 'ignore') 
    ixTest = ixTest.drop(ixNas,   errors = 'ignore') 



    ## Reshape Datasets
    axMerged = dfInput.index.join(dfOutput.index, how = 'inner')

    dfInput = dfInput.loc[axMerged]
    dfOutput = dfOutput.loc[axMerged]

    ixTrain = ixTrain.join(axMerged, how = "inner")
    ixValidation = ixValidation.join(axMerged, how = "inner")
    ixTest = ixTest.join(axMerged, how = "inner")


    dfInputTrain = dfInput.loc[ixTrain]
    aInputTrain = np.reshape(dfInputTrain.values, (dfInputTrain.shape[0], iBackwardTimeWindow, iNrInputFeatures))

    dfInputValidation = dfInput.loc[ixValidation]
    aInputValidation = np.reshape(dfInputValidation.values, (dfInputValidation.shape[0], iBackwardTimeWindow, iNrInputFeatures))

    dfInputTest = dfInput.loc[ixTest]
    aInputTest = np.reshape(dfInputTest.values, (dfInputTest.shape[0], iBackwardTimeWindow, iNrInputFeatures))

    dfOutputTrain = dfOutput.loc[ixTrain]
    aOutputTrain = np.reshape(dfOutputTrain.values, (dfOutputTrain.shape[0], iForwardTimeWindow, iNrOutputFeatures))

    dfOutputValidation = dfOutput.loc[ixValidation]
    aOutputValidation = np.reshape(dfOutputValidation.values, (dfOutputValidation.shape[0], iForwardTimeWindow, iNrOutputFeatures))

    dfOutputTest = dfOutput.loc[ixTest]
    aOutputTest = np.reshape(dfOutputTest.values, (dfOutputTest.shape[0], iForwardTimeWindow, iNrOutputFeatures))


    aInputTrain = np.asarray(aInputTrain, np.float32)
    aInputValidation = np.asarray(aInputValidation, np.float32)
    aInputTest = np.asarray(aInputTest, np.float32)
    aOutputTrain = np.asarray(aOutputTrain, np.float32)
    aOutputValidation = np.asarray(aOutputValidation, np.float32)
    aOutputTest = np.asarray(aOutputTest, np.float32)

    # MODEL DEVELOPMENT
    
    ## Set Early Stopping
    oEarlyStop = EarlyStopping(
        monitor = 'val_loss', 
        mode = 'min', 
        verbose = 0 , 
        patience = 20, 
        restore_best_weights = True)



    ## Define Custom Loss Function
    
    ### While loss function is defined following criteria is taken into consideration:

    ### Opposite signs should be penalized.
    ### Opposite sings will be worse when the magnitute of error increases.
    ### Any of same sign is better than any of the opposite signs.
    ### Same sign is the best when the error is 0.
    ### Following logic also should have been implemented but it was unsuccessful to implement due to forcing negative errors. It will be used as 'metric' function.
    
    ### Same sign is positive error is better than negative error (err = act - pred )
    
    def fCalculateLoss(aActual, aPrediction):
        aLossDueToError = tf.math.subtract(aActual ,aPrediction)
        aLossDueToError = tf.math.abs(aLossDueToError)

        fPenalty = tf.math.reduce_max(aLossDueToError)

        aLossDueToSignDiff = tf.math.abs(tf.math.subtract(tf.math.sign(aActual), tf.math.sign(aPrediction)) )
        aLossDueToSignDiff = tf.where(aLossDueToSignDiff == 0, aLossDueToSignDiff, fPenalty)

        aTotalLoss = aLossDueToError + aLossDueToSignDiff

        return tf.math.reduce_mean(aTotalLoss)
    
    ## BUILD MODEL
    
    ### MLP
    if sModelType == 'MLP':
        aInputMlp = keras.Input(
            shape=(iBackwardTimeWindow, iNrInputFeatures))

        aW = keras.layers.Flatten()(aInputMlp)
        aW = keras.layers.Dense(iNrOfHiddenNeurons)(aW)
        aW = keras.layers.Dense(iForwardTimeWindow*iNrOutputFeatures)(aW)
        aW = keras.layers.Reshape((iForwardTimeWindow, iNrOutputFeatures))(aW)

        aOutputMlp = aW
        oModelMlp = keras.Model(
            inputs=aInputMlp,
            outputs=aOutputMlp
        )

        oOptimizerMlp = tf.keras.optimizers.Adam(learning_rate=1e-04)
        oModelMlp.compile(optimizer=oOptimizerMlp,
                                 loss = fCalculateLoss
                                )

        oPredictiveModel = oModelMlp

        tf.keras.utils.plot_model(oModelMlp,  show_shapes=True, to_file=sModelName +'\Model architecture.png')


    ## Fit Model
    iEpochSize = 10000
    dtStartTime = time.time()
    oPredictiveModel.fit(
        aInputTrain, 
        aOutputTrain, 
        epochs=iEpochSize, 
        batch_size=iBatchSize, 
        verbose=0, 
        validation_data= (aInputValidation, aOutputValidation),
        validation_batch_size= iBatchSize
        ,callbacks=[oEarlyStop]
    )
    dtEndTime = time.time()
    dtTrainingDuration = dtEndTime -dtStartTime



    ## Save Epoch History
    plt.figure(figsize = (20,10))
    dfHistory = pd.DataFrame(oPredictiveModel.history.history)
    oFig = sns.lineplot(data = dfHistory)
    oFig.get_figure().savefig(sModelName + '\epochs.png')
    dfHistory.to_csv(sModelName + '\dfHistory.csv')



    ## Save Model
    oPredictiveModel.save_weights(sModelName+'\model weights')



    ## Test Model
    oPredictiveModel.load_weights(sModelName+'\model weights')

    aPrediction = oPredictiveModel.predict(aInputTest)
    aPrediction = aPrediction.reshape((-1, iForwardTimeWindow * iNrOutputFeatures))
    dfPrediction = pd.DataFrame(data = aPrediction, index = ixTest, columns = aIxOutputColumns)

    aActual = aOutputTest.reshape((-1, iForwardTimeWindow * iNrOutputFeatures))
    dfActual =  pd.DataFrame(data = aActual, index = ixTest, columns = aIxOutputColumns).copy()


    ## Save Results
    dfActual.to_csv(sModelName + '\dfActual.csv')
    dfPrediction.to_csv(sModelName + '\dfPrediction.csv')

    dfPerformance = pd.DataFrame(data = [dtTrainingDuration], columns = ['value'], index = ['training duration'] )
    dfPerformance.to_csv(sModelName + '\dfPerformance.csv')

