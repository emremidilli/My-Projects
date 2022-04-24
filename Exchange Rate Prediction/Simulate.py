import pandas as pd
import Predict
import Calculate_Metrics
import os


def __init__(dfBestModels, ixToSimulate):
    
    dfOhlc = pd.read_csv('Data\dfOhlc.csv')
    dfOhlc['timestamp'] = pd.DatetimeIndex(dfOhlc['timestamp'])
    dfOhlc.set_index('timestamp', inplace=True)
    
    # CONVERT [return] TO [close]
    def dfGetClosePricesFromReturns(dfToConvert):
        iForwardTimeWindow = dfToConvert.shape[1]
        dfClose = dfToConvert.copy() 
        
        aTimeSteps = list()
        for sCol in dfClose.columns:
            iTimeStep = int(sCol[0])
            sSymbolFeature = sCol[1]
            aSymbolFeature = sSymbolFeature.split(':')
            sSymbol = aSymbolFeature[0]

            dfReturns = dfToConvert.loc[:,sCol ]

            if iTimeStep == 0:
                dfOpen = dfOhlc.loc[dfClose.index, sSymbol+':open']
            else:
                dfOpen =  dfClose.loc[:, ((iTimeStep-1), sSymbolFeature)]

            dfClose.loc[:, sCol] = (dfOpen * dfReturns) + dfOpen
            
            
            aTimeSteps = aTimeSteps + [iTimeStep]

        dfClose.rename(columns=lambda s: str(s).replace("return", "close"), inplace=True)
    
        dfClose.columns =  [int(i) for i in aTimeSteps]
        
        dfClose[-1] = dfOhlc.loc[dfClose.index, sSymbol+':open']
        
        dfClose.sort_index(axis=1, inplace=True)
        
        
        dfClose = dfClose.iloc[::iForwardTimeWindow]

        return dfClose

    for sExchangeRate, sRow in dfBestModels.iterrows():
        sModelType = sRow['Model Type']
    
        dfPrediction , dfActual = Predict.__init__(sExchangeRate, sModelType, ixToSimulate)
    
        dfActualReturnCombinations = Calculate_Metrics.dfGetCombinationsOfReturns(dfActual)
        dfPredictionReturnCombinations = Calculate_Metrics.dfGetCombinationsOfReturns(dfPrediction)
    
        dfActualClose = dfGetClosePricesFromReturns(dfActual)
        dfPredictionClose = dfGetClosePricesFromReturns(dfPrediction)
    
        sSimulationFolderPath = 'Data/'+ sExchangeRate +'//Simulation//'
    
        os.makedirs(sSimulationFolderPath, exist_ok = True)
        
        dfActual.to_csv(sSimulationFolderPath + 'dfActualReturnIndividual.csv', index=True, index_label='timestamp')
        dfPrediction.to_csv(sSimulationFolderPath + 'dfPredictionReturnIndividual.csv', index=True, index_label='timestamp')
        
        dfActualReturnCombinations.to_csv(sSimulationFolderPath + 'dfActualReturnCombinations.csv', index=True, index_label='timestamp')
        dfPredictionReturnCombinations.to_csv(sSimulationFolderPath + 'dfPredictionReturnCombinations.csv', index=True, index_label='timestamp')
        dfActualClose.to_csv(sSimulationFolderPath + 'dfActualClose.csv', index=True, index_label='timestamp')
        dfPredictionClose.to_csv(sSimulationFolderPath + 'dfPredictionClose.csv', index=True, index_label='timestamp')
        