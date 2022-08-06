import pandas as pd
import numpy as np
import Predict
import Calculate_Metrics
import os
import Optimize_Portfolio


def __init__(dfBestModels,ixToSimulate ):
    
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
    
    
    
    dfAllActualReturnIndividual = pd.DataFrame()
    dfAllPredictionReturnIndividual   = pd.DataFrame()
    
    dfAllActualReturnCombinations = pd.DataFrame()
    dfAllPredictionReturnCombinations  = pd.DataFrame()
    
    dfAllActualClose = pd.DataFrame()
    dfAllPredictionClose = pd.DataFrame()
    
    dfAllActualTest = pd.DataFrame()
    dfAllPredictionTest = pd.DataFrame()
    
    
    
    for sExchangeRate, sRow in dfBestModels.iterrows():
        sModelType = sRow['Model Type']
        sModelFolderPath =  'Data/'+ sExchangeRate +'//'+ sModelType + '//Optimum Design//0//'
        
    
        dfPredictionReturnIndividual , dfActualReturnIndividual = Predict.__init__(sExchangeRate, sModelType, ixToSimulate)

        
    
        dfActualReturnCombinations = Calculate_Metrics.dfGetCombinationsOfReturns(dfActualReturnIndividual)
        dfPredictionReturnCombinations = Calculate_Metrics.dfGetCombinationsOfReturns(dfPredictionReturnIndividual)
        
    
        dfActualClose = dfGetClosePricesFromReturns(dfActualReturnIndividual)
        dfPredictionClose = dfGetClosePricesFromReturns(dfPredictionReturnIndividual)
        
        dfActualReturnIndividual = dfActualReturnIndividual.droplevel(level = 1, axis = 1)
        dfPredictionReturnIndividual = dfPredictionReturnIndividual.droplevel(level = 1, axis = 1)
        
        
        dfActualTest = pd.read_csv(sModelFolderPath + 'dfActual.csv', header = [0, 1], index_col = 0).droplevel(axis = 1, level = 1)
        dfPredictionTest = pd.read_csv(sModelFolderPath + 'dfPrediction.csv', header = [0, 1], index_col = 0).droplevel(axis = 1, level = 1)
        
        
        
        dfActualReturnIndividual['Exchange Rate'] = sExchangeRate
        dfPredictionReturnIndividual['Exchange Rate']= sExchangeRate
        
        dfActualReturnCombinations['Exchange Rate'] = sExchangeRate
        dfPredictionReturnCombinations['Exchange Rate'] = sExchangeRate
        
        dfActualClose['Exchange Rate'] = sExchangeRate
        dfPredictionClose['Exchange Rate'] = sExchangeRate
        
        dfActualTest['Exchange Rate'] = sExchangeRate
        dfPredictionTest['Exchange Rate'] = sExchangeRate
        
        
        if len(dfAllActualReturnIndividual) == 0:
            dfAllActualReturnIndividual = dfActualReturnIndividual
            dfAllPredictionReturnIndividual =dfPredictionReturnIndividual
            
            dfAllActualReturnCombinations = dfActualReturnCombinations
            dfAllPredictionReturnCombinations = dfPredictionReturnCombinations
            
            dfAllActualClose = dfActualClose
            dfAllPredictionClose = dfPredictionClose
            
            dfAllActualTest = dfActualTest
            dfAllPredictionTest = dfPredictionTest
        else:
        
            dfAllActualReturnIndividual = dfAllActualReturnIndividual.append(dfActualReturnIndividual)
            dfAllPredictionReturnIndividual =dfAllPredictionReturnIndividual.append(dfPredictionReturnIndividual)
            
            dfAllActualReturnCombinations =dfAllActualReturnCombinations.append(dfActualReturnCombinations)
            dfAllPredictionReturnCombinations =dfAllPredictionReturnCombinations.append(dfPredictionReturnCombinations)
            
            dfAllActualClose =dfAllActualClose.append(dfActualClose)
            dfAllPredictionClose =dfAllPredictionClose.append(dfPredictionClose)
            
            dfAllActualTest =dfAllActualTest.append(dfActualTest)
            dfAllPredictionTest =dfAllPredictionTest.append(dfPredictionTest)
    
    
    

    aTimeSteps = list(dfAllActualReturnIndividual.columns[:-1])
    aExchangeRates = list(dfBestModels.index)
    
    decMaximumRisk = 0.25
    decInitialBalance = 100
    


    ixPartitionedSimulationTimeSteps = dfAllPredictionClose.index.unique()
    
    dfSimulationProfits = pd.DataFrame(
        index = ixPartitionedSimulationTimeSteps,
        columns = ['Expected', 'Realized']
    )
    

    decRealizedBalance = decInitialBalance
    decExpectedBalance = decInitialBalance
    for ixStep in ixPartitionedSimulationTimeSteps:
        dfExpectedPricesClose = pd.DataFrame(index = aExchangeRates, columns = aTimeSteps)
        dfExpectedPricesOpen = pd.DataFrame(index = aExchangeRates, columns = aTimeSteps)
        dfExpectedSpread = pd.DataFrame(index = aExchangeRates, columns = aTimeSteps)
        dfPredictionError = pd.DataFrame(index = aExchangeRates)
    
        dfRealizedPricesClose = pd.DataFrame(index = aExchangeRates, columns = aTimeSteps)
        dfRealizedPricesOpen = pd.DataFrame(index = aExchangeRates, columns = aTimeSteps)
        dfRealizedSpread = pd.DataFrame(index = aExchangeRates, columns = aTimeSteps)
    
        
        for sExchangeRate in aExchangeRates:
            sSimulationFolderPath = 'Data/'+ sExchangeRate +'//Simulation//'
            
            dfPredictionClose = dfAllPredictionClose[dfAllPredictionClose['Exchange Rate'] == sExchangeRate].iloc[:,:-1]
            dfActualClose = dfAllActualClose[dfAllActualClose['Exchange Rate'] == sExchangeRate].iloc[:,:-1]
            
            dfActualTest = dfAllActualTest[dfAllActualTest['Exchange Rate'] == sExchangeRate].iloc[:,:-1]
            dfPredictionTest = dfAllPredictionTest[dfAllPredictionTest['Exchange Rate'] == sExchangeRate].iloc[:,:-1]
            dfActualTest = Calculate_Metrics.dfGetCombinationsOfReturns(dfActualTest)
            dfPredictionTest = Calculate_Metrics.dfGetCombinationsOfReturns(dfPredictionTest)
    
            if len(dfPredictionError.columns) == 0:
                dfPredictionError[list(dfPredictionTest.columns)] = np.nan
                

            dfExpectedPricesClose.loc[sExchangeRate] = dfPredictionClose.loc[ixStep].iloc[1:].values
            dfExpectedPricesOpen.loc[sExchangeRate] = dfPredictionClose.loc[ixStep].iloc[:-1].values
            dfExpectedSpread.loc[sExchangeRate] = dfOhlc.loc[ixStep, sExchangeRate + ':spread']
    
            dfRealizedPricesClose.loc[sExchangeRate] = dfActualClose.loc[ixStep].iloc[1:].values
            dfRealizedPricesOpen.loc[sExchangeRate] = dfActualClose.loc[ixStep].iloc[:-1].values
            dfRealizedSpread.loc[sExchangeRate] = dfOhlc.loc[ixStep, sExchangeRate + ':spread']
    
            dfPredictionError.loc[sExchangeRate] = Calculate_Metrics.fCalculateCustomMetric(dfActualTest, dfPredictionTest, 0)
    
    

    
        oPmPrediction = Optimize_Portfolio.PortfolioManagement(dfExpectedPricesClose ,dfExpectedPricesOpen, dfExpectedSpread, aExchangeRates , aTimeSteps, decRealizedBalance, decMaximumRisk, dfPredictionError)
        aAmountsPrediction, aPositionTypesPredictions, dfAlgorithmHistory = oPmPrediction.Main()
    
        oPmRealized = Optimize_Portfolio.PortfolioManagement(dfRealizedPricesClose ,dfRealizedPricesOpen, dfRealizedSpread, aExchangeRates , aTimeSteps, decRealizedBalance, decMaximumRisk, dfPredictionError)
    
        fExpectedProfit = oPmPrediction.decEvaluateFitness(aAmountsPrediction, aPositionTypesPredictions)
        fRealizedProfit = oPmRealized.decEvaluateFitness(aAmountsPrediction, aPositionTypesPredictions)
        
        decRealizedBalance = decRealizedBalance + fRealizedProfit
        decExpectedBalance = decExpectedBalance + fExpectedProfit
        
        dfSimulationProfits.loc[ixStep, 'Expected'] = fExpectedProfit
        dfSimulationProfits.loc[ixStep, 'Realized'] = fRealizedProfit
        
        
    sSimulationFolderPath = 'Data//Simulation//' 
    os.makedirs(sSimulationFolderPath, exist_ok = True)
    
    dfAllActualReturnIndividual.to_csv(sSimulationFolderPath + 'dfAllActualReturnIndividual.csv', index=True, index_label='timestamp')
    dfAllPredictionReturnIndividual.to_csv(sSimulationFolderPath + 'dfAllPredictionReturnIndividual.csv', index=True, index_label='timestamp')
    
    dfAllActualReturnCombinations.to_csv(sSimulationFolderPath + 'dfAllActualReturnCombinations.csv', index=True, index_label='timestamp')
    dfAllPredictionReturnCombinations.to_csv(sSimulationFolderPath + 'dfAllPredictionReturnCombinations.csv', index=True, index_label='timestamp')
    
    dfAllActualClose.to_csv(sSimulationFolderPath + 'dfAllActualClose.csv', index=True, index_label='timestamp')
    dfAllPredictionClose.to_csv(sSimulationFolderPath + 'dfAllPredictionClose.csv', index=True, index_label='timestamp')
    
    dfAllActualTest.to_csv(sSimulationFolderPath + 'dfAllActualTest.csv', index=True, index_label='timestamp')
    dfAllPredictionTest.to_csv(sSimulationFolderPath + 'dfAllPredictionTest.csv', index=True, index_label='timestamp')
    
    
    dfSimulationProfits.index.name = 'timestamp'
    dfSimulationProfits.to_csv(sSimulationFolderPath + 'dfSimulationProfits.csv', index=True, index_label='timestamp')


