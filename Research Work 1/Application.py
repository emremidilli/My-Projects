import pandas as pd
from datetime import datetime
import MetaTrader5 as mt5
import pytz
import sys
import pickle
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from Optimize_Portfolio import PortfolioManagement
from Long_Short_Term_Memory import Long_Short_Term_Memory
from Neural_Attention_Mechanism import Neural_Attention_Mechanism


gc_o_TIME_ZONE = pytz.timezone("Etc/UTC")
gc_dt_FROM = datetime(2020, 1, 1, tzinfo=gc_o_TIME_ZONE)
gc_dt_TO = datetime(2021, 1, 5, tzinfo=gc_o_TIME_ZONE)
gc_dt_SIMULATION_MODEL_FROM = "2021-01-01 00:00:00"


gc_a_SYMBOLS = ['WTI','BRENT', 'NAT.GAS'] #['ETHUSD', 'LTCUSD', 'XRPUSD', 'BTCUSD']

gc_a_SEASONAL_FEATURES = ["year", "month", "day", "dayofweek", "hour"]
gc_a_MARKET_FEATURES = ['open', 'high', 'low', 'close']
gc_a_INPUT_FEATURES = gc_a_MARKET_FEATURES + gc_a_SEASONAL_FEATURES
gc_a_METRICS = ["mse", "rmse", "mae", "r2", "mape", "accuracy", "precision", "recall", "f1-score"]


gc_i_BACKWARD_TIME_WINDOW = -4
gc_i_FORWARD_TIME_WINDOW = 4


gc_dec_TRAINING_RATIO = 0.6
gc_dec_VALIDATION_RATIO = 0.2
gc_dec_TEST_RATIO = round(1 - (gc_dec_TRAINING_RATIO + gc_dec_VALIDATION_RATIO), 2)


gc_dec_MAX_RISK_RMSE = 0.10
gc_dec_INITIAL_BALANCE = 1000


g_aBackwardTimeSteps = range(gc_i_BACKWARD_TIME_WINDOW, 0)
g_aForwardTimeSteps = range(0, gc_i_FORWARD_TIME_WINDOW)

gc_i_PERIODS_OF_CLASSES = 10
gc_dec_LIMIT_OF_CLASSES = 0.025


def ConvertSpreadValues(dfRates, aSymbolInfo):
    iDigits = aSymbolInfo.digits
    dfRates['spread'] = dfRates['spread'] * pow(10, -iDigits)


def dfShiftTimeSteps(dfRates, aTimeSteps, aFeatures):

    dicColumnIndices = pd.MultiIndex.from_product(
        [aTimeSteps, aFeatures], names=["time step", "feature"])

    dfShiftedRates = pd.DataFrame(
        columns=dicColumnIndices, index=dfRates.index)

    for i in aTimeSteps:
        dfShiftedRates[i] = dfRates.shift(-i)

    dfShiftedRates.dropna(inplace=True)

    return dfShiftedRates


def dfGetMarketData(sSymbol):

    if not mt5.initialize():
        print("initialize() failed, error code =", mt5.last_error())
        sys.exit()

    aSymbolInfo = mt5.symbol_info(sSymbol)
    if not aSymbolInfo:
        print("symbol_info() failed, error code =", mt5.last_error())
        sys.exit()

    aRates = mt5.copy_rates_range(
        sSymbol, mt5.TIMEFRAME_H4, gc_dt_FROM, gc_dt_TO)
    if len(aRates) == 0:
        print("copy_rates_range() failed, error code =", mt5.last_error())
        sys.exit()

    mt5.shutdown()

    dfRates = pd.DataFrame(aRates)

    dfRates['time'] = pd.to_datetime(dfRates['time'], unit='s')
    dfRates.set_index('time', inplace=True)
    dfRates.drop('real_volume', axis=1, inplace=True)

    ConvertSpreadValues(dfRates, aSymbolInfo)
    AddSeasonalFeatures(dfRates)
    AddReturns(dfRates)

    return dfRates


def AddSeasonalFeatures(dfRates):

    for sSeasonalFeature in gc_a_SEASONAL_FEATURES:
        exec("dfRates[sSeasonalFeature] = dfRates.index." + sSeasonalFeature)


def AddReturns(dfRates):
    dfRates["return"] = (dfRates["open"] - dfRates["close"])/dfRates["open"]
    


def AddReturnClassLabels(dfToClassify, aClassLabels):
    
    oIntervalNegativeEnd= aClassLabels[0]
    oIntervalPositiveEnd = aClassLabels[len(aClassLabels)-1]
    
    decNegativeEnd = oIntervalNegativeEnd.left
    decPositiveEnd = oIntervalPositiveEnd.right

    dfLabels = pd.cut(dfToClassify["return"], bins = aClassLabels, labels = False)
    dfLabels[dfToClassify["return"]<=decNegativeEnd] = oIntervalNegativeEnd
    dfLabels[dfToClassify["return"]>decPositiveEnd] =oIntervalPositiveEnd
    

    dfLabels = pd.get_dummies(dfLabels)
    
    dfToClassify[aClassLabels] = dfLabels


def dfPredict(sSymbol, oPredictiveModel, dfInput, dtOutputIndices, dfOutputColumns):
    
    sScalersDirectory = os.path.join(sSymbol, "__scalers__")
    
    sScalerFilePathInput =os.path.join(sScalersDirectory, "input.sav")
    sScalerFilePathOutput = os.path.join(sScalersDirectory, "target.sav")

    oScalerInput = pickle.load(open(sScalerFilePathInput, 'rb'))
    oScalerOutput = pickle.load(open(sScalerFilePathOutput, 'rb'))

    oScalerInput.partial_fit(dfInput)

    dfInputScaled = oScalerInput.transform(dfInput)

    _, aScaledPrediction = oPredictiveModel.aPredict(dfInputScaled)

    aPrediction = oScalerOutput.inverse_transform(aScaledPrediction)

    pickle.dump(oScalerInput, open(sScalerFilePathInput, 'wb'))
    pickle.dump(oScalerOutput, open(sScalerFilePathOutput, 'wb'))

    dfPrediction = pd.DataFrame(
        aPrediction, index=dtOutputIndices, columns=dfOutputColumns)
    dfScaledPrediction = pd.DataFrame(
        aScaledPrediction, index=dtOutputIndices, columns=dfOutputColumns)

    return dfPrediction, dfScaledPrediction


def dfGetMetrics(dfActual, dfPrediction):

    dfMetrics = pd.DataFrame(columns=dfActual.columns, index=gc_a_METRICS)

    dfMetrics.loc["mse"] = metrics.mean_squared_error(
        dfActual, dfPrediction, multioutput='raw_values')

    dfMetrics.loc["mae"] = metrics.mean_absolute_error(
        dfActual, dfPrediction, multioutput='raw_values')

    dfMetrics.loc["r2"] = metrics.r2_score(
        dfActual, dfPrediction, multioutput='raw_values')

    dfMetrics.loc["mape"] = metrics.mean_absolute_percentage_error(
        dfActual, dfPrediction, multioutput='raw_values')

    dfMetrics.loc["rmse"] = metrics.mean_squared_error(
        dfActual, dfPrediction, multioutput='raw_values', squared=False)
    
    
    dfColumns = dfActual.columns
    
    for i in range(len(dfColumns)):
        sColumn = dfColumns[i]
        aActual = dfActual[sColumn]
        aPredicted = dfPrediction[sColumn]
        decAccuracy = metrics.accuracy_score(aActual, aPredicted)
        decPrecision = metrics.precision_score(aActual, aPredicted, zero_division=0)
        decRecall = metrics.recall_score(aActual, aPredicted, zero_division=0)
        decF1Score = metrics.f1_score(aActual, aPredicted, zero_division=0)
        
        dfMetrics.loc["accuracy"][sColumn] = decAccuracy
        dfMetrics.loc["precision"][sColumn] = decPrecision
        dfMetrics.loc["recall"][sColumn] = decRecall
        dfMetrics.loc["f1-score"][sColumn] = decF1Score
        
        
    return dfMetrics




def dfTrain(bIsClassification = False):
    
    if bIsClassification == False:
        aOutputFeatures = ['open', 'close', 'spread']
    else:
        aOutputFeatures = pd.interval_range(start=-gc_dec_LIMIT_OF_CLASSES, periods = gc_i_PERIODS_OF_CLASSES, end=gc_dec_LIMIT_OF_CLASSES)

    
    dfUnscaledMetrics = pd.DataFrame()
    dfScaledMetrics = pd.DataFrame()
    serModels = pd.Series(dtype=object)
    
    
    dfSimulationModelInput = pd.DataFrame()
    dfSimulationModelOutput = pd.DataFrame()
    
    for sSymbol in gc_a_SYMBOLS:
    
        dfRates = dfGetMarketData(sSymbol)
        
        if bIsClassification == True:
            AddReturnClassLabels(dfRates, aOutputFeatures)
        
    
        dfInput = dfShiftTimeSteps(dfRates, g_aBackwardTimeSteps, gc_a_INPUT_FEATURES)
        dfOutput = dfShiftTimeSteps(dfRates, g_aForwardTimeSteps, aOutputFeatures)
        dfMerged = pd.merge(dfInput, dfOutput, left_index=True, right_index=True)
    
        dfPredictiveModelInput = dfMerged[dfInput.columns][:gc_dt_SIMULATION_MODEL_FROM]
        dfPredictiveModelOutput = dfMerged[dfOutput.columns][:gc_dt_SIMULATION_MODEL_FROM]
    
        dfTempSimulationInput = pd.DataFrame(
            data=dfMerged[dfInput.columns][gc_dt_SIMULATION_MODEL_FROM:].values,
            columns=pd.MultiIndex.from_product([[sSymbol], g_aBackwardTimeSteps, gc_a_INPUT_FEATURES], names=[
                                                "symbol", "time step", "feature"]),
            index=dfMerged[dfInput.columns][gc_dt_SIMULATION_MODEL_FROM:].index
        )
    
        dfTempSimulationOutput = pd.DataFrame(
            data=dfMerged[dfOutput.columns][gc_dt_SIMULATION_MODEL_FROM:].values,
            columns=pd.MultiIndex.from_product([[sSymbol], g_aForwardTimeSteps, aOutputFeatures], names=[
                                                "symbol", "time step", "feature"]),
            index=dfMerged[dfOutput.columns][gc_dt_SIMULATION_MODEL_FROM:].index
        )
    
        if len(dfSimulationModelInput) == 0:
            dfSimulationModelInput = dfTempSimulationInput
            dfSimulationModelOutput = dfTempSimulationOutput
        else:
            dfSimulationModelInput = dfSimulationModelInput.join(dfTempSimulationInput)
            dfSimulationModelOutput = dfSimulationModelOutput.join(dfTempSimulationOutput)
    
        dfInputTrainValidation, dfInputTest, dfOutputTrainValidation, dfOutputTest = train_test_split(
            dfPredictiveModelInput,
            dfPredictiveModelOutput,
            test_size=gc_dec_TEST_RATIO,
            shuffle=False)
    
        dfInputTrain, dfInputValidation, dfOutputTrain, dfOutputValidation = train_test_split(
            dfInputTrainValidation,
            dfOutputTrainValidation,
            test_size=gc_dec_VALIDATION_RATIO * gc_dec_TEST_RATIO,
            shuffle=False)
    
    
        oScalerInput = StandardScaler()
        oScalerOutput = MinMaxScaler()
        
        aScaledInputTrain = oScalerInput.fit_transform(dfInputTrain)
        aScaledOutputTrain = oScalerOutput.fit_transform(dfOutputTrain)
    
        oScalerInput.partial_fit(dfInputValidation)
        oScalerOutput.partial_fit(dfOutputValidation)
        aScaledInputValidation = oScalerInput.transform(dfInputValidation)
        aScaledOutputValidation = oScalerOutput.transform(dfOutputValidation)
    
        oScalerInput.partial_fit(dfInputTest)
        oScalerOutput.partial_fit(dfOutputTest)

        aScaledOutputTest = oScalerOutput.transform(dfOutputTest)
    
        sScalersDirectory = os.path.join(sSymbol, "__scalers__")
        
        sScalerFilePathInput =os.path.join(sScalersDirectory, "input.sav")
        sScalerFilePathOutput = os.path.join(sScalersDirectory, "target.sav")
        
        os.makedirs(os.path.dirname(sScalerFilePathInput), exist_ok=True)
        os.makedirs(os.path.dirname(sScalerFilePathOutput), exist_ok=True)
    
        pickle.dump(oScalerInput, open(sScalerFilePathInput, 'wb'))
        pickle.dump(oScalerOutput, open(sScalerFilePathOutput, 'wb'))
        
        
        if bIsClassification == False:
            oPredictiveModel = Long_Short_Term_Memory(
                sSymbol, 
                len(gc_a_INPUT_FEATURES), 
                len(aOutputFeatures), 
                len(g_aBackwardTimeSteps), 
                len(g_aForwardTimeSteps)
                )
        else:
            oPredictiveModel = Neural_Attention_Mechanism(sSymbol, 
                len(gc_a_INPUT_FEATURES), 
                len(aOutputFeatures), 
                len(g_aBackwardTimeSteps), 
                len(g_aForwardTimeSteps)
                )
        
        
        
        oPredictiveModel.Train(aScaledInputTrain, aScaledOutputTrain,aScaledInputValidation, aScaledOutputValidation)
    
    

        dfPrediction, dfScaledPrediction = dfPredict(sSymbol, oPredictiveModel, dfInputTest, dfOutputTest.index, dfOutputTest.columns)
    
        dfUnscaledMetrics = dfUnscaledMetrics.append(
            pd.DataFrame(
                dfGetMetrics(dfOutputTest, dfPrediction).values,
                index=pd.MultiIndex.from_product(
                    [[sSymbol], gc_a_METRICS], names=["symbol", "metrics"]),
                columns=dfOutputTest.columns
            )
        )
    
        dfScaledOutputTest = pd.DataFrame(aScaledOutputTest, columns=dfOutputTest.columns, index=dfOutputTest.index)
    
        dfScaledMetrics = dfScaledMetrics.append(
            pd.DataFrame(
                dfGetMetrics(dfScaledOutputTest, dfScaledPrediction).values,
                index=pd.MultiIndex.from_product(
                    [[sSymbol], gc_a_METRICS], names=["symbol", "metrics"]),
                columns=dfOutputTest.columns
            )
        )
        
        
    
        dfHistory = pd.DataFrame(oPredictiveModel.history.history)
        
        dfHistory[[
            "training loss", 
            "validation loss"]].plot()
        
        
        dfHistory[[
            'training precision',
            'validation precision'
            ]].plot()
        
        
        dfHistory[[
            'training accuracy',
            'validation accuracy'
            ]].plot()

        dfHistory[[
            'training recall',
            'validation recall'
            ]].plot()


        dfHistory[[
            'training f1-score',
            'validation f1-score'
            ]].plot()


        serModels = serModels.append(
            pd.Series(data=oPredictiveModel, index=[sSymbol]))


    dfPredictionErrorRmse = dfScaledMetrics.xs('rmse', axis = 0, level = 1, drop_level = True).astype(float)
    dfPredictionErrorRmse = dfPredictionErrorRmse.astype(float).mean(axis = 1, level = "time step")
    
    return dfSimulationModelInput, dfSimulationModelOutput, serModels, dfPredictionErrorRmse, dfUnscaledMetrics, dfScaledMetrics



def dfSimulate(dfSimulationModelInput, dfSimulationModelOutput, serModels, dfPredictionErrorRmse):
    dfSimulationModelInput = dfSimulationModelInput[::gc_i_FORWARD_TIME_WINDOW]
    dfSimulationModelOutput = dfSimulationModelOutput[::gc_i_FORWARD_TIME_WINDOW]
    
    dfExpectedProfit = pd.DataFrame()
    dfActualProfit = pd.DataFrame()
    dfBalance = pd.DataFrame()
    
    decBalance = gc_dec_INITIAL_BALANCE
    
    for dtSimulationModelTimeStamp, _ in dfSimulationModelInput.iterrows():
    
        dfExpectedPricesClose = pd.DataFrame()
        dfExpectedPricesOpen = pd.DataFrame()
        dfExpectedSpread = pd.DataFrame()
    
        dfActualPricesClose = pd.DataFrame()
        dfActualPricesOpen = pd.DataFrame()
        dfActualSpread = pd.DataFrame()
    
        dfBalance = dfBalance.append(pd.DataFrame([decBalance], index = [dtSimulationModelTimeStamp]))
    
    
        if decBalance != 0:
    
            for sSymbol in gc_a_SYMBOLS:
                serSimulationModelInput = dfSimulationModelInput[sSymbol][dtSimulationModelTimeStamp:dtSimulationModelTimeStamp]
                serSimulationModelOutput = dfSimulationModelOutput[sSymbol][dtSimulationModelTimeStamp:dtSimulationModelTimeStamp]
    
                oPredictiveModel = serModels[sSymbol]
                dfSimulationPrediction, _ = dfPredict(sSymbol,oPredictiveModel, serSimulationModelInput,serSimulationModelOutput.index, serSimulationModelOutput.columns)
    
    
                dfExpectedPricesClose = dfExpectedPricesClose.append(
                    pd.DataFrame(
                        dfSimulationPrediction.xs('close', axis = 1, level = 1, drop_level = True).values,
                        index=[sSymbol],
                        columns = g_aForwardTimeSteps)
                    )
    
    
                dfExpectedPricesOpen = dfExpectedPricesOpen.append(
                    pd.DataFrame(
                        dfSimulationPrediction.xs('open', axis = 1, level = 1, drop_level = True).values,
                        index=[sSymbol],
                        columns = g_aForwardTimeSteps)
                    )
    
    
                dfExpectedSpread = dfExpectedSpread.append(
                    pd.DataFrame(
                        dfSimulationPrediction.xs('spread', axis = 1, level = 1, drop_level = True).values,
                        index=[sSymbol],
                        columns = g_aForwardTimeSteps)
                    )
    
    
                dfActualPricesClose = dfActualPricesClose.append(pd.DataFrame(
                    dfSimulationModelOutput[sSymbol].xs('close', axis = 1, level = 1, drop_level = True)[dtSimulationModelTimeStamp:dtSimulationModelTimeStamp].values,
                    index=[sSymbol],
                    columns = g_aForwardTimeSteps)
                    )
    
                dfActualPricesOpen = dfActualPricesOpen.append(pd.DataFrame(
                    dfSimulationModelOutput[sSymbol].xs('open', axis = 1, level = 1, drop_level = True)[dtSimulationModelTimeStamp:dtSimulationModelTimeStamp].values,
                    index=[sSymbol],
                    columns = g_aForwardTimeSteps)
                    )
    
    
                dfActualSpread = dfActualSpread.append(pd.DataFrame(
                    dfSimulationModelOutput[sSymbol].xs('spread', axis = 1, level = 1, drop_level = True)[dtSimulationModelTimeStamp:dtSimulationModelTimeStamp].values,
                    index=[sSymbol],
                    columns = g_aForwardTimeSteps)
                    )
    
    
            oPortfolioOptimizerExpected = PortfolioManagement(
                dfExpectedPricesClose ,
                dfExpectedPricesOpen,
                dfActualSpread,
                pd.DataFrame(gc_a_SYMBOLS) ,
                pd.DataFrame(g_aForwardTimeSteps, columns = ["TIME_STEP"]),
                decBalance,
                gc_dec_MAX_RISK_RMSE,
                dfPredictionErrorRmse)
    
    
            oPortfolioOptimizerActual = PortfolioManagement(
                dfActualPricesClose ,
                dfActualPricesOpen,
                dfActualSpread,
                pd.DataFrame(gc_a_SYMBOLS) ,
                pd.DataFrame(g_aForwardTimeSteps, columns = ["TIME_STEP"]),
                decBalance,
                gc_dec_MAX_RISK_RMSE,
                dfPredictionErrorRmse)
    
    
            aOptimumAmounts, aOptiumumPositions = oPortfolioOptimizerExpected.Main()
            decExpectedProfit = oPortfolioOptimizerExpected.decEvaluateFitness(aOptimumAmounts, aOptiumumPositions)
            decActualProfit = oPortfolioOptimizerActual.decEvaluateFitness(aOptimumAmounts, aOptiumumPositions)
    
    
            decBalance = decBalance +  decActualProfit
    
            dfExpectedProfit = dfExpectedProfit.append(pd.DataFrame([decExpectedProfit], index = [dtSimulationModelTimeStamp]))
            dfActualProfit = dfActualProfit.append(pd.DataFrame([decActualProfit], index = [dtSimulationModelTimeStamp] ))




dfTrain(True)
