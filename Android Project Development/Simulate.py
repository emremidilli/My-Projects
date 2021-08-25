import Train
import Preprocess
import Optimize_Portfolio
from Connect_to_Database import execute_sql
import Neural_Attention_Mechanism
import Multi_Layer_Perceptron
import pickle
import pandas as pd


gc_dec_INITIAL_BALANCE = 150
gc_dec_MAX_RISK_MAPE = 100
gc_dt_FROM_DATE = '2021-03-29 00:00:00.000'
gc_dt_TO_DATE = '2021-04-01 00:00:00.000'
gc_i_TIME_STEPS_TO_SKIP = 4
        

    
def dfPredict(sModelId, dtTimeStamp):
    dfInput, dfTimeStepsInput = Preprocess.dfGetFeatureValues(sModelId, "1",dtTimeStamp, dtTimeStamp)
    dfOutput, dfTimeStepsOutput = Preprocess.dfGetFeatureValues(sModelId, "2",dtTimeStamp, dtTimeStamp)
    
    iFeatureSizeX, iWindowLengthX = Preprocess.dfGetDimensionSize(dfTimeStepsInput)
    iFeatureSizeY ,iWindowLengthY = Preprocess.dfGetDimensionSize(dfTimeStepsOutput)
                
    dfInputIndex = dfInput.index
    
    if dfInput.shape[0] > 0:
                        
        sScalerFilePathInput = Train.gc_s_SCALERS_PATH +  str(sModelId) + ' input.sav'
        sScalerFilePathOutput = Train.gc_s_SCALERS_PATH +  str(sModelId) + ' target.sav'
        
        oScalerInput = pickle.load(open(sScalerFilePathInput, 'rb'))
        oScalerOutput = pickle.load(open(sScalerFilePathOutput, 'rb'))
        
        oScalerInput.partial_fit(dfInput)
        
        dfInputScaled = oScalerInput.transform(dfInput)
        
        # oNeuralAttentionModel = Neural_Attention_Mechanism.Neural_Attention_Mechanism(sModelId, iFeatureSizeX, iFeatureSizeY, iWindowLengthX, iWindowLengthY)
        # aPrediction = oNeuralAttentionModel.predict(dfInputScaled)
        
        oMultiLayerPerceptron = Multi_Layer_Perceptron.Multi_Layer_Perceptron(sModelId, iFeatureSizeX, iFeatureSizeY, iWindowLengthX, iWindowLengthY)
        aPrediction = oMultiLayerPerceptron.aPredict(dfInputScaled)
        
        
        aPrediction = oScalerOutput.inverse_transform(aPrediction)
        dfPrediction = pd.DataFrame(aPrediction)
        dfPrediction["INDEX"] = dfInputIndex
        dfPrediction = dfPrediction.set_index("INDEX")
        dfPrediction.columns = dfOutput.columns
        
        pickle.dump(oScalerInput, open(sScalerFilePathInput, 'wb'))
        pickle.dump(oScalerOutput, open(sScalerFilePathOutput, 'wb'))
        
        
        return dfPrediction
    
    
    return None



def dfGetModelTimeSteps(sModelId, sFeatureTypeId):

    sSql = "exec SP_GET_TIME_STEPS "+ sModelId +", " + sFeatureTypeId
    dfModelTimeSteps = execute_sql(sSql, "")
    
    return dfModelTimeSteps

def decGetExpectedSpreadRate(sModelId, dtTimeStamp):
    
    sSqlGetModelFeatures = "SELECT * FROM FN_GET_MODEL_FEATURES("+ sModelId +", 1)"
    dfModelFeatures = execute_sql(sSqlGetModelFeatures, "")
    
    
    iFeatureIdClose = dfModelFeatures[dfModelFeatures['FEATURE_SHORT_DESCRIPTION'] == '<CLOSE>']["FEATURE_ID"].values[0]
    iFeatureIdSpread = dfModelFeatures[dfModelFeatures['FEATURE_SHORT_DESCRIPTION'] == '<SPREAD>']["FEATURE_ID"].values[0]
    
    
    sSqlGetFeatureValuesClose = "SELECT * FROM VW_FEATURE_VALUES WHERE TIME_STAMP = '"+ dtTimeStamp +"' AND FEATURE_ID = " + str(iFeatureIdClose)
    dfModelFeatureValuesClose = execute_sql(sSqlGetFeatureValuesClose, "")


    iRowNumberClose = dfModelFeatureValuesClose["ROW_NUMBER"].values[0]
    iRowNumberClose = iRowNumberClose - 1
    
    
    sSqlGetFeatureValuesSpread = "SELECT * FROM VW_FEATURE_VALUES WHERE TIME_STAMP = '"+ dtTimeStamp +"' AND FEATURE_ID = " + str(iFeatureIdSpread)
    dfModelFeatureValuesSpread = execute_sql(sSqlGetFeatureValuesSpread, "")
    iRowNumberSpread = dfModelFeatureValuesSpread["ROW_NUMBER"].values[0]
    iRowNumberSpread = iRowNumberSpread - 1    
    
    
    sSqlGetFeatureValuesClosePrev = "SELECT * FROM VW_FEATURE_VALUES WHERE ROW_NUMBER = " + str(iRowNumberClose)  +" AND FEATURE_ID = " + str(iFeatureIdClose)
    dfFeatureValuesClosePrev = execute_sql(sSqlGetFeatureValuesClosePrev, "")
    decCloseValue = dfFeatureValuesClosePrev["VALUE"].values[0]


    sSqlGetFeatureValuesSpreadPrev = "SELECT * FROM VW_FEATURE_VALUES WHERE ROW_NUMBER = " + str(iRowNumberSpread)  +" AND FEATURE_ID = " + str(iFeatureIdSpread)
    dfFeatureValuesSpreadPrev = execute_sql(sSqlGetFeatureValuesSpreadPrev, "")
    decSpreadValue = dfFeatureValuesSpreadPrev["VALUE"].values[0]
    
    decSpreadRatio = decSpreadValue/decCloseValue
    
    
    return decSpreadRatio
    
def dfGetPredictionErrorMape(sModelId, sModelDescription):
    sSql = "SELECT * FROM FN_GET_AVG_MODEL_STATISTICS("+ sModelId +", 5)"
    dfPredictionErrorMape = execute_sql(sSql, "")
    dfPredictionErrorMape = dfPredictionErrorMape.set_index("TIME_STEP").transpose()
    dfPredictionErrorMape.index =  [sModelDescription]
    
    
    return dfPredictionErrorMape

def dfConvertdPricesDataFrames(sModelDescription , dfPrediction, dfModelTimeSteps, dfForwardTimeSteps, sFeatureName):    
    dfTimeSteps = dfModelTimeSteps[dfModelTimeSteps['FEATURE_SHORT_DESCRIPTION'] == sFeatureName]

    dfExpectedPrices = dfPrediction[[*dfTimeSteps["ID"]]]

    dfExpectedPrices.columns = dfForwardTimeSteps["TIME_STEP"]

    dfExpectedPrices.index =  [sModelDescription]

    return dfExpectedPrices


def dfGetActualSpread(sModelId,sModelDescription,  dtTimeStamp, dfForwardTimeSteps):        
    sSqlGetModelFeatures = "SELECT * FROM FN_GET_MODEL_FEATURES("+ sModelId +", 1)"
    dfModelFeatures = execute_sql(sSqlGetModelFeatures, "")
    
    
    iFeatureIdSpread = dfModelFeatures[dfModelFeatures['FEATURE_SHORT_DESCRIPTION'] == '<SPREAD>']["FEATURE_ID"].values[0]
    sSqlGetFeatureValuesSpread = "SELECT * FROM VW_FEATURE_VALUES WHERE TIME_STAMP = '"+ dtTimeStamp +"' AND FEATURE_ID = " + str(iFeatureIdSpread)
    dfModelFeatureValuesSpread = execute_sql(sSqlGetFeatureValuesSpread, "")
    iRowNumberSpread = dfModelFeatureValuesSpread["ROW_NUMBER"].values[0]
    
    
    sSqlGetFeatureValuesSpread = "SELECT * FROM VW_FEATURE_VALUES WHERE ROW_NUMBER BETWEEN " + str(iRowNumberSpread + int(dfForwardTimeSteps.min())) + " AND " + str(iRowNumberSpread + int(dfForwardTimeSteps.max())) + " AND FEATURE_ID = " + str(iFeatureIdSpread)
    dfFeatureValuesSpread = execute_sql(sSqlGetFeatureValuesSpread, "")
    dfFeatureValuesSpread = pd.DataFrame(dfFeatureValuesSpread["VALUE"])
    
    dfFeatureValuesSpread = dfFeatureValuesSpread.transpose()
    
    dfFeatureValuesSpread.index = [sModelDescription]
    
    return dfFeatureValuesSpread



def dfGetTimeStamps():
    sSqlSelectTimeStamps = "SELECT DISTINCT TIME_STAMP FROM TBL_FEATURE_VALUES WHERE TIME_STAMP BETWEEN '"+ gc_dt_FROM_DATE +"' AND '"+ gc_dt_TO_DATE +"' ORDER BY TIME_STAMP ASC "
    dfTimeStamps = execute_sql(sSqlSelectTimeStamps, "")
    
    return dfTimeStamps


def main():
    
    dfTimeStamps = dfGetTimeStamps()
    
    decBalance = gc_dec_INITIAL_BALANCE
    

    dfModels = Preprocess.dfGetModels()
    dfFinancialProducts = pd.DataFrame(dfModels["SHORT_DESCRIPTION"])
    
    dfExpectedProfit = pd.DataFrame()
    dfActualProfit = pd.DataFrame()
    dfBalance = pd.DataFrame()
    dfSimulatedTimeStamps = pd.DataFrame()

    
    for i_iIndex, i_aRow in dfTimeStamps.iterrows():
        
        if i_iIndex % gc_i_TIME_STEPS_TO_SKIP == 0:
            
            dtTimeStamp = str(i_aRow["TIME_STAMP"])
            
            print(dtTimeStamp)

            dfExpectedPricesClose = pd.DataFrame()
            dfExpectedPricesOpen = pd.DataFrame()
            dfExpectedSpread = pd.DataFrame()
            dfPredictionErrorMape = pd.DataFrame()
            
            dfActualPricesClose = pd.DataFrame()
            dfActualPricesOpen = pd.DataFrame()
            dfActualSpread = pd.DataFrame()
            
            
            for j_iIndex, j_aRow in dfModels.iterrows():
                sModelId = str(j_aRow["ID"])
                sModelDescription = j_aRow["SHORT_DESCRIPTION"]
            
                dfModelPredictionErrorMape = dfGetPredictionErrorMape(sModelId, sModelDescription)
                
                decSpreadRatio = decGetExpectedSpreadRate(sModelId, dtTimeStamp)
                
                dfModelTimeSteps = dfGetModelTimeSteps(sModelId, "2")
                
                dfForwardTimeSteps = pd.DataFrame(dfModelTimeSteps["TIME_STEP"].unique(), columns = ["TIME_STEP"])
                
                dfPrediction = dfPredict(sModelId , dtTimeStamp)
                
                dfExpectedModelPricesClose = dfConvertdPricesDataFrames(sModelDescription, dfPrediction, dfModelTimeSteps, dfForwardTimeSteps, '<CLOSE>')
                dfExpectedModelPricesOpen = dfConvertdPricesDataFrames(sModelDescription, dfPrediction, dfModelTimeSteps, dfForwardTimeSteps, '<OPEN>')
                
                dfExpectedModelSpread = dfExpectedModelPricesClose * decSpreadRatio
                
                dfExpectedPricesOpen = dfExpectedPricesOpen.append(dfExpectedModelPricesOpen)
                dfExpectedPricesClose = dfExpectedPricesClose.append(dfExpectedModelPricesClose)
                dfExpectedSpread = dfExpectedSpread.append(dfExpectedModelSpread)
                dfPredictionErrorMape = dfPredictionErrorMape.append(dfModelPredictionErrorMape)
                
                
                dfActual, _ = Preprocess.dfGetFeatureValues(sModelId, "2",dtTimeStamp, dtTimeStamp)
                
                dfActualModelPricesClose = dfConvertdPricesDataFrames(sModelDescription, dfActual, dfModelTimeSteps, dfForwardTimeSteps, '<CLOSE>')
                dfActualModelPricesOpen = dfConvertdPricesDataFrames(sModelDescription, dfActual, dfModelTimeSteps, dfForwardTimeSteps, '<OPEN>')
                dfActualModelSpread = dfGetActualSpread(sModelId,sModelDescription,  dtTimeStamp, dfForwardTimeSteps)
                
                dfActualPricesClose = dfActualPricesClose.append(dfActualModelPricesClose)
                dfActualPricesOpen = dfActualPricesOpen.append(dfActualModelPricesOpen)
                dfActualSpread = dfActualSpread.append(dfActualModelSpread)
            
 
            oPortfolioManagerExpected = Optimize_Portfolio.PortfolioManagement(dfExpectedPricesClose ,dfExpectedPricesOpen, dfExpectedSpread, dfFinancialProducts , dfForwardTimeSteps, decBalance, gc_dec_MAX_RISK_MAPE, dfPredictionErrorMape)
            aOptimumAmounts, aOptiumumPositions = oPortfolioManagerExpected.Main()
            
            decExpectedProfit = oPortfolioManagerExpected.decEvaluateFitness(aOptimumAmounts, aOptiumumPositions)
            
            
            oPortfolioManagerActual = Optimize_Portfolio.PortfolioManagement(dfActualPricesClose ,dfActualPricesOpen, dfActualSpread, dfFinancialProducts , dfForwardTimeSteps, decBalance, gc_dec_MAX_RISK_MAPE, dfPredictionErrorMape)
            decActualProfit = oPortfolioManagerActual.decEvaluateFitness(aOptimumAmounts, aOptiumumPositions)
            
            
            decBalance = decBalance +  decActualProfit
    
            dfExpectedProfit = dfExpectedProfit.append([decExpectedProfit])
            dfActualProfit = dfActualProfit.append([decActualProfit])
            dfBalance = dfBalance.append([decBalance])
            
            dfSimulatedTimeStamps = dfSimulatedTimeStamps.append([dtTimeStamp])
            
            
    
    # dfExpectedProfit.index = dfSimulatedTimeStamps
    dfActualProfit.index = dfSimulatedTimeStamps
    dfBalance.index = dfSimulatedTimeStamps
            
            
    return dfExpectedProfit, dfActualProfit, dfBalance
    
    
    
        
dfExpectedProfit, dfActualProfit, dfBalance = main()

