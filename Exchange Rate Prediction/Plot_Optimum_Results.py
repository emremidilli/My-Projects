# LIBRARIES
import pandas as pd

import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error, max_error, r2_score

import seaborn as sns

import matplotlib.pyplot as plt

import os

import tensorflow as tf

from Calculate_Metrics import dfGetCombinationsOfReturns, fCalculateCustomMetric

def __init__(sOutputSymbol, sModelType):
    # CONFIGURATION
    
    sModelName = 'Data/'+ sOutputSymbol +'//'+ sModelType + '//Optimum Model//'
    
    dfOhlc = pd.read_csv('Data\dfOhlc.csv', index_col = 0)
    dfActual = pd.read_csv(sModelName+'\dfActual.csv',header=[0, 1], index_col=0)
    dfPrediction = pd.read_csv(sModelName+'\dfPrediction.csv',header=[0, 1], index_col=0)
    dfPerformance = pd.read_csv(sModelName+'\dfPerformance.csv', index_col=0)
    
    iForwardTimeWindow = dfActual.shape[1]
    
    
    # CONVERT [return] TO [close]
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
        dfClose = dfClose.iloc[::iForwardTimeWindow]
        return dfClose
    
    
    dfActualClose = dfGetClosePricesFromReturns(dfActual)
    dfPredictionClose = dfGetClosePricesFromReturns(dfPrediction)
    
    
    # PLOT RESULTS
    
    ## Single Time Step Comparison
    iComparisionTimeStep = 0
    
    dfTestComparision = pd.DataFrame(dfPredictionClose.iloc[:,iComparisionTimeStep])
    dfTestComparision = dfTestComparision.join(dfActualClose.iloc[:,iComparisionTimeStep], how = "inner", lsuffix="prediction")
    dfTestComparision.columns = ["Prediction", "Actual"]
    
    plt.figure(figsize = (20,10))
    oFig = sns.scatterplot(data = dfTestComparision, x = "Actual", y ="Prediction")
    oFig.get_figure().savefig(sModelName + '\closing price scatter.png')
    
    ## Multi Step Comparison
    aOutputFeatures = [sOutputSymbol +':close']
    
    
    iNrOfCols = 6
    iNrOfRows = int(((len(dfActualClose)/iNrOfCols)/iForwardTimeWindow) + 1)
    oFig, aAxises = plt.subplots(iNrOfRows, iNrOfCols, figsize=(30,80), sharex = True)
    oFig.tight_layout()
        
    i = 0
    for iSampleNr in range(0, len(dfActualClose), iForwardTimeWindow):
        iFrom = iSampleNr
        iTo = iFrom + iForwardTimeWindow
    
        if iTo >= len(dfActualClose):
            iTo = len(dfActualClose) 
        
        dfStepComparision = dfActualClose.iloc[iFrom:iTo].loc[:, (slice(None), slice(aOutputFeatures[0]))].loc[:, '0']
        
        dfStepComparision.columns = ["Actual"]
    
        dfStepComparision["Prediction"] = dfPredictionClose.iloc[iFrom].iloc[0:iTo-iFrom].loc[:, aOutputFeatures].values
        
    
        sTitleName = str(iFrom) + "---" + str(iTo) + "---" + str(round(r2_score(dfStepComparision["Actual"], dfStepComparision["Prediction"]),1))
    
        iSampleGraphRow =  int(i/iNrOfCols)
        iSampleGraphCol = int(i%iNrOfCols)
    
        dfStepComparision.reset_index(inplace = True)
        sns.lineplot(ax =aAxises[iSampleGraphRow,iSampleGraphCol] , data = dfStepComparision, legend = False,  marker = '^').set_title(sTitleName)
    
        i = i + 1
    
    oFig.legend(aAxises[0][0].lines, ['actual', 'prediction'], frameon=False, loc='lower center', ncol=2,  bbox_to_anchor=(0.5,-0.01))  
    oFig.get_figure().savefig(sModelName + '\closing price multi step.png')
