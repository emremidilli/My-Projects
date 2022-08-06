import pandas as pd

import numpy as np

import os

from sklearn.linear_model import LinearRegression


def __init__(sOutputSymbol,sModelType):
    # CONFIGURATION    
    sFolderPath = 'Data/'+ sOutputSymbol +'//'+ sModelType + '//'
    dfFullFactorialExperiments = pd.read_csv( sFolderPath + 'Full Factorial Design//Experiments.csv', index_col = 'Run ID')
    
    
    iNumberOfIteration = 10
    fDelta = 0.1
    
    # STEEPEST DESCENT
    dfFactors = dfFullFactorialExperiments.iloc[:, :-1]
    dfResponse = dfFullFactorialExperiments['Response']
    
    ## FIT LINEAR REGRESSION
    oLinearRegression  = LinearRegression()
    oLinearRegression.fit(dfFactors, dfResponse)
    
    ## SET NEW EXPERIMENTS
    aRatiosToDecrease = ((oLinearRegression.coef_)/(oLinearRegression.coef_[0]))*-1 #first factor is used as base factor
    aRatiosToDecrease = aRatiosToDecrease[0] * fDelta
    
    aCenterValues = np.array((dfFactors.max()+dfFactors.min())/2)
    aIterations = range(0, iNumberOfIteration)
    aNewDesign = aCenterValues + (np.array(np.add(aIterations, 1)).reshape(-1, 1) * aRatiosToDecrease * aCenterValues)
    aNewDesign = aNewDesign.astype(np.int)
    
    dfNewDesign =pd.DataFrame(
        data = aNewDesign,
        columns = dfFactors.columns,
        index = aIterations
    )
    
    ixExperimetnsLessThan10 = dfNewDesign[(dfNewDesign < 10).any(axis= 1)].index
    dfNewDesign.loc[ixExperimetnsLessThan10]
    dfNewDesign.drop(ixExperimetnsLessThan10, axis = 0 , inplace = True)
    
    sDesignFolder = sFolderPath + '//Steepest Descent//'
    os.makedirs(sDesignFolder, exist_ok = True)
    dfNewDesign.to_csv(sDesignFolder +'Design.csv', index=True, index_label='Run ID')
    
    return dfNewDesign