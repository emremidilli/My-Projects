import pandas as pd
import numpy as np
from doepy import build
import os

from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler

import itertools

import matplotlib.pyplot as plt
from matplotlib import cm


import Predictive_Model
import Calculate_Metrics
import Compile_Responses


class Full_Factorial_Design():

    def __init__(sOutputSymbol, sModelType):
        sOutputSymbol = sOutputSymbol
        sModelType = sModelType
        
        c_iNrOfReplicate = 4
        dicFactors = {
                'Batch Size':[60, 70],
                'Number of Hidden Neurons':[10, 14]
            }
        
        
        sFolderPath = 'Data/'+ sOutputSymbol +'//'+ sModelType + '//'
        
        dfDesign = pd.DataFrame()
        for i in range(1, c_iNrOfReplicate+1):
            dfDesign = dfDesign.append(build.full_fact(dicFactors).astype(int))
        
        dfDesign = shuffle(dfDesign)
        dfDesign = dfDesign.reset_index().drop(['index'], axis = 1)
        
        sDesignFolder = sFolderPath + '\\Full Factorial Design\\'
        os.makedirs(sDesignFolder, exist_ok = True)
        dfDesign.to_csv(sDesignFolder + 'Design.csv', index=True, index_label='Run ID')
        
        return dfDesign
    
    
class Steepest_Descent_Process():
    
    def __init__(sOutputSymbol,sModelType):
        # CONFIGURATION    
        sFolderPath = 'Data/'+ sOutputSymbol +'//'+ sModelType + '//'
        dfFullFactorialExperiments = pd.read_csv( sFolderPath + 'Full Factorial Design//Experiments.csv', index_col = 'Run ID')
        sDesignFolder = sFolderPath + '//Steepest Descent//'
        os.makedirs(sDesignFolder, exist_ok = True)  
        
        # STEEPEST DESCENT
        dfFactors = dfFullFactorialExperiments.iloc[:, :-1]
        dfResponse = dfFullFactorialExperiments['Response']
        
        ## Convert Uncoded Factors to Coded Factors
        oScaler = MinMaxScaler()
        aCodedFactors = oScaler.fit_transform(dfFactors, (-1, 1))
        dfCodedFactors = pd.DataFrame(
            data = aCodedFactors,
            columns = list(oScaler.feature_names_in_),
            index = dfFactors.index
        )
        
        ## Fit Linear Regression
        oLinearRegression  = LinearRegression()
        oLinearRegression.fit(dfCodedFactors, dfResponse)
        
        ## Save Regression Score
        fRegressionScore = oLinearRegression.score(dfCodedFactors, dfResponse)
        dicPerformance = {
            'Score': ['Regression Score','Coefficients', 'Feature Names'],
            'Value' : [fRegressionScore, 
                       list(oLinearRegression.coef_),
                      list(oLinearRegression.feature_names_in_)]
        }
        
        dfPerformance = pd.DataFrame(data = dicPerformance).set_index('Score')
        dfPerformance.to_csv(sDesignFolder +'dfPerformance.csv', index=True, index_label='Score')
                
        
        ## Perform New Experiments
        iIndexBaseFactor = 1
        aRatiosToDecrease = ((oLinearRegression.coef_)/(oLinearRegression.coef_[iIndexBaseFactor]))*np.sign(oLinearRegression.coef_)*-1 # second factor is used as base factor
        
        aCenterValues = np.array((dfFactors.max()+dfFactors.min())/2).astype(int)
        aStepSizes = np.array((dfFactors.max()-dfFactors.min())/2).astype(int)
        
        iIteration = 0
        while (True):
            
            aNewDesign = aCenterValues + (iIteration * aRatiosToDecrease * aStepSizes)
            aNewDesign = aNewDesign.astype(int)
        
            dfIteration = pd.DataFrame(data =np.reshape(aNewDesign, (1,-1)), columns = dfFullFactorialExperiments.iloc[:, :-1].columns, index = [iIteration])
            dfIteration.index.name = 'Run ID'
        
            if (iIteration == 0):
                dfNewDesign = dfIteration 
            else:
                dfNewDesign = pd.read_csv(sDesignFolder + 'Design.csv', index_col = 'Run ID')
                dfNewDesign = dfNewDesign.append(dfIteration)
        
            
            dfNewDesign.to_csv(sDesignFolder +'Design.csv', index=True, index_label='Run ID')
        
            Predictive_Model.__init__(sOutputSymbol, sModelType, 'Steepest Descent', iIteration)
            Calculate_Metrics.__init__(sOutputSymbol, sModelType, 'Steepest Descent', iIteration)
            Compile_Responses.__init__(sOutputSymbol, sModelType, 'Steepest Descent')
        
            dfSteepestDescentExperiments = pd.read_csv(sDesignFolder + 'Experiments.csv', index_col = 'Run ID')
        
            fCurrentResponse = dfSteepestDescentExperiments.loc[iIteration,  'Response']
        
            if (iIteration == 0):
                iIteration =iIteration + 1
                continue
        
            fPreviousResponse = dfSteepestDescentExperiments.loc[iIteration-1,  'Response']
        
            if fPreviousResponse < fCurrentResponse:
                break
        
        
            iIteration = iIteration + 1
        
 
class Central_Composite_Design():
    
    def __init__(sOutputSymbol,sModelType ):    
        # CONFIGURATION
        sFolderPath = 'Data/'+ sOutputSymbol +'//'+ sModelType + '//'
        
        # FIND STEP SIZES
        dfFullFactorialDesign = pd.read_csv( sFolderPath + 'Full Factorial Design//Design.csv', index_col = 'Run ID')
        aStepSizes = np.array((dfFullFactorialDesign .max()-dfFullFactorialDesign.min())/2).astype(int)
        
        
        # FIND CURVATURE
        dfSteepestDescentExperiments = pd.read_csv( sFolderPath + '\Steepest Descent\Experiments.csv', index_col = 'Run ID')
        dfFirstCurvature = dfSteepestDescentExperiments.iloc[-2, :-1]
        
        # CENTRAL COMPOSITE DESIGN
        dfUpperLevel = (dfFirstCurvature  + aStepSizes).to_frame().transpose()
        dfLowerLevel = (dfFirstCurvature  - aStepSizes).to_frame().transpose()
        dfNewFactors = dfLowerLevel.append(dfUpperLevel)
        dfNewFactors.reset_index(drop = True, inplace = True)
        
        dicFactors = {}
        
        for sFactor in dfNewFactors.columns:
            dicFactors[sFactor] = list(dfNewFactors[sFactor].values)
        
        dfDesign = build.central_composite(dicFactors, center=(4, 4), alpha = 'r' ,face='cci').astype(int)
        
        dfDesign = dfDesign.reset_index().drop(['index'], axis = 1)
        
        sDesignFolder = sFolderPath + '\\Central Composite Design\\'
        os.makedirs(sDesignFolder, exist_ok = True)
        dfDesign.to_csv(sDesignFolder + 'Design.csv', index=True, index_label='Run ID')
        
        return dfDesign
    
    
class Response_Surface_Method():
    def __init__(sOutputSymbol, sModelType):
        
        # CONFIGURATION
        sFolderPath = 'Data/'+ sOutputSymbol +'//'+ sModelType + '//'
        
        dfCentralCompositeDesign = pd.read_csv( sFolderPath + '\Central Composite Design\Design.csv', index_col = 'Run ID')
        dfCentralCompositeExperiments = pd.read_csv( sFolderPath + '\Central Composite Design\Experiments.csv', index_col = 'Run ID')

        # 2ND ORDER MODEL
        oPolynomicalFeatures = PolynomialFeatures(degree = 2, include_bias = False)
        dfFactors = pd.DataFrame(
            data = oPolynomicalFeatures.fit_transform(dfCentralCompositeDesign),
            columns  = oPolynomicalFeatures.get_feature_names_out(),
            index = dfCentralCompositeDesign.index
        )
        
        dfResponse = dfCentralCompositeExperiments['Response']
        
        oLinearRegression  = LinearRegression()
        oLinearRegression.fit(dfFactors, dfResponse)
        
        ## Coefficient of Determination
        fRegressionScore = oLinearRegression.score(dfFactors, dfResponse)
        
        
        ## Equation
        aRegressionCoeff = oLinearRegression.coef_
        fRegressionIntercept =  oLinearRegression.intercept_
        
        
        ## Grid Search Optimization
        aCombinations = np.array([])
        i = 0
        for sCol in dfCentralCompositeDesign.columns:
            if i == 0:
                aCombinations= np.arange(dfCentralCompositeDesign[sCol].min(), dfCentralCompositeDesign[sCol].max(), 1)
            else:
                aCombinations = itertools.product(aCombinations, np.arange(dfCentralCompositeDesign[sCol].min(), dfCentralCompositeDesign[sCol].max(), 1))
                
            i = i +1 
        
        aCombinations = list(aCombinations)
        dfCombinations = pd.DataFrame(aCombinations, 
                     columns = dfCentralCompositeDesign.columns)
        
        dfPolynomialCombinations = pd.DataFrame(data = oPolynomicalFeatures.transform(dfCombinations),
                                      columns = oPolynomicalFeatures.get_feature_names_out()
                                     )
        
        dfCombinations['Response'] =  oLinearRegression.predict(dfPolynomialCombinations)
        
        sDesignFolder = sFolderPath + '\\Response Surface Method\\'
        os.makedirs(sDesignFolder, exist_ok = True)
        dfCombinations.to_csv(sDesignFolder + 'Grid Search.csv', index=True, index_label='Run ID')
        
        
        ## Optimum Configuration
        dfOptimumConfig = dfCombinations[dfCombinations['Response'] == dfCombinations['Response'].min()].head(1)
        
        fig = plt.figure(figsize=(20,8))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.plot_trisurf(dfCombinations.iloc[:, 0], 
                        dfCombinations.iloc[:, 1], 
                        dfCombinations['Response'], 
                        linewidth=0, 
                        antialiased=False,
                        cmap=cm.hsv
                       )
        fig.get_figure().savefig(sFolderPath + '\Response Surface Method\surface plot.png')
        plt.show()
        
        ## Save Performance
        dicPerformance=  {'Performance' : ['Regression Score', 'Feature Names' ,'Intercept', 'Coefficients', 'Hyperparameter Names', 'Optimum Hyperparameters', 'Optimum Response' ] ,
            'Value' : [fRegressionScore, 
                       list(oPolynomicalFeatures.get_feature_names_out()),
                       fRegressionIntercept, 
                       list(aRegressionCoeff),
                       list(dfOptimumConfig.iloc[:,:-1].columns), 
                       list(dfOptimumConfig.iloc[0,:-1].values),  
                       dfOptimumConfig.iloc[0,-1] ]}
        
        dfPerformance = pd.DataFrame(dicPerformance).set_index('Performance')
        dfPerformance.to_csv(sDesignFolder + '/dfPerformance.csv', index=True, index_label='Performance')