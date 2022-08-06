import pandas as pd

import numpy as np

import os



def __init__(sOutputSymbol, sModelType, sDesignType):
    
    # CONFIGURATIONS
    sFolderPath = 'Data/'+ sOutputSymbol +'//'+ sModelType + '//'+ sDesignType+ '//'
    dfDesign = pd.read_csv( sFolderPath + 'Design.csv', index_col = 'Run ID')
    
    
    # COMPILE [Response] VALUES
    dfExperiments = dfDesign.copy()
    dfExperiments['Response'] = np.nan
    
    for sTrialNr in next(os.walk(sFolderPath))[1]:
        if sTrialNr.isdigit() == True:
            dfPerformance = pd.read_csv(sFolderPath + '//' + sTrialNr + '//' +'dfPerformance.csv', index_col=0)
            dfExperiments.loc[int(sTrialNr), 'Response'] = dfPerformance.loc['custom_metric', 'value']
            
    dfExperiments.to_csv(sFolderPath + 'Experiments.csv')