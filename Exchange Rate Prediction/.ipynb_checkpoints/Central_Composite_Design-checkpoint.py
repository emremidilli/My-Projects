import pandas as pd
import numpy as np
from doepy import build
import os


def __init__(sOutputSymbol,sModelType ):    
    # CONFIGURATION
    sFolderPath = 'Data/'+ sOutputSymbol +'//'+ sModelType + '//'
    fDelta = 0.1
    
    # FIND CURVATURE
    dfSteepestDescentExperiments = pd.read_csv( sFolderPath + '\Steepest Descent\Experiments.csv', index_col = 'Run ID')
    dfFirstCurvature = dfSteepestDescentExperiments[dfSteepestDescentExperiments['Response'].diff(-1) > 0].head(1)
    
    iBatchSizeCenter = dfFirstCurvature.loc[:, 'Batch Size'].iloc[0]
    iBatchSizeLower  = int(iBatchSizeCenter - (iBatchSizeCenter * fDelta))
    iBatchSizeUpper  = int(iBatchSizeCenter + (iBatchSizeCenter * fDelta))
    
    iNumberOfHiddenNeuronsCenter = dfFirstCurvature.loc[:, 'Number of Hidden Neurons'].iloc[0]
    iNumberOfHiddenNeuronsLower  = int(iNumberOfHiddenNeuronsCenter - (iNumberOfHiddenNeuronsCenter * fDelta))
    iNumberOfHiddenNeuronsUpper  = int(iNumberOfHiddenNeuronsCenter + (iNumberOfHiddenNeuronsCenter * fDelta))
    
    
    # CENTRAL COMPOSITE DESIGN
    
    dicFactors = {
        'Batch Size':[iBatchSizeLower,iBatchSizeUpper ],
        'Number of Hidden Neurons':[iNumberOfHiddenNeuronsLower,iNumberOfHiddenNeuronsUpper ]
    }

    dfDesign = build.central_composite(dicFactors, face='cci').astype(np.int)
    
    dfDesign = dfDesign.reset_index().drop(['index'], axis = 1)
    
    sDesignFolder = sFolderPath + '\\Central Composite Design\\'
    os.makedirs(sDesignFolder, exist_ok = True)
    dfDesign.to_csv(sDesignFolder + 'Design.csv', index=True, index_label='Run ID')
    
    return dfDesign