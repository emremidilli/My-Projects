import pandas as pd
import numpy as np
from doepy import build
from sklearn.utils import shuffle
import os


def __init__(sOutputSymbol, sModelType):
    sOutputSymbol = sOutputSymbol
    sModelType = sModelType
    
    c_iNrOfReplicate = 4
    dicFactors = {
            'Batch Size':[32, 256],
            'Number of Hidden Neurons':[50, 250]
        }
    
    
    sFolderPath = 'Data/'+ sOutputSymbol +'//'+ sModelType + '//'
    
    dfDesign = pd.DataFrame()
    for i in range(1, c_iNrOfReplicate+1):
        dfDesign = dfDesign.append(build.full_fact(dicFactors).astype(np.int))
    
    dfDesign = shuffle(dfDesign)
    dfDesign = dfDesign.reset_index().drop(['index'], axis = 1)
    
    sDesignFolder = sFolderPath + '\\Full Factorial Design\\'
    os.makedirs(sDesignFolder, exist_ok = True)
    dfDesign.to_csv(sDesignFolder + 'Design.csv', index=True, index_label='Run ID')
    
    return dfDesign