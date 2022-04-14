import Predictive_Model
import Calculate_Metrics
import Compile_Responses

from Design_of_Experiments import Central_Composite_Design,Full_Factorial_Design,Response_Surface_Method, Steepest_Descent_Process

import pandas as pd 

aOutputSymbols  = pd.read_csv('Data/cryptocurrencies.csv')['Symbol'].values
for sOutputSymbol in aOutputSymbols:

    sModelType = 'Luongs Attention'
    
    dfFullFactorialDesign = Full_Factorial_Design.__init__(sOutputSymbol,sModelType)
    
    for iTrialId, _ in dfFullFactorialDesign.iterrows():
        Predictive_Model.__init__(sOutputSymbol, sModelType, 'Full Factorial Design', iTrialId)
        Calculate_Metrics.__init__(sOutputSymbol, sModelType, 'Full Factorial Design', iTrialId)
    
    Compile_Responses.__init__(sOutputSymbol, sModelType, 'Full Factorial Design')
    
    Steepest_Descent_Process.__init__(sOutputSymbol, sModelType)
    
    dfCentralCompositeDesign = Central_Composite_Design.__init__(sOutputSymbol, sModelType)
    
    for iTrialId, _ in dfCentralCompositeDesign.iterrows():
        Predictive_Model.__init__(sOutputSymbol, sModelType, 'Central Composite Design', iTrialId)
        Calculate_Metrics.__init__(sOutputSymbol, sModelType, 'Central Composite Design', iTrialId)
    
    Compile_Responses.__init__(sOutputSymbol, sModelType, 'Central Composite Design')
    
    Response_Surface_Method.__init__(sOutputSymbol, sModelType)
    
    
    