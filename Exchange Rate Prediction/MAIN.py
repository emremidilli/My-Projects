import Predictive_Model
import Calculate_Metrics
import Compile_Responses

from Design_of_Experiments import Central_Composite_Design,Full_Factorial_Design,Response_Surface_Method, Steepest_Descent_Process

import pandas as pd 

import Simulate

aExchangeRates  = pd.read_csv('Data/cryptocurrencies.csv')['Symbol'].values
# for sOutputSymbol in aExchangeRates:

#     # sModelType = 'MLP'
    
#     for sModelType in ['LSTM', 'Conv-EncDec', 'Luongs-Att']:
        
    
#     # dfFullFactorialDesign = Full_Factorial_Design.__init__(sOutputSymbol,sModelType)
    
#     # for iTrialId, _ in dfFullFactorialDesign.iterrows():
#     #     Predictive_Model.__init__(sOutputSymbol, sModelType, 'Full Factorial Design', iTrialId)
#     #     Calculate_Metrics.__init__(sOutputSymbol, sModelType, 'Full Factorial Design', iTrialId)
    
#     # Compile_Responses.__init__(sOutputSymbol, sModelType, 'Full Factorial Design')
    
#     # Steepest_Descent_Process.__init__(sOutputSymbol, sModelType)
    
#     # dfCentralCompositeDesign = Central_Composite_Design.__init__(sOutputSymbol, sModelType)
    
#     # for iTrialId, _ in dfCentralCompositeDesign.iterrows():
#     #     Predictive_Model.__init__(sOutputSymbol, sModelType, 'Central Composite Design', iTrialId)
#     #     Calculate_Metrics.__init__(sOutputSymbol, sModelType, 'Central Composite Design', iTrialId)
    
#     # Compile_Responses.__init__(sOutputSymbol, sModelType, 'Central Composite Design')
    
#         dfOptimumDesign = Response_Surface_Method.__init__(sOutputSymbol, sModelType)
        
#         Predictive_Model.__init__(sOutputSymbol, sModelType, 'Optimum Design', 0)
#         Calculate_Metrics.__init__(sOutputSymbol, sModelType, 'Optimum Design', 0)
#         Compile_Responses.__init__(sOutputSymbol, sModelType, 'Optimum Design')
    


dfBestModels = pd.DataFrame(
    data ={
        'Exchange Rate':aExchangeRates ,
        'Model Type':['MLP', 'LSTM', 'MLP', 'LSTM', 'LSTM']
    }
).set_index('Exchange Rate')


dfOhlc = pd.read_csv('Data\dfOhlc.csv')
dfOhlc['timestamp'] = pd.DatetimeIndex(dfOhlc['timestamp'])
dfOhlc.set_index('timestamp', inplace=True)


ixToSimulate = dfOhlc.iloc[-15:].index
Simulate.__init__(dfBestModels, ixToSimulate)