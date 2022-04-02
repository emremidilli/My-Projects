import Full_Factorial_Design
import Predictive_Model
import Calculate_Metrics
import Compile_Responses
import Steepest_Descent_Design
import Central_Composite_Design
import Response_Surface_Method


sOutputSymbol = 'BTCUSD'
sModelType = 'MLP'

# dfFullFactorialDesign = Full_Factorial_Design.__init__(sOutputSymbol,sModelType)

# for iTrialId, _ in dfFullFactorialDesign.iterrows():
#     Predictive_Model.__init__(sOutputSymbol, sModelType, 'Full Factorial Design', iTrialId)
#     Calculate_Metrics.__init__(sOutputSymbol, sModelType, 'Full Factorial Design', iTrialId)

# Compile_Responses.__init__(sOutputSymbol, sModelType, 'Full Factorial Design')

dfSteepestDescentDesign = Steepest_Descent_Design.__init__(sOutputSymbol, sModelType)

for iTrialId, _ in dfSteepestDescentDesign.iterrows():
    Predictive_Model.__init__(sOutputSymbol, sModelType, 'Steepest Descent', iTrialId)
    Calculate_Metrics.__init__(sOutputSymbol, sModelType, 'Steepest Descent', iTrialId)

Compile_Responses.__init__(sOutputSymbol, sModelType, 'Steepest Descent')

# dfCentralCompositeDesign = Central_Composite_Design.__init__(sOutputSymbol, sModelType)

# dfCentralCompositeDesign = Central_Composite_Design.__init__(sOutputSymbol, sModelType)

# for iTrialId, _ in dfCentralCompositeDesign.iterrows():
#     Predictive_Model.__init__(sOutputSymbol, sModelType, 'Central Composite Design', iTrialId)
#     Calculate_Metrics.__init__(sOutputSymbol, sModelType, 'Central Composite Design', iTrialId)

# Compile_Responses.__init__(sOutputSymbol, sModelType, 'Central Composite Design')

# Response_Surface_Method.__init__(sOutputSymbol, sModelType)