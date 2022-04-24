import pandas as pd
import random
import numpy as np
from deap import base
from deap import creator
from deap import tools
import operator
import enum
from datetime import datetime



class PortfolioManagement():
        
    def __init__(self, dfExpectedPricesClose ,dfExpectedPricesOpen, dfExpectedSpread, dfFinancialProducts , dfForwardTimeSteps, decInitialBalance, decMaximumRiskRmse, dfPredictionErrorRmse):
        self.dfExpectedPricesClose= dfExpectedPricesClose
        self.dfExpectedPricesOpen= dfExpectedPricesOpen
        self.dfExpectedSpread= dfExpectedSpread
        self.dfForwardTimeSteps = dfForwardTimeSteps
        self.dfFinancialProducts = dfFinancialProducts
        self.decInitialBalance = decInitialBalance
        self.dfIndices = self.dfGetIndices()
        self.aPriceDiffs = self.aGetPriceDiffs(self.dfExpectedPricesClose, self.dfExpectedPricesOpen)
        self.decStepSizeRatio = 0.01
        self.aAverageRisks = self.aGetAverageRisks(dfPredictionErrorRmse)
        self.decMaximumRiskRmse = decMaximumRiskRmse
        
        

    def dfGetIndices(self):
        aI = []
        aJ = []
        aK = []
                      
        for iIndex, iRow in self.dfFinancialProducts.iterrows():
            for jIndex, jRow in self.dfForwardTimeSteps.iterrows():
                jTimeStep =jRow["TIME_STEP"]
                for kIndex, kRow in self.dfForwardTimeSteps.iterrows():
                    kTimeStep = kRow["TIME_STEP"]
                    if jTimeStep <= kTimeStep:
                        aI.append(iIndex)
                        aJ.append(jIndex)
                        aK.append(kIndex)

        dicIndices = {
                    "i":aI,
                    "j":aJ,
                    "k":aK,
                    }        
        
        
        dfIndices = pd.DataFrame(data=dicIndices)
        
        return dfIndices        


    def aGetPriceDiffs(self, dfPricesClose, dfPricesOpen):
        
        aPriceDiffs = []
        for iIndex, aRow in self.dfIndices.iterrows():
            i = aRow["i"]
            j = aRow["j"]
            k = aRow["k"]
            
            
            decPriceDiff = dfPricesClose.iloc[i][k] - dfPricesOpen.iloc[i][j]
            decPriceDiff = decPriceDiff/dfPricesOpen.iloc[i][j]
            
            aPriceDiffs.append(decPriceDiff)

        return aPriceDiffs
       
        
       
    def dfGetDecisionVariables(self, sVariableLetter = "x"):
        aTypes = []
        aLabels = []
        aLowerBounds = []
        aUpperBounds = []
        aLowerStepSizes = []
        aUpperStepSizes = []
        

        if sVariableLetter == "x":
            iType = variable_types.decimal
        else:
            iType = variable_types.integer   
            
        for iIndex, aRow in self.dfIndices.iterrows():
            i = aRow["i"]
            j = aRow["j"]
            k = aRow["k"]
            
            sLabel = str(sVariableLetter) + "_" + str(i) + "_" + str(j) + "_" + str(k)
            
            if sVariableLetter == "x":
                decUpperBound = self.decInitialBalance
                decLowerBound = 0
                
                decUpperStepSize = decUpperBound * self.decStepSizeRatio
                decLowerStepSize = 0
            else:
                decLowerBound = -1
                decUpperBound = 1
                
                decLowerStepSize = -1
                decUpperStepSize = 1
                
            aLabels.append(sLabel)
            aTypes.append(iType)
            aLowerBounds.append(decLowerBound)
            aUpperBounds.append(decUpperBound)
            
            aLowerStepSizes.append(decLowerStepSize)
            aUpperStepSizes.append(decUpperStepSize)
            

        dicDecisionVariables = {
                    "label":aLabels,
                    "variable_type": aTypes,
                    "lower_bound" : aLowerBounds,
                    "upper_bound" : aUpperBounds,
                    "step_size_lower" : aLowerStepSizes,
                    "step_size_upper" : aUpperStepSizes
                    }

        

        dfDecisonVariables = pd.DataFrame(data=dicDecisionVariables)       
        return dfDecisonVariables
    
    
    def dfGetObjectiveFunctions(self):
        dicObjectiveFunctions =  {
                        "label":["Fitness"]
                        }
    
        dfObjectiveFunctions = pd.DataFrame(data=dicObjectiveFunctions)
        
        return dfObjectiveFunctions
    
    
    
    def aGetAverageRisks(self, dfPredictionErrorRmse):
        
        aAverageRisks = []
        
        for iIndex, aRow in self.dfIndices.iterrows():

            i = aRow["i"]
            j = aRow["j"]
            k = aRow["k"]
            
            aIntervalErrors = dfPredictionErrorRmse.iloc[i][j:k+1]
            
            decAverageRisk = np.mean(aIntervalErrors)
            
            aAverageRisks.append(decAverageRisk)
        

        return aAverageRisks
            
            
            
        

    def aCalculateReturns(self,aAmountsX, aPositionsY):
        aReturns = []
        
        for iIndex, aRow in self.dfIndices.iterrows():
            i = aRow["i"]
            j = aRow["j"]
            
            decSpread = self.dfExpectedSpread.iloc[i,j]
            decSpreadRatio = decSpread/self.dfExpectedPricesOpen.iloc[i,j]
            decReturn = self.aPriceDiffs[iIndex] * aAmountsX[iIndex] * aPositionsY[iIndex]
            decLossDueToSpread = abs(aPositionsY[iIndex])*aAmountsX[iIndex]*decSpreadRatio

            decReturn = decReturn - decLossDueToSpread

            aReturns.append(decReturn)
            
        return aReturns


    def aCalculateBalances(self, aReturns, aAmounts):
        
        aBalances = np.zeros(self.dfForwardTimeSteps.shape[0])
        
        
        for jIndex, jRow in self.dfForwardTimeSteps.iterrows():
            
            aIndicesOfClosingOrder = self.dfIndices[self.dfIndices['k'] == jIndex].index
            aIndicesOfOpeningOrder = self.dfIndices[self.dfIndices['j'] == jIndex].index
            
            aProfitsToBeReturned =  np.array(aReturns)[aIndicesOfClosingOrder]
            aProfitsToBeReturned = list(aProfitsToBeReturned)
            
            aCapitalsToBeReturned = np.array(aAmounts)[aIndicesOfClosingOrder]
            aCapitalsToBeReturned = list(aCapitalsToBeReturned)
            
            aCapitalsToBeInvested = np.array(aAmounts)[aIndicesOfOpeningOrder]
            aCapitalsToBeInvested = list(aCapitalsToBeInvested)            
            
            
            decProfitToBeReturned = sum(aProfitsToBeReturned)
            decCapitalToBeReturned = sum(aCapitalsToBeReturned)
            decCapitalToBeInvested = sum(aCapitalsToBeInvested)
            
            
            if jIndex == 0:
                 aBalances[jIndex] =  self.decInitialBalance +  decCapitalToBeReturned + decProfitToBeReturned - decCapitalToBeInvested
            else:
                aBalances[jIndex] =  aBalances[jIndex - 1] +  decCapitalToBeReturned + decProfitToBeReturned - decCapitalToBeInvested
            
        aBalances = list(aBalances)
        return aBalances
    
    

    def EnsureNonNegativeReturns(self, aIndividualX, aIndividualY):
        aAmountsX =  list(aIndividualX)
        aPositionsY = list(aIndividualY)
        
        aReturns = self.aCalculateReturns(aAmountsX , aPositionsY)
        for i in range(len(aIndividualX)):
            decReturn = aReturns[i]
            
            if decReturn <= 0:
                aIndividualX[i] = 0 
                aIndividualY[i] = 0


    def EnsureVariableDependency(self, aIndividualX, aIndividualY):
        for i in range(len(aIndividualX)):
            decX = aIndividualX[i]
            iY = aIndividualY[i]
            aIndividualY[i] = int(round(aIndividualY[i]))
        
            if iY == 0:
                aIndividualX[i] = 0
            
            if decX == 0:
                aIndividualY[i] = 0
                
    def EnsureRiskConstraint(self, aIndividualX, aIndividualY):
        
        while True:
         
            if sum(aIndividualX) == 0:
                break
    
            aWeightedRisks = list(np.multiply(self.aAverageRisks,aIndividualX))
            decWeightedAverage = sum(aWeightedRisks)/sum(aIndividualX)
                        
            if decWeightedAverage > self.decMaximumRiskRmse:
                decMaxRisk = max(aWeightedRisks)
                iIndexMaxRisk = aWeightedRisks.index(decMaxRisk)
                
                aIndividualX[iIndexMaxRisk] = 0
                aIndividualY[iIndexMaxRisk] = 0
            else:
                break



    def EnsureNonNegativeBalances(self, aIndividualX, aIndividualY):
        aAmountsX =  list(aIndividualX)
        aPositionsY = list(aIndividualY)
        
        aReturns = self.aCalculateReturns(aAmountsX , aPositionsY)
        aBalances = self.aCalculateBalances(aReturns, aAmountsX)
        
        aNegativeBalances = [i for i in aBalances if i < 0]
        
        if len(aNegativeBalances) == 0:
            return
        
        
        decMinimumBalance = min(aNegativeBalances)
        iIndexMinimumBalance = aBalances.index(decMinimumBalance)
        
        aAffectingIndices = self.dfIndices[self.dfIndices['j'] == iIndexMinimumBalance].index
        
        aAffectingReturns = np.array(aReturns)[aAffectingIndices]
        aAffectingReturns = list(aAffectingReturns)
        
    
        aNonZeroAffectingReturns = [i for i in aAffectingReturns if i != 0]        

        decMinAffectingReturn =min(aNonZeroAffectingReturns)

        iMinimumAffectingReturnIndex = aAffectingReturns.index(decMinAffectingReturn)
        iMinimumAffectingReturnIndex = aAffectingIndices[iMinimumAffectingReturnIndex]
        
        
        i = self.dfIndices.iloc[iMinimumAffectingReturnIndex].i
        j = iIndexMinimumBalance
        k = self.dfIndices.iloc[iMinimumAffectingReturnIndex].k
        
        # print("balance is broken: " + str(len(aNegativeBalances)) + "*****" + str(sum(aNegativeBalances)) + "*****" + str(decMinimumBalance) + "*****" + str(aIndividualX[iMinimumAffectingReturnIndex]) + "*****" + str(decMinAffectingReturn) + "******" +  str(aIndividualY[iMinimumAffectingReturnIndex]) + "******" + str(i)+ "******" + str(j)+ "******" + str(k))
          
    
        aIndividualX[iMinimumAffectingReturnIndex] = 0
        aIndividualY[iMinimumAffectingReturnIndex] = 0

        self.EnsureNonNegativeBalances(aIndividualX,aIndividualY)
        

    def decEvaluateFitness(self, aIndividualX, aIndividualY):
        aAmountsX =  list(aIndividualX)
        aPositionsY = list(aIndividualY)
        
        aReturns = self.aCalculateReturns(aAmountsX , aPositionsY)
        aBalances = self.aCalculateBalances(aReturns, aAmountsX)
        
        decFinalBalance = aBalances[len(aBalances)-1]
        decTotalProfit = decFinalBalance - self.decInitialBalance
        
        return decTotalProfit
    
    
    
    def Main(self):
        dfDecisonVariablesX = self.dfGetDecisionVariables("x")
        dfDecisonVariablesY = self.dfGetDecisionVariables("y")
        
        dfObjectiveFunctions =  self.dfGetObjectiveFunctions()
        
        creator.create("FitnessFunction", base.Fitness, weights=(1.0,))        
        
        oToolbox = base.Toolbox()
        oToolbox.register("evaluate", self.decEvaluateFitness)
        
        oGeneticAlgorithm = GeneticAlgorithm(self, creator.FitnessFunction,dfObjectiveFunctions, dfDecisonVariablesX, dfDecisonVariablesY)
        aOptimumResult = oGeneticAlgorithm.Optimize(oToolbox)
        
        # oParticleSwarmAlgorithm = ParticleSwarmOptimization(self, creator.FitnessFunction,dfObjectiveFunctions, dfDecisonVariablesX, dfDecisonVariablesY)
        # aOptimumResult = oParticleSwarmAlgorithm.Optimize(oToolbox)
        
        aOptimumAmounts, aOptiumumPositions = np.array_split(aOptimumResult, 2)
        

        return aOptimumAmounts, aOptiumumPositions


class variable_types(enum.Enum):
    integer = 1
    decimal = 2
    
    def convert_to_random(eType, vMin, vMax):
        if eType == variable_types.integer:
            return random.randint(vMin, vMax)
        elif eType == variable_types.decimal:
            return random.uniform(vMin, vMax)


class GeneticAlgorithm():
    
    def __init__(self, oPortfolioManager, FitnessFunction,dfObjectiveFunctions, dfDecisonVariablesX, dfDecisonVariablesY, iMaxNumberOfGenerations = 20, iPopulationSize = 50, decSimilarityRatio = 1.0, decCrossoverRate = 0.5,decMutationRate = 0.05, decMutationCrowdingDegree = 0.4):
        self.FitnessFunction = FitnessFunction
        self.dfObjectiveFunctions = dfObjectiveFunctions
        self.dfDecisonVariablesX = dfDecisonVariablesX
        self.dfDecisonVariablesY = dfDecisonVariablesY
        self.iMaxNumberOfGenerations = iMaxNumberOfGenerations
        self.iPopulationSize = iPopulationSize
        self.decSimilarityRatio = decSimilarityRatio
        self.decCrossoverRate = decCrossoverRate
        self.decMutationRate = decMutationRate
        self.decMutationCrowdingDegree = decMutationCrowdingDegree
        self.oPortfolioManager = oPortfolioManager

    
    def GenerateIndividual(self, toolbox, dfDecisonVariables, sGroupOfIndividual):

        sDecisionVariables = ""
        
        for iIndex, oRow in dfDecisonVariables.iterrows():
            sLabel = oRow["label"]
            eVariableType = oRow["variable_type"]
            decLowerBound = oRow["lower_bound"]
            decUpperBound = oRow["upper_bound"]
            
            toolbox.register(sLabel, variable_types.convert_to_random, eVariableType, decLowerBound, decUpperBound)
            
            
            if sDecisionVariables == "":
                sDecisionVariables = "toolbox."+sLabel
            else:
                sDecisionVariables = sDecisionVariables + "," + "toolbox."+sLabel

        toolbox.register(sGroupOfIndividual, tools.initCycle,creator.Individual, eval(sDecisionVariables))
    
            
    def select(self, aPopulationX, aPopulationY):
        aOffspringX = tools.selTournament(individuals = aPopulationX, k = len(aPopulationX), tournsize=self.iPopulationSize)
        aOffspringY = tools.selTournament(individuals = aPopulationY, k = len(aPopulationY), tournsize=self.iPopulationSize)
        
        return aOffspringX, aOffspringY

    
    def crossover(self, aIndividual1X, aIndividual2X, aIndividual1Y, aIndividual2Y):
        if random.random() < self.decCrossoverRate:
            size = len(aIndividual1X)
            cxpoint1 = random.randint(1, size)
            cxpoint2 = random.randint(1, size - 1)
            if cxpoint2 >= cxpoint1:
                cxpoint2 += 1
            else:
                cxpoint1, cxpoint2 = cxpoint2, cxpoint1
        
            aIndividual1X[cxpoint1:cxpoint2], aIndividual2X[cxpoint1:cxpoint2] = aIndividual2X[cxpoint1:cxpoint2].copy(), aIndividual1X[cxpoint1:cxpoint2].copy()
            aIndividual1Y[cxpoint1:cxpoint2], aIndividual2Y[cxpoint1:cxpoint2] = aIndividual2Y[cxpoint1:cxpoint2].copy(), aIndividual1Y[cxpoint1:cxpoint2].copy()
                            
            del aIndividual1X.fitness.values
            del aIndividual2X.fitness.values
            del aIndividual1Y.fitness.values
            del aIndividual2Y.fitness.values
                

    def mutate(self, aIndividual, dfDecisonVariables):            
        tools.mutPolynomialBounded(aIndividual,self.decMutationCrowdingDegree, dfDecisonVariables["lower_bound"].values.tolist(),dfDecisonVariables["upper_bound"].values.tolist(), self.decMutationRate)
    
    
    def Optimize(self, toolbox):
        creator.create("Individual", list, fitness = self.FitnessFunction)
        
        self.GenerateIndividual(toolbox, self.dfDecisonVariablesX, "x")
        self.GenerateIndividual(toolbox, self.dfDecisonVariablesY, "y")
        
        toolbox.register("select", self.select)
        toolbox.register("crossover", self.crossover)
        toolbox.register("mutate", self.mutate)
        
        toolbox.register("population_x", tools.initRepeat, list ,toolbox.x)
        toolbox.register("population_y", tools.initRepeat, list ,toolbox.y)
        
        aPopulationX = toolbox.population_x(self.iPopulationSize)
        aPopulationY = toolbox.population_y(self.iPopulationSize)
        
        self.oPortfolioManager.aAdjustPopulation(aPopulationX, aPopulationY)
        
        
        dfLabelsX =  self.dfDecisonVariablesX["label"]
        dfLabelsY =  self.dfDecisonVariablesY["label"] 
        dfBothDecisionVariableLabels = dfLabelsX.append(dfLabelsY)
                
        oOutputFile = ResultLogs.OpenOutputFile(dfBothDecisionVariableLabels,self.dfObjectiveFunctions["label"])
        
        
        aFitnesses = list(map(toolbox.evaluate, aPopulationX, aPopulationY))
        
        for ind, fit in zip(aPopulationX, aFitnesses):
            ind.fitness.values = (fit,)
            
            
        for ind, fit in zip(aPopulationY, aFitnesses):
            ind.fitness.values = (fit,)

           
        iRunId = 1
        for i in range(self.iMaxNumberOfGenerations):
            print("Generation #: " + str(i))
           
            aOffspringX, aOffspringY = toolbox.select(aPopulationX, aPopulationY)

            aOffspringX = list(map(toolbox.clone, aOffspringX)) 
            aOffspringY = list(map(toolbox.clone, aOffspringY))
            
            for aChild1X, aChild2X, aChild1Y,aChild2Y in zip(aOffspringX[::2], aOffspringX[1::2], aOffspringY[::2], aOffspringY[1::2]):            
                toolbox.crossover(aChild1X, aChild2X, aChild1Y, aChild2Y)
                
            for aMutantX in aOffspringX:
                toolbox.mutate(aMutantX,self.dfDecisonVariablesX)
    
            for aMutantY in aOffspringY:
                toolbox.mutate(aMutantY, self.dfDecisonVariablesY)
                
            
            self.oPortfolioManager.aAdjustPopulation(aOffspringX, aOffspringY)


            aFitnesses = list(map(toolbox.evaluate, aOffspringX, aOffspringY))

            for ind, fit in zip(aOffspringX, aFitnesses):
                ind.fitness.values = (fit,) 

            for ind, fit in zip(aOffspringY, aFitnesses):
                ind.fitness.values = (fit,)
            
 
            aPopulationX[:] = aOffspringX
            aPopulationY[:] = aOffspringY 
            
           
            for iIndex in range(len(aPopulationX)):
                aIndX = aPopulationX[iIndex]
                aIndY = aPopulationY[iIndex]
                aIndFull = aIndX + aIndY
                
                ResultLogs.LogToOutputFile(oOutputFile, iRunId, i+1,aIndFull, aIndX.fitness.values)
                iRunId = iRunId + 1
                
            aBestIndX = tools.selBest(aPopulationX, 1)[0]
            aBestIndY = tools.selBest(aPopulationY, 1)[0]
            
            iBestOccurance = aPopulationX.count(aBestIndX)
            
            if iBestOccurance/self.iPopulationSize >= self.decSimilarityRatio:
                break
                           
            
        aBestIndFull = aBestIndX + aBestIndY
        ResultLogs.LogToOutputFile(oOutputFile, "OPTIMUM RESULT", "", aBestIndFull, aBestIndX.fitness.values)
                
        return aBestIndFull


class ParticleSwarmOptimization():
    def __init__(self,oPortfolioManager, FitnessFunction,dfObjectiveFunctions, dfDecisonVariablesX, dfDecisonVariablesY, iMaxNumberOfGenerations = 20,iPopulationSize = 50 , decSimilarityRatio = 1, decInertiaWeight = 0.5, decExplotationCoefficient= 0.25, decExplorationCoefficient= 0.25):
        self.FitnessFunction = FitnessFunction
        self.dfObjectiveFunctions = dfObjectiveFunctions
        self.dfDecisonVariablesX = dfDecisonVariablesX
        self.dfDecisonVariablesY = dfDecisonVariablesY
        self.iMaxNumberOfGenerations = iMaxNumberOfGenerations
        self.iPopulationSize = iPopulationSize      
        self.decSimilarityRatio = decSimilarityRatio
        self.decInertiaWeight = decInertiaWeight
        self.decExplotationCoefficient = decExplotationCoefficient
        self.decExplorationCoefficient = decExplorationCoefficient
        self.oPortfolioManager = oPortfolioManager
        

    def GenerateParticle(self, toolbox,  dfDecisonVariables):
        sDecisionVariables = ""
        sDecisionVariableSpeeds = ""
        for iIndex, oRow in dfDecisonVariables.iterrows():
            sLabel = oRow["label"]
            sLabelSpeed= sLabel + "_speed"
            eVariableType = oRow["variable_type"]
            decLowerBound = oRow["lower_bound"]
            decUpperBound = oRow["upper_bound"]
            decStepSizeLower = oRow["step_size_lower"]
            decStepSizeUpper = oRow["step_size_upper"]    
            
            toolbox.register(sLabel, variable_types.convert_to_random,eVariableType , decLowerBound, decUpperBound)
            toolbox.register(sLabelSpeed, variable_types.convert_to_random, eVariableType, decStepSizeLower, decStepSizeUpper)
            
            if sDecisionVariables == "":
                sDecisionVariables = "toolbox."+sLabel
                sDecisionVariableSpeeds = "toolbox."+sLabelSpeed
            else:
                sDecisionVariables = sDecisionVariables + "," + "toolbox."+sLabel
                sDecisionVariableSpeeds = sDecisionVariableSpeeds + "," + "toolbox."+sLabelSpeed
        
                
        oParticle = creator.Particle(tools.initCycle(creator.Particle,eval(sDecisionVariables)))
        oParticle.aPersonalBest = oParticle
        oParticle.aSpeed = tools.initCycle(list, eval(sDecisionVariableSpeeds))
        oParticle.aSpeedMin = list(dfDecisonVariables["step_size_lower"])
        oParticle.aSpeedMax = list(dfDecisonVariables["step_size_upper"])
        oParticle.aLabel = list(dfDecisonVariables["label"])
            
        return oParticle
     
    
    def UpdateParticle(self, oParticle, oParticleGlobalBest, dfDecisonVariables):
        aRandom1 = np.random.uniform(0,1, len(oParticle))
        aCoeffCognitiveComponent = self.decExplotationCoefficient * aRandom1
        
        aDistanceToPersonalBest = map(operator.sub, oParticle.aPersonalBest, oParticle)
        aCognitiveComponent = list(map(operator.mul, aCoeffCognitiveComponent, aDistanceToPersonalBest))
        
        aRandom2 = np.random.uniform(0,1, len(oParticle))
        aCoeffSocialComponent = self.decExplorationCoefficient * aRandom2
        
        aDistanceToGlobalBest = map(operator.sub, oParticleGlobalBest, oParticle)
        aSocialComponent = list(map(operator.mul, aCoeffSocialComponent, aDistanceToGlobalBest))
        
        for i in range(len(oParticle)):
            eVariableType = dfDecisonVariables["variable_type"][i]
            decUpperBound = dfDecisonVariables["upper_bound"][i]
            decLowerBound = dfDecisonVariables["lower_bound"][i]
            decStepSizeLower = dfDecisonVariables["step_size_lower"][i]
            decStepSizeUpper = dfDecisonVariables["step_size_upper"][i]         
            
            decSpeedValue = oParticle.aSpeed[i] + aCognitiveComponent[i] + aSocialComponent[i]
            if decSpeedValue < decStepSizeLower :
                oParticle.aSpeed[i] = decStepSizeLower
            elif decSpeedValue > decStepSizeUpper:
                oParticle.aSpeed[i] = decStepSizeUpper
            
            if eVariableType == variable_types.integer:
                oParticle.aSpeed[i] = int(oParticle.aSpeed[i])
            
            oParticle[i] = oParticle[i] + oParticle.aSpeed[i]
            
            decValue = oParticle[i]
            if decValue < decLowerBound:
                oParticle[i] = decLowerBound
            elif decValue > decUpperBound:
                oParticle[i] = decUpperBound
            
            if eVariableType == variable_types.integer:
                oParticle[i] = int(oParticle[i])
        
        


    def Optimize(self, toolbox):
        creator.create("Particle", list, fitness = self.FitnessFunction, aSpeed = list ,aSpeedMin = list, aSpeedMax = list, aPersonalBest = list, aLabel = list)
        
        toolbox.register("x", self.GenerateParticle, toolbox, self.dfDecisonVariablesX)
        toolbox.register("y", self.GenerateParticle, toolbox, self.dfDecisonVariablesY)

        toolbox.register("update", self.UpdateParticle)

        toolbox.register("population_x", tools.initRepeat, list ,toolbox.x)
        toolbox.register("population_y", tools.initRepeat, list ,toolbox.y)

        aPopulationX = toolbox.population_x(self.iPopulationSize)
        aPopulationY = toolbox.population_y(self.iPopulationSize)
        
        self.oPortfolioManager.aAdjustPopulation(aPopulationX, aPopulationY)
        
        dfLabelsX =  self.dfDecisonVariablesX["label"] 
        dfLabelsY =  self.dfDecisonVariablesY["label"] 
        dfBothDecisionVariableLabels = dfLabelsX.append(dfLabelsY)
        
                
        oOutputFile = ResultLogs.OpenOutputFile(dfBothDecisionVariableLabels,self.dfObjectiveFunctions["label"])
                

        oParticleXGlobalBest = list
        oParticleYGlobalBest = list
        
        iRunId = 1
        for i in range(self.iMaxNumberOfGenerations):
            print("Generation #: " + str(i))
            
            for iIndex in range(len(aPopulationX)):

                oParticleX = aPopulationX[iIndex]
                oParticleY = aPopulationY[iIndex]
                
                aFitness = (toolbox.evaluate(oParticleX, oParticleY),)
                
                oParticleX.fitness.values = aFitness
                oParticleY.fitness.values = aFitness


                oParticle = oParticleX + oParticleY
                ResultLogs.LogToOutputFile(oOutputFile, iRunId, i+1,oParticle, aFitness)
                
                if not oParticleX.aPersonalBest:
                    oParticleX.aPersonalBest = creator.Particle(oParticleX)
                    oParticleX.aPersonalBest.fitness.values = aFitness
                    
                    oParticleY.aPersonalBest = creator.Particle(oParticleY)
                    oParticleY.aPersonalBest.fitness.values = aFitness
                    
                else:
                    if oParticleX.aPersonalBest.fitness < oParticleX.fitness:
                        oParticleX.aPersonalBest = creator.Particle(oParticleX)
                        oParticleX.aPersonalBest.fitness.values = aFitness
                        
                        oParticleY.aPersonalBest = creator.Particle(oParticleY)
                        oParticleY.aPersonalBest.fitness.values = aFitness
                        

                if oParticleXGlobalBest == list:
                    oParticleXGlobalBest = creator.Particle(oParticleX)
                    oParticleXGlobalBest.fitness.values = aFitness
                    
                    oParticleYGlobalBest = creator.Particle(oParticleY)
                    oParticleYGlobalBest.fitness.values = aFitness                                       
                else:
                    if oParticleXGlobalBest.fitness < oParticleX.fitness:
                       oParticleXGlobalBest = creator.Particle(oParticleX)
                       oParticleXGlobalBest.fitness.values = aFitness
                       
                       oParticleYGlobalBest = creator.Particle(oParticleY)
                       oParticleYGlobalBest.fitness.values = aFitness
                       
                iRunId = iRunId + 1
                

            for iIndex in range(len(aPopulationX)):
                oParticleX = aPopulationX[iIndex]
                # oParticleY = aPopulationY[iIndex]
                
                toolbox.update(oParticleX, oParticleXGlobalBest, self.dfDecisonVariablesX)     
                # toolbox.update(oParticleY, oParticleXGlobalBest, self.dfDecisonVariablesY)
            
            
            self.oPortfolioManager.aAdjustPopulation(aPopulationX, aPopulationY)
            
            iBestOccurance = aPopulationX.count(oParticleXGlobalBest)
            if iBestOccurance/self.iPopulationSize >= self.decSimilarityRatio:
                break
        

        oParticleGlobalBest = oParticleXGlobalBest + oParticleYGlobalBest
        ResultLogs.LogToOutputFile(oOutputFile, "OPTIMUM RESULT", "", oParticleGlobalBest, oParticleXGlobalBest.fitness.values)
                                
        return oParticleGlobalBest
    

class ResultLogs():
    def OpenOutputFile(aDecisionVariableLabels,aFitnessLabels):
        sTimeStamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        sFileName = sTimeStamp + '.txt'
        
        oOutputFile = open(sFileName, "a")

        sHeader = "Run ID"
        sHeader = sHeader + ';' + "Generation ID"
        sHeader = sHeader + ';' + "Time Stamp"
        sHeader = sHeader + ';' + ';'.join(str(x) for x in aDecisionVariableLabels)
        sHeader = sHeader + ';' +';'.join(str(x) for x in aFitnessLabels)
        oOutputFile.write(sHeader)

        return oOutputFile
    
    
    def LogToOutputFile(oOutputFile, iRunId, iGenerationId, aDecisionVariables, aFitnessValues):
        sTimeStamp = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
        sValues = str(iRunId)
        sValues = sValues + ';' + str(iGenerationId)
        sValues = sValues + ';' + str(sTimeStamp)
        sValues = sValues + ';' + ';'.join(str(x) for x in aDecisionVariables)
        sValues = sValues + ';' + ';'.join(str(x) for x in aFitnessValues)
        
        oOutputFile.write('\n' + sValues)
