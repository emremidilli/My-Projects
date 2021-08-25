import operator
from deap import creator
from deap import tools
import enum
import random
import numpy as np
from datetime import datetime


class variable_types(enum.Enum):
    integer = 1
    decimal = 2
    
    
    def convert_to_random(eType, vMin, vMax):
        if eType == variable_types.integer:
            return random.randint(vMin, vMax)
        elif eType == variable_types.decimal:
            return random.uniform(vMin, vMax)



class genetic_algorithm():
    def __init__(self, fitness_function,dfObjectiveFunctions, dfDecisonVariables, iMaxNumberOfGenerations = 50, iPopulationSize = 20, decSimilarityRatio = 0.75, decCrossoverRate = 0.5,decMutationRate = 0.05, decMutationCrowdingDegree = 0.4):
        self.fitness_function = fitness_function
        self.dfObjectiveFunctions = dfObjectiveFunctions
        self.dfDecisonVariables = dfDecisonVariables
        self.iMaxNumberOfGenerations = iMaxNumberOfGenerations
        self.iPopulationSize = iPopulationSize
        self.decSimilarityRatio = decSimilarityRatio
        self.decCrossoverRate = decCrossoverRate
        self.decMutationRate = decMutationRate
        self.decMutationCrowdingDegree = decMutationCrowdingDegree
        
        
    def generate_individual(self, toolbox):
        creator.create("Individual", list, fitness = self.fitness_function)
        
        sGeneOrder = ""
        
        for iIndex, oRow in self.dfDecisonVariables.iterrows():
            sLabel = oRow["label"]
            eVariableType = oRow["variable_type"]
            decLowerBound = oRow["lower_bound"]
            decUpperBound = oRow["upper_bound"]
            toolbox.register(sLabel, variable_types.convert_to_random, eVariableType, decLowerBound, decUpperBound)
            
            if sGeneOrder == "":
                sGeneOrder = "toolbox."+sLabel
            else:
                sGeneOrder = sGeneOrder + "," + "toolbox."+sLabel
            
        toolbox.register("individual", tools.initCycle,creator.Individual, eval(sGeneOrder))
        
        
        
    def crossover(self, oIndividual1, oIndividual2):
        if random.random() < self.decCrossoverRate:
            size = len(oIndividual1)
            cxpoint1 = random.randint(1, size)
            cxpoint2 = random.randint(1, size - 1)
            if cxpoint2 >= cxpoint1:
                cxpoint2 += 1
            else:
                cxpoint1, cxpoint2 = cxpoint2, cxpoint1
        
            oIndividual1[cxpoint1:cxpoint2], oIndividual2[cxpoint1:cxpoint2] = oIndividual2[cxpoint1:cxpoint2].copy(), oIndividual1[cxpoint1:cxpoint2].copy()
                

            for i in range(len(oIndividual1)):
                eVariableType = self.dfDecisonVariables["variable_type"][i]
                decUpperBound = self.dfDecisonVariables["upper_bound"][i]
                decLowerBound = self.dfDecisonVariables["lower_bound"][i]
    
                if oIndividual1[i] < decLowerBound:
                    oIndividual1[i] = decLowerBound
                elif oIndividual1[i] > decUpperBound:
                    oIndividual1[i] = decUpperBound
                
                if oIndividual2[i] < decLowerBound:
                    oIndividual2[i] = decLowerBound
                elif oIndividual2[i] > decUpperBound:
                    oIndividual2[i] = decUpperBound
                
                if eVariableType == variable_types.integer:
                    oIndividual1[i]= int(oIndividual1[i])
                    oIndividual2[i]= int(oIndividual2[i])
                
                del oIndividual1.fitness.values
                del oIndividual2.fitness.values

    
    
    def mutate(self, oIndividual):            
        tools.mutPolynomialBounded(oIndividual,self.decMutationCrowdingDegree, self.dfDecisonVariables["lower_bound"].values.tolist(),self.dfDecisonVariables["upper_bound"].values.tolist(), self.decMutationRate)
        
        for i in range(len(oIndividual)):
            eVariableType = self.dfDecisonVariables["variable_type"][i]
            decUpperBound = self.dfDecisonVariables["upper_bound"][i]
            decLowerBound = self.dfDecisonVariables["lower_bound"][i]
            if oIndividual[i] < decLowerBound:
                oIndividual[i] = decLowerBound
            elif oIndividual[i] > decUpperBound:
                oIndividual[i] = decUpperBound
                        
            if eVariableType == variable_types.integer:
                oIndividual[i]= int(oIndividual[i])
        
        oIndividual.fitness.values
            
            
    def optimize(self, toolbox):
        self.generate_individual(toolbox)
        
        toolbox.register("population", tools.initRepeat, list ,toolbox.individual)
        toolbox.register("select", tools.selTournament, tournsize=self.iPopulationSize)
        toolbox.register("crossover", self.crossover)
        toolbox.register("mutate", self.mutate)
        
        aPopulation = toolbox.population(self.iPopulationSize)
        
        oOutputFile = Result_Logs.open_output_file(self.dfDecisonVariables["label"],self.dfObjectiveFunctions["label"] )
        
        aFitnesses = list(map(toolbox.evaluate, aPopulation))
        for ind, fit in zip(aPopulation, aFitnesses):
            ind.fitness.values = (fit,)
            
            
        iRunId = 1
        for i in range(self.iMaxNumberOfGenerations):
            print(i)
            # Select the next generation individuals
            offspring = toolbox.select(aPopulation, len(aPopulation))
            # Clone the selected individuals
            offspring = list(map(toolbox.clone, offspring))
            
            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                toolbox.crossover(child1, child2)
            
            for mutant in offspring:
                toolbox.mutate(mutant)
    
            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            aFitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, aFitnesses):
                ind.fitness.values = (fit,)
                
            # The population is entirely replaced by the offspring
            aPopulation[:] = offspring
            
            # Gather all the aFitnesses in one list and print the stats
            for ind in aPopulation:
                Result_Logs.log_to_output_file(oOutputFile, iRunId, i+1,ind, ind.fitness.values)
                iRunId = iRunId + 1
                
            aBestInd = tools.selBest(aPopulation, 1)[0]
            
            iBestOccurance = aPopulation.count(aBestInd)
            
            if iBestOccurance/self.iPopulationSize >= self.decSimilarityRatio:
                break
    
        Result_Logs.log_to_output_file(oOutputFile, "OPTIMUM RESULT", "", aBestInd, aBestInd.fitness.values)
                
        return aBestInd
    
class particle_swarm_optimization():
    def __init__(self,fitness_function, dfObjectiveFunctions,dfDecisonVariables, iMaxNumberOfGenerations,iPopulationSize, decSimilarityRatio, decInertiaWeight, decExplotationCoefficient, decExplorationCoefficient):
        self.fitness_function = fitness_function
        self.dfObjectiveFunctions = dfObjectiveFunctions
        self.dfDecisonVariables = dfDecisonVariables
        self.iMaxNumberOfGenerations = iMaxNumberOfGenerations
        self.iPopulationSize = iPopulationSize      
        self.decSimilarityRatio = decSimilarityRatio
        self.decInertiaWeight = decInertiaWeight
        self.decExplotationCoefficient = decExplotationCoefficient
        self.decExplorationCoefficient = decExplorationCoefficient
        

        
    def generate_particle(self, toolbox):
        creator.create("Particle", list, fitness = self.fitness_function, aSpeed = list ,aSpeedMin = list, aSpeedMax = list, aPersonalBest = list, aLabel = list)
        
        sDecisionVariables = ""
        sDecisionVariableSpeeds = ""
        for iIndex, oRow in self.dfDecisonVariables.iterrows():
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
        oParticle.aSpeedMin = list(self.dfDecisonVariables["lower_bound"])
        oParticle.aSpeedMax = list(self.dfDecisonVariables["upper_bound"])
        oParticle.aLabel = list(self.dfDecisonVariables["label"])
        
        return oParticle
    
    
    def update_particle(self, oParticle, oParticleGlobalBest):
        print("Before Update" + str(oParticle))
        
        aRandom1 = np.random.uniform(0,1, len(oParticle))
        aCoeffCognitiveComponent = self.decExplotationCoefficient * aRandom1
        
        aDistanceToPersonalBest = map(operator.sub, oParticle.aPersonalBest, oParticle)
        aCognitiveComponent = list(map(operator.mul, aCoeffCognitiveComponent, aDistanceToPersonalBest))
        
        aRandom2 = np.random.uniform(0,1, len(oParticle))
        aCoeffSocialComponent = self.decExplorationCoefficient * aRandom2
        
        aDistanceToGlobalBest = map(operator.sub, oParticleGlobalBest, oParticle)
        aSocialComponent = list(map(operator.mul, aCoeffSocialComponent, aDistanceToGlobalBest))
        
        for i in range(len(oParticle)):
            eVariableType = self.dfDecisonVariables["variable_type"][i]
            decUpperBound = self.dfDecisonVariables["upper_bound"][i]
            decLowerBound = self.dfDecisonVariables["lower_bound"][i]
            decStepSizeLower = self.dfDecisonVariables["step_size_lower"][i]
            decStepSizeUpper = self.dfDecisonVariables["step_size_upper"][i]         
            
            
            decSpeedValue = oParticle.aSpeed[i] + aCognitiveComponent[i] + aSocialComponent[i]
            # if decSpeedValue < decStepSizeLower :
            #     oParticle.aSpeed[i] = decStepSizeLower
            # elif decSpeedValue > decStepSizeUpper:
            #     oParticle.aSpeed[i] = decStepSizeUpper
            
            # if eVariableType == variable_types.integer:
            #     oParticle.aSpeed[i] = int(oParticle.aSpeed[i])
            
            oParticle[i] = oParticle[i] + oParticle.aSpeed[i]
            
            
            decValue = oParticle[i]
            if decValue < decLowerBound:
                oParticle[i] = decLowerBound
            elif decValue > decUpperBound:
                oParticle[i] = decUpperBound
            
            if eVariableType == variable_types.integer:
                oParticle[i] = int(oParticle[i])
        
        
                
        print("After Update" + str(oParticle))

    def optimize(self, toolbox):
        toolbox.register("particle", self.generate_particle, toolbox)
        toolbox.register("population", tools.initRepeat, list ,toolbox.particle)
        toolbox.register("update", self.update_particle)
        
        oOutputFile = Result_Logs.open_output_file(self.dfDecisonVariables["label"],self.dfObjectiveFunctions["label"] )
        
        aPopulation = toolbox.population(self.iPopulationSize)
        oParticleGlobalBest = list
        
        iRunId = 1
        for i in range(self.iMaxNumberOfGenerations):
            for oParticle in aPopulation:

                oParticle.fitness.values = (toolbox.evaluate(oParticle),)
                
                print("particle: " + str(iRunId+1))
                
                Result_Logs.log_to_output_file(oOutputFile, iRunId, i+1,oParticle, oParticle.fitness.values)
                
                if not oParticle.aPersonalBest:
                    oParticle.aPersonalBest = creator.Particle(oParticle)
                    oParticle.aPersonalBest.fitness.values = oParticle.fitness.values                    
                else:                    
                    if oParticle.aPersonalBest.fitness < oParticle.fitness:
                        oParticle.aPersonalBest = creator.Particle(oParticle)
                        oParticle.aPersonalBest.fitness.values = oParticle.fitness.values    
                    
                if oParticleGlobalBest == list:
                    oParticleGlobalBest = creator.Particle(oParticle)
                    oParticleGlobalBest.fitness.values = oParticle.fitness.values
                else:
                    if oParticleGlobalBest.fitness < oParticle.fitness:
                       oParticleGlobalBest = creator.Particle(oParticle)
                       oParticleGlobalBest.fitness.values = oParticle.fitness.values
                       
                print("global best: " + str(oParticleGlobalBest.fitness.values))

                iRunId = iRunId + 1
            for oParticle in aPopulation:
                toolbox.update(oParticle, oParticleGlobalBest)
                
                
            iBestOccurance = aPopulation.count(oParticleGlobalBest)
            if iBestOccurance/self.iPopulationSize >= self.decSimilarityRatio:
                break
                
        Result_Logs.log_to_output_file(oOutputFile, "OPTIMUM RESULT", "", oParticleGlobalBest, oParticleGlobalBest.fitness.values)
                                
        return oParticleGlobalBest
    

class Result_Logs():
    def open_output_file(aDecisionVariableLabels,aFitnessLabels):
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
    
    
    def log_to_output_file(oOutputFile, iRunId, iGenerationId, aDecisionVariables, aFitnessValues):
        sTimeStamp = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
        sValues = str(iRunId)
        sValues = sValues + ';' + str(iGenerationId)
        sValues = sValues + ';' + str(sTimeStamp)
        sValues = sValues + ';' + ';'.join(str(x) for x in aDecisionVariables)
        sValues = sValues + ';' + ';'.join(str(x) for x in aFitnessValues)
        
        oOutputFile.write('\n' + sValues)
        
        