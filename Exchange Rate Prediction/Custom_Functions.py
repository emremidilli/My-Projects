import pandas as pd
import numpy as np
import tensorflow as tf

def fCalculateLoss(aActual, aPrediction):
    aErrors = tf.abs(
        tf.subtract(aActual, aPrediction) 
    ) #(row: iBatchSize, col: iForwardTimeWindow-1)

    fBiggestError = tf.math.reduce_max(aErrors)
    
    aLossesDueToErrors = tf.reduce_sum(aErrors, 0) #(row: 1, col: iForwardTimeWindow)
    
    aDeltaSignsOfReturns = tf.abs(
        tf.sign(aActual) - tf.sign(aPrediction)
    ) # will be between [0 : no sign diff, 2: diff signs]  (row: iBatchSize, col: iForwardTimeWindow)
    
    aLossesDueToSignsOfReturns = tf.math.reduce_sum(aDeltaSignsOfReturns, 0) #(row: 1, col: iForwardTimeWindow)
    aTotalLosses = (aLossesDueToErrors * aLossesDueToSignsOfReturns * fBiggestError * 999) + aLossesDueToErrors
    
    fLoss = tf.math.reduce_mean(aTotalLosses)
    
    return fLoss