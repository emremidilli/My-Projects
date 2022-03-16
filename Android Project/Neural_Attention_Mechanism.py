import numpy as np
import os
import tensorflow as tf
import pandas as pd
from sklearn import metrics


class Encoder(tf.keras.Model):
    def __init__(self, iFeatureSizeX, iEncoderUnits, iBatchSize):
        super(Encoder, self).__init__()
        self.iBatchSize = iBatchSize
        self.iEncoderUnits = iEncoderUnits
        
        self.gru = tf.keras.layers.GRU(self.iEncoderUnits,
                                        return_sequences=True,
                                        return_state=True,
                                        recurrent_initializer='glorot_uniform')

    def call(self, x, hidden): #incoming X must be a matrix since embedding layer is cancelled. Matrix shape is (iBackwardTimeWindow, backward_feature_size)
      aOutput, aState = self.gru(x, initial_state = hidden)
      return aOutput, aState

    def aInitializeHiddenState(self):
        return tf.zeros((self.iBatchSize, self.iEncoderUnits))
    
    

class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, iUnits):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(iUnits, use_bias=False)
        self.W2 = tf.keras.layers.Dense(iUnits, use_bias=False)
        self.V = tf.keras.layers.Dense(1)
    
    def call(self, aQuery, aValues): # aQuery = aDecoderHidden, aValues = aEncoderOutput
        # aQuery hidden state shape == (iBatchSize, hidden size)(64, 1024)
        # aQueryWithTimeAxis shape == (iBatchSize, 1, hidden size) (64, 1 , 1024)
        # aValues shape == (iBatchSize, max_len, hidden size) (64, 16, 1024)
        # we are doing this to broadcast addition along the time axis to calculate the score
        aQueryWithTimeAxis = tf.expand_dims(aQuery, 1)
    
        # score shape == (iBatchSize, max_length, 1) (64, 16, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (iBatchSize, max_length, iUnits) (64, 16, 1024)
        aScore = self.V(
            tf.nn.tanh(
                self.W1(aQueryWithTimeAxis) + self.W2(aValues)))
        
        #aAttentionWeights shape == (iBatchSize, max_length, 1) (64, 16,1)
        aAttentionWeights = tf.nn.softmax(aScore, axis=1)
    
        # aContextVector shape after sum == (iBatchSize, hidden_size) (64, 1024)
        aContextVector = aAttentionWeights * aValues
        aContextVector = tf.reduce_sum(aContextVector, axis=1)
        
        
    
        return aContextVector, aAttentionWeights


class Decoder(tf.keras.Model):
    def __init__(self, iFeatureSizeX, iDecoderUnits, iBatchSize): 
        super(Decoder, self).__init__()
        self.iBatchSize = iBatchSize
        self.iDecoderUnits = iDecoderUnits
    
        self.gru = tf.keras.layers.GRU(self.iDecoderUnits,
                                        return_sequences=True, #We use return_sequences=True here because we'd like to access the complete encoded sequence rather than the final summary state.
                                        return_state=True,
                                        recurrent_initializer='glorot_uniform')
        
        self.W = tf.keras.layers.Dense(self.iDecoderUnits, 
                                      activation=tf.math.tanh,
                                      use_bias=False)
        
        self.fc = tf.keras.layers.Dense(iFeatureSizeX, activation="sigmoid")
    
        self.attention = BahdanauAttention(self.iDecoderUnits)
    
    def call(self, aX, aHidden, aEncoderOutput): # dec_input, dec_hidden, enc_output
    
        # enc_output shape == (iBatchSize, max_length, hidden_size)
        aContextVector, aAttentionWeights = self.attention(aHidden, aEncoderOutput)
    
        aX = tf.concat([tf.expand_dims(aContextVector, 1), aX], axis=-1)
    
        # passing the concatenated vector to the GRU
        aOutput, aState = self.gru(aX)
        
        # output shape == (iBatchSize * 1, hidden_size)
        aOutput = tf.reshape(aOutput, (-1, aOutput.shape[2]))
        
        aOutput = self.W(aOutput)
        
        # output shape == (iBatchSize, vocab)
        aX = self.fc(aOutput)
        
        return aX, aState, aAttentionWeights
    


class Neural_Attention_Mechanism(tf.keras.Model):
    def __init__(self, sModelId,iFeatureSizeX , iFeatureSizeY, iWindowLengthX,iWindowLengthY, oOptimizer, fncLoss, iEpochSize, iBatchSize, iNumberOfHiddenNeurons): 
        super(Neural_Attention_Mechanism, self).__init__()
        self.sModelId = sModelId
        
        self.sModelDirectory = os.path.join(self.sModelId, "__model__")

        self.iFeatureSizeX = iFeatureSizeX
        self.iFeatureSizeY = iFeatureSizeY
        self.iBackwardTimeWindow = iWindowLengthX
        self.iForwardTimeWindow = iWindowLengthY
        
        self.oOptimizer = oOptimizer
        self.fncLoss = fncLoss
        
        self.iEpochSize = iEpochSize
        self.iBatchSize = iBatchSize
        self.iNumberOfHiddenNeurons = iNumberOfHiddenNeurons

        self.oEncoder = Encoder(self.iFeatureSizeX, self.iNumberOfHiddenNeurons, self.iBatchSize)
        self.oDecoder = Decoder(self.iFeatureSizeY, self.iNumberOfHiddenNeurons, self.iBatchSize)
    
    
    @tf.function
    def TrainStep(self, aInput, aTarget, aEncoderHidden):        
        fLoss = 0.0
        with tf.GradientTape() as tape:            
            aEncoderOutput, aEncoderHidden = self.oEncoder(aInput, aEncoderHidden)
            aDecoderHidden = aEncoderHidden            
            # start one-hot vector is the one hot vector of t.
            # dec_input = tf.expand_dims(targ[:, 0], 1) 
            aDecoderInput = tf.expand_dims(np.zeros((self.iBatchSize,self.iFeatureSizeY)) , 1)
            # Teacher forcing - feeding the target as the next input
            for t in range(0, self.iForwardTimeWindow):
                predictions, aDecoderHidden, _ = self.oDecoder(aDecoderInput, aDecoderHidden, aEncoderOutput)  
                            
                fLoss += self.fncLoss(aTarget[:, t], predictions)
                # using teacher forcing
                aDecoderInput = tf.expand_dims(aTarget[:, t], 1)
          

        aVariables = self.oEncoder.trainable_variables + self.oDecoder.trainable_variables
        aGradients = tape.gradient(fLoss, aVariables)
        self.oOptimizer.apply_gradients(zip(aGradients, aVariables))
        
    

    def dicGetMetricsAndLosses(self, aInputTrain, aOutputTrain, aInputValidation, aOutputValidation):
        aPredictionsTrain= self.aPredict(aInputTrain)
        aPredictionsValidation= self.aPredict(aInputValidation)
        fValidationLoss = 0.0            
        fTrainingLoss = 0.0

        
        for t in range(0, self.iForwardTimeWindow):
            aPredictionValidationTimeStep = aPredictionsValidation[:, t*self.iFeatureSizeY:(t+1)*self.iFeatureSizeY]
            aActualValidationTimeStep = aOutputValidation[:, t*self.iFeatureSizeY:(t+1)*self.iFeatureSizeY]
            
            aActualTrainTimeStep = aOutputTrain[:, t*self.iFeatureSizeY:(t+1)*self.iFeatureSizeY]
            aPredictionTrainTimeStep = aPredictionsTrain[:, t*self.iFeatureSizeY:(t+1)*self.iFeatureSizeY]

            fValidationLoss += self.fncLoss(aActualValidationTimeStep, aPredictionValidationTimeStep)
            
        
        fValidationLoss = fValidationLoss/self.iForwardTimeWindow
        fTrainingLoss = fTrainingLoss/self.iForwardTimeWindow
        
        fTrainingLoss = fTrainingLoss
        fValidationLoss = fValidationLoss
        
        dicHistory = {
                'training loss':[fTrainingLoss], 
                'validation loss': [fValidationLoss]
                }
        
        return dicHistory
        
    
    def Fit(self, aInputTrain, aOutputTrain, aInputValidation, aOutputValidation):        
        sCheckpointDirectory = os.path.join(self.sModelId, "__Training checkpoints__")
        sCheckpointPrefix = os.path.join(sCheckpointDirectory, "ckpt")
        
        oCheckPoint = tf.train.Checkpoint(optimizer=self.oOptimizer,encoder=self.oEncoder,decoder=self.oDecoder)
        
        iStepsPerEpoch = len(aInputTrain)//self.iBatchSize

        dsTrain = (tf.data.Dataset.from_tensor_slices((aInputTrain,aOutputTrain)))
        dsTrain = dsTrain.batch(self.iBatchSize, drop_remainder=True)
        
        dfHistory = pd.DataFrame()
        
        for iEpoch in range(self.iEpochSize):
            aEncoderHidden = self.oEncoder.aInitializeHiddenState()
     
            for (iBatch, (aInput, aTarget)) in enumerate(dsTrain.take(iStepsPerEpoch)):
                tf.print("epoch nr: "+ str(iEpoch) + " batch nr: "+ str(iBatch))
                aInput = tf.reshape(aInput, (self.iBatchSize, self.iBackwardTimeWindow, self.iFeatureSizeX, 1))
                aTarget = tf.reshape(aTarget, (self.iBatchSize, self.iForwardTimeWindow, self.iFeatureSizeY, 1 )) #since forward window length includes also t.
                
                aInput = tf.reshape(aInput, (self.iBatchSize, self.iBackwardTimeWindow, self.iFeatureSizeX))
                aTarget = tf.reshape(aTarget, (self.iBatchSize, self.iForwardTimeWindow,self.iFeatureSizeY))
                
                self.TrainStep(aInput, aTarget, aEncoderHidden)
                
                
            if (iEpoch + 1) % 2 == 0 or iEpoch == 0:
                oCheckPoint.write(file_prefix = sCheckpointPrefix)
 
            oCheckPoint.restore(tf.train.latest_checkpoint(sCheckpointDirectory))
        
            self.save_weights(self.sModelDirectory)
            
            dicHistory = self.dicGetMetricsAndLosses(
                aInputTrain, 
                aOutputTrain, 
                aInputValidation, 
                aOutputValidation
            )
            
            dfHistory = dfHistory.append(
                pd.DataFrame(
                    dicHistory, index = [iEpoch])
                )
            
        dfHistory.columns = pd.MultiIndex.from_product([['history'], dfHistory.columns])
        self.history = dfHistory
        
        self.save_weights(self.sModelDirectory)
        
    
    
    def aPredict(self, aInput) :
        self.load_weights(self.sModelDirectory)
        
        iBatchSize = len(aInput)
        
        dsTest = (tf.data.Dataset.from_tensor_slices((aInput)))
        
        dsTest = dsTest.batch(iBatchSize, drop_remainder=True)
        
        aEncoderHidden = tf.zeros((iBatchSize, self.iNumberOfHiddenNeurons))
    
        aPredictions = tf.zeros(1)
        for (iBatch, (aInputTest)) in enumerate(dsTest.take(-1)):
            aInputTest = tf.reshape(aInputTest, (iBatchSize, self.iBackwardTimeWindow, self.iFeatureSizeX, 1))
            aInputTest = tf.reshape(aInputTest, (iBatchSize, self.iBackwardTimeWindow, self.iFeatureSizeX))
    
            aEncoderOutput, aEncoderHidden = self.oEncoder(aInputTest, aEncoderHidden)
            
            aDecoderHidden = aEncoderHidden
            
            aDecoderInput = tf.expand_dims(np.zeros((iBatchSize,self.iFeatureSizeY)) , 1)
            
            aTimeStepPredictions = tf.zeros(1)
            for t in range(0, self.iForwardTimeWindow):
                aPred, dec_hidden, _ = self.oDecoder(aDecoderInput, aDecoderHidden, aEncoderOutput) # passing enc_output to the decoder
                aDecoderInput = tf.expand_dims(aPred, 1)
                
                if t == 0:
                    aTimeStepPredictions = aPred
                else:
                    aTimeStepPredictions = tf.concat([aTimeStepPredictions, aPred],1)
            
            aPredictions = aTimeStepPredictions
        
        aPredictions = aPredictions.numpy()
        
        return aPredictions