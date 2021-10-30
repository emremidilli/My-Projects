import os
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM
from tensorflow.keras import regularizers
import numpy as np



class Long_Short_Term_Memory(Sequential):
    def __init__(self, sModelId,iFeatureSizeX , iFeatureSizeY, iWindowLengthX,iWindowLengthY, bIsClassification = True): 
        super(Long_Short_Term_Memory, self).__init__()
        self.sModelId = sModelId
        
        self.sModelDirectory = os.path.join(self.sModelId, "__model__")

        self.iFeatureSizeX = iFeatureSizeX
        self.iFeatureSizeY = iFeatureSizeY
        self.backward_window_length = iWindowLengthX
        self.forward_window_length = iWindowLengthY
        self.bIsClassification = bIsClassification
    
        self.SetHyperparameters()
        
    
    def SetHyperparameters(self, iBatchSize = 512):
        
        self.iBatchSize = iBatchSize
        

        self.iNumberOfHiddenNeuron = self.iFeatureSizeX*self.backward_window_length*2

        self.optimizer = tf.keras.optimizers.Adam(learning_rate= 1e-05, beta_1=0.1)
        
        if self.bIsClassification == True:
            self.loss_function = tf.keras.losses.BinaryCrossentropy()
        else:
            self.loss_function = tf.keras.losses.MeanSquaredError()
            

        oKernelRegulizer = regularizers.l2(0.001)

        self.add(LSTM(self.iBatchSize, return_sequences=True))
        
        self.add(Dense((self.iNumberOfHiddenNeuron)))
        
        self.add(Dense((self.iNumberOfHiddenNeuron)))
        
        if self.bIsClassification == True:
            self.add(Dense((self.forward_window_length * self.iFeatureSizeY), activation="sigmoid"))
        else:
            self.add(Dense((self.forward_window_length * self.iFeatureSizeY), activation="relu"))

        self.compile(optimizer=self.optimizer, loss=self.loss_function)



    
    def Train(self, aInputTrain, aOutputTrain, aInputValidation, aOutputValidation):        
        checkpoint_dir = os.path.join(self.sModelId, "__training checkpoints__")
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")    
        checkpoint = tf.train.Checkpoint(optimizer=self.optimizer,modelLstm = self)
        
        oEarlyStop = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 0 , patience = 20)

        aInputTrainExpanded = tf.expand_dims(aInputTrain, axis = 1)
        aOutputTrainExpanded = tf.expand_dims(aOutputTrain, axis = 1)
        aInputValidationExpanded = tf.expand_dims(aInputValidation, axis = 1)
        aOutputValidationExpanded = tf.expand_dims(aOutputValidation, axis = 1)
        
        self.fit(
            aInputTrainExpanded, 
            aOutputTrainExpanded, 
            epochs=10000, 
            batch_size=self.iBatchSize, 
            verbose=1, 
            validation_data= (aInputValidationExpanded, aOutputValidationExpanded), callbacks=[oEarlyStop] )
            
        checkpoint.write(file_prefix = checkpoint_prefix)
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        self.save_weights(self.sModelDirectory)
        
        
        pd.DataFrame(self.history.history).plot()
        
        
        
        
    
    def aPredict(self, aInputTest):
        self.load_weights(self.sModelDirectory)
        
        aInputTestExpanded = tf.expand_dims(aInputTest, axis = 1)
        
        
        aPredictionsProb = self.predict(aInputTestExpanded)
        
        if self.bIsClassification == True:
            aPredictionsOneHot = tf.one_hot(tf.nn.top_k(aPredictionsProb).indices, tf.shape(aPredictionsProb)[2])
            aPredictionsOneHot = tf.squeeze(aPredictionsOneHot, [1, 2])
        
        aPredictionsProb = tf.squeeze(aPredictionsProb, [1])

        aPredictionsProb = aPredictionsProb.numpy()
        
        if self.bIsClassification == True:
            aPredictionsOneHot = aPredictionsOneHot.numpy()
            return aPredictionsProb, aPredictionsOneHot
        else:
            return aPredictionsProb