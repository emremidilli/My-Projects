import numpy as np
import os
import tensorflow as tf
import pandas as pd
from sklearn import metrics


class Encoder(tf.keras.Model):
    def __init__(self, one_hot_size, enc_units, batch_sz, activation_function, dropout_rate, recurrent_dropout_rate, oKernelRegulizer):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                        return_sequences=True,
                                        return_state=True,
                                        recurrent_initializer='glorot_uniform',
                                        # activation = activation_function, 
                                        dropout = dropout_rate , 
                                        recurrent_dropout= recurrent_dropout_rate,
                                        kernel_regularizer=oKernelRegulizer)

    def call(self, x, hidden): #incoming X must be a matrix since embedding layer is cancelled. Matrix shape is (iBackwardTimeWindow, backward_feature_size) 
      output, state = self.gru(x, initial_state = hidden)
      return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))
    
    

class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units, oKernelRegulizer):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units, kernel_regularizer=oKernelRegulizer)
        self.W2 = tf.keras.layers.Dense(units, kernel_regularizer=oKernelRegulizer)
        self.V = tf.keras.layers.Dense(1,kernel_regularizer=oKernelRegulizer)
    
    def call(self, query, values): # query = decoder_hidden, values = encoder_output
        # query hidden state shape == (batch_size, hidden size)(64, 1024)
        # query_with_time_axis shape == (batch_size, 1, hidden size) (64, 1 , 1024)
        # values shape == (batch_size, max_len, hidden size) (64, 16, 1024)
        # we are doing this to broadcast addition along the time axis to calculate the score
        query_with_time_axis = tf.expand_dims(query, 1)
    
        # score shape == (batch_size, max_length, 1) (64, 16, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units) (64, 16, 1024)
        score = self.V(tf.nn.tanh(
            self.W1(query_with_time_axis) + self.W2(values)))
        
        #Attention_weights shape == (batch_size, max_length, 1) (64, 16,1)
        attention_weights = tf.nn.softmax(score, axis=1)
    
        # context_vector shape after sum == (batch_size, hidden_size) (64, 1024)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        
        
    
        return context_vector, attention_weights


class Decoder(tf.keras.Model):
    def __init__(self, one_hot_size, dec_units, batch_sz, activation_function, dropout_rate, recurrent_dropout_rate, oKernelRegulizer): 
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
    
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                        return_sequences=True, #We use return_sequences=True here because we'd like to access the complete encoded sequence rather than the final summary state.
                                        return_state=True,
                                        recurrent_initializer='glorot_uniform',
                                        # activation = activation_function,
                                        dropout = dropout_rate , 
                                        recurrent_dropout= recurrent_dropout_rate,
                                        kernel_regularizer = oKernelRegulizer)
        
        self.fc = tf.keras.layers.Dense(one_hot_size, activation = 'relu', 
                                        kernel_regularizer=oKernelRegulizer,)
    
        self.attention = BahdanauAttention(self.dec_units, oKernelRegulizer)
    
    def call(self, x, hidden, enc_output, bReturnOneHotEncoded = False): # dec_input, dec_hidden, enc_output
    
        # enc_output shape == (batch_size, max_length, hidden_size)
        context_vector, attention_weights = self.attention(hidden, enc_output)
    
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
    
        # passing the concatenated vector to the GRU
        output, state = self.gru(x)
        
        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))
    
        # output shape == (batch_size, vocab)
        x = self.fc(output)
        
        if bReturnOneHotEncoded == True:
            aOneHotEncoded = tf.one_hot(tf.nn.top_k(x).indices, tf.shape(x)[1])
            aOneHotEncoded = tf.squeeze(aOneHotEncoded, [1])
            return x, state, attention_weights, aOneHotEncoded
        else:
            return x, state, attention_weights
    


class Neural_Attention_Mechanism(tf.keras.Model):
    def __init__(self, model_id,iFeatureSizeX , iFeatureSizeY, iWindowLengthX,iWindowLengthY): 
        super(Neural_Attention_Mechanism, self).__init__()
        self.model_id = model_id
        
        self.model_directory = os.path.join(self.model_id, "__model__")

        self.feature_size_input = iFeatureSizeX
        self.feature_size_target = iFeatureSizeY
        self.iBackwardTimeWindow = iWindowLengthX
        self.iForwardTimeWindow = iWindowLengthY
        
        self.set_hyperparameters()
        
    
    def set_hyperparameters(self,epoch_size = 50, batch_size = 128, iNumberOfHiddenNeurons = None, dropout_rate_encoder = 0.0,dropout_rate_decoder=0.0, recurrent_dropout_rate_encoder = 0.0, recurrent_dropout_rate_decoder=0.0, learning_rate = 0.01, momentum_rate=0.9):
        self.epoch_size = epoch_size
        self.batch_size = batch_size
        
        if iNumberOfHiddenNeurons is None:
            iNumberOfHiddenNeurons = self.feature_size_input*self.iBackwardTimeWindow*2
            
        self.iNumberOfHiddenNeurons = iNumberOfHiddenNeurons
        self.dropout_rate_encoder = dropout_rate_encoder
        self.dropout_rate_decoder = dropout_rate_decoder
        self.recurrent_dropout_rate_encoder = recurrent_dropout_rate_encoder
        self.recurrent_dropout_rate_decoder = recurrent_dropout_rate_decoder
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        
        # self.optimizer = tf.keras.optimizers.Adam(learning_rate= self.learning_rate, beta_1=self.momentum_rate)
    
        self.optimizer = tf.keras.optimizers.SGD(learning_rate= self.learning_rate, clipvalue=0.5)
    
        
        self.loss_function = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.activation_function = 'tanh'
        
        self.oKernelRegulizer = tf.keras.regularizers.l1(0.0001)
        
        self.encoder = Encoder(self.feature_size_input, self.iNumberOfHiddenNeurons, self.batch_size,self.activation_function, self.dropout_rate_encoder, self.recurrent_dropout_rate_encoder, self.oKernelRegulizer)
        self.decoder = Decoder(self.feature_size_target, self.iNumberOfHiddenNeurons, self.batch_size, self.activation_function, self.dropout_rate_decoder, self.recurrent_dropout_rate_decoder, self.oKernelRegulizer)
    
    
    @tf.function
    def train_step(self, inp, targ, enc_hidden):        
        loss = 0.0
        with tf.GradientTape() as tape:            
            enc_output, enc_hidden = self.encoder(inp, enc_hidden)
            dec_hidden = enc_hidden            
            # start one-hot vector is the one hot vector of t.
            # dec_input = tf.expand_dims(targ[:, 0], 1) 
            dec_input = tf.expand_dims(np.zeros((self.batch_size,self.feature_size_target)) , 1)
            # Teacher forcing - feeding the target as the next input
            for t in range(1, targ.shape[1]):
                # passing enc_output to the decoder
                predictions, dec_hidden, _ = self.decoder(dec_input, dec_hidden, enc_output)  
                            
                loss += self.loss_function(targ[:, t], predictions)
                # using teacher forcing
                dec_input = tf.expand_dims(targ[:, t], 1)
          

        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        
    

    def dicGetMetricsAndLosses(self, aScaledInputTrain, aScaledOutputTrain, aScaledInputValidation, aScaledOutputValidation):
        aPredictionsValidation, aPredicitonsValidationOneHot = self.aPredict(aScaledInputValidation)
        aPredictionsTrain, aPredicitonsTrainOneHot = self.aPredict(aScaledInputTrain)
           
        decValidationLoss = decValidationAccuracy = decValidationPrecision = decValidationRecall = decValidationF1Score = 0.0            
        decTrainingLoss = decTrainingAccuracy = decTrainingPrecision = decTrainingRecall =  decTrainingF1Score = 0.0

        
        for t in range(0, self.iForwardTimeWindow):
            aPredictionValidationTimeStep = aPredictionsValidation[:, t*self.feature_size_target:(t+1)*self.feature_size_target]
            aPredictionValidationOneHotTimeStep = aPredicitonsValidationOneHot[:, t*self.feature_size_target:(t+1)*self.feature_size_target]
            aActualValidationTimeStep = aScaledOutputValidation[:, t*self.feature_size_target:(t+1)*self.feature_size_target]
            
            aActualTrainTimeStep = aScaledOutputTrain[:, t*self.feature_size_target:(t+1)*self.feature_size_target]
            aPredictionTrainTimeStep = aPredictionsTrain[:, t*self.feature_size_target:(t+1)*self.feature_size_target]
            aPredictionTrainOneHotTimeStep = aPredicitonsTrainOneHot[:, t*self.feature_size_target:(t+1)*self.feature_size_target]
            

            decValidationAccuracy +=  metrics.accuracy_score(aActualValidationTimeStep, aPredictionValidationOneHotTimeStep)
            decValidationPrecision += metrics.precision_score(aActualValidationTimeStep, aPredictionValidationOneHotTimeStep, zero_division=0, average='micro')
            decValidationRecall += metrics.recall_score(aActualValidationTimeStep, aPredictionValidationOneHotTimeStep, zero_division=0, average='micro')
            decValidationF1Score += metrics.f1_score(aActualValidationTimeStep, aPredictionValidationOneHotTimeStep, zero_division=0, average='micro')
            decValidationLoss += self.loss_function(aActualValidationTimeStep, aPredictionValidationTimeStep)
            

            decTrainingAccuracy += metrics.accuracy_score(aActualTrainTimeStep, aPredictionTrainOneHotTimeStep)
            decTrainingPrecision += metrics.precision_score(aActualTrainTimeStep, aPredictionTrainOneHotTimeStep, zero_division=0, average='micro')
            decTrainingRecall += metrics.recall_score(aActualTrainTimeStep, aPredictionTrainOneHotTimeStep, zero_division=0, average='micro')
            decTrainingF1Score +=  metrics.f1_score(aActualTrainTimeStep, aPredictionTrainOneHotTimeStep, zero_division=0, average='micro')
            decTrainingLoss += self.loss_function(aActualTrainTimeStep, aPredictionTrainTimeStep)             

        
        
        decValidationAccuracy = decValidationAccuracy/self.iForwardTimeWindow
        decValidationPrecision = decValidationPrecision/self.iForwardTimeWindow
        decValidationRecall = decValidationRecall/self.iForwardTimeWindow
        decValidationF1Score = decValidationF1Score/self.iForwardTimeWindow
        decValidationLoss = decValidationLoss/self.iForwardTimeWindow
        
        
        decTrainingAccuracy = decTrainingAccuracy/self.iForwardTimeWindow
        decTrainingPrecision = decTrainingPrecision/self.iForwardTimeWindow
        decTrainingRecall = decTrainingRecall/self.iForwardTimeWindow
        decTrainingF1Score = decTrainingF1Score/self.iForwardTimeWindow
        decTrainingLoss = decTrainingLoss/self.iForwardTimeWindow
        
        
        decTrainingLoss = decTrainingLoss.numpy()
        decValidationLoss = decValidationLoss.numpy()
        
        dicHistory = {
                'training loss':[decTrainingLoss], 
                'validation loss': [decValidationLoss],
                'training accuracy': [decTrainingAccuracy],
                'validation accuracy': [decValidationAccuracy],
                'training precision': [decTrainingPrecision],
                'validation precision': [decValidationPrecision],
                'training recall': [decTrainingRecall],
                'validation recall': [decValidationRecall],
                'training f1-score': [decTrainingF1Score],
                'validation f1-score': [decValidationF1Score]
                }
        
        return dicHistory
        
    
    def train(self, aScaledInputTrain, aScaledOutputTrain, aScaledInputValidation, aScaledOutputValidation):        
        checkpoint_dir = os.path.join(self.model_id, "__training checkpoints__")
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        
        
        oCheckPoint = tf.train.Checkpoint(optimizer=self.optimizer,encoder=self.encoder,decoder=self.decoder)
        
        steps_per_epoch = len(aScaledInputTrain)//self.batch_size
        buffer_size = len(aScaledInputTrain)
        
        dataset = (tf.data.Dataset.from_tensor_slices((aScaledInputTrain,aScaledOutputTrain)))
        dataset = dataset.shuffle(buffer_size)
        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        
        dfHistory = pd.DataFrame()
        
        for epoch in range(self.epoch_size):
            encoder_hidden = self.encoder.initialize_hidden_state()

                    
            for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
                inp = tf.reshape(inp, (self.batch_size, self.iBackwardTimeWindow, self.feature_size_input, 1))
                targ = tf.reshape(targ, (self.batch_size, self.iForwardTimeWindow, self.feature_size_target, 1 )) #since forward window length includes also t.
                
                inp = tf.reshape(inp, (self.batch_size, self.iBackwardTimeWindow, self.feature_size_input))
                targ = tf.reshape(targ, (self.batch_size, self.iForwardTimeWindow,self.feature_size_target))
                
                self.train_step(inp, targ, encoder_hidden)
                
                
            if (epoch + 1) % 2 == 0 or epoch == 0:
                oCheckPoint.write(file_prefix = checkpoint_prefix)
 
            oCheckPoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        
            self.save_weights(self.model_directory)
            
            dicHistory = self.dicGetMetricsAndLosses(aScaledInputTrain, aScaledOutputTrain, aScaledInputValidation, aScaledOutputValidation)
            
            dfHistory = dfHistory.append(
                pd.DataFrame(
                    dicHistory, index = [epoch])
                )
            
        dfHistory.columns = pd.MultiIndex.from_product([['history'], dfHistory.columns])
        self.history = dfHistory
        
        self.save_weights(self.model_directory)
        
    
    
    def aPredict(self, aInput) :
        self.load_weights(self.model_directory)
        
        iBatchSize = len(aInput)
        
        dataset_test = (tf.data.Dataset.from_tensor_slices((aInput)))
        
        dataset_test = dataset_test.batch(iBatchSize, drop_remainder=True)
        
        encoder_hidden = tf.zeros((iBatchSize, self.iNumberOfHiddenNeurons))
    
        aPredictionsProb = tf.zeros(1)
        aPredictionsOneHot =  tf.zeros(1)
        for (batch, (inp_test)) in enumerate(dataset_test.take(-1)):
            inp_test = tf.reshape(inp_test, (iBatchSize, self.iBackwardTimeWindow, self.feature_size_input, 1))
            inp_test = tf.reshape(inp_test, (iBatchSize, self.iBackwardTimeWindow, self.feature_size_input))
    
            encoder_output, encoder_hidden = self.encoder(inp_test, encoder_hidden)
            
            decoder_hidden = encoder_hidden
            
            decoder_input = tf.expand_dims(np.zeros((iBatchSize,self.feature_size_target)) , 1)
            
            pProb = tf.zeros(1)
            pOneHot = tf.zeros(1)
            for t in range(0, self.iForwardTimeWindow):
                pred, dec_hidden, _, aPredOneHot = self.decoder(decoder_input, decoder_hidden, encoder_output ,True) # passing enc_output to the decoder
                
                decoder_input = tf.expand_dims(aPredOneHot, 1)
                
                if t == 0:
                    pProb = pred
                    pOneHot = aPredOneHot
                else:
                    pProb = tf.concat([pProb, pred],1)
                    pOneHot = tf.concat([pOneHot, aPredOneHot],1)

            
            aPredictionsProb = pProb
            aPredictionsOneHot = pOneHot
        
        aPredictionsProb = aPredictionsProb.numpy()
        aPredictionsOneHot = aPredictionsOneHot.numpy()
        
        return aPredictionsProb, aPredictionsOneHot