# https://www.tensorflow.org/tutorials/text/nmt_with_attention#translate

import numpy as np
import os
import tensorflow as tf


class Encoder(tf.keras.Model):
    def __init__(self, one_hot_size, enc_units, batch_sz, activation_function, dropout_rate, recurrent_dropout_rate):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                        return_sequences=True,
                                        return_state=True,
                                        recurrent_initializer='glorot_uniform',activation = activation_function, dropout = dropout_rate , recurrent_dropout= recurrent_dropout_rate)

    def call(self, x, hidden): #incoming X must be a matrix since embedding layer is cancelled. Matrix shape is (backward_window_length, backward_feature_size) 
      output, state = self.gru(x, initial_state = hidden)
      return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)
    
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
    def __init__(self, one_hot_size, dec_units, batch_sz, activation_function, dropout_rate, recurrent_dropout_rate): 
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
    
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                        return_sequences=True, #We use return_sequences=True here because we'd like to access the complete encoded sequence rather than the final summary state.
                                        return_state=True,
                                        recurrent_initializer='glorot_uniform',activation = activation_function,dropout = dropout_rate , recurrent_dropout= recurrent_dropout_rate)
        
        self.fc = tf.keras.layers.Dense(one_hot_size)
    
        self.attention = BahdanauAttention(self.dec_units)
    
    def call(self, x, hidden, enc_output): # dec_input, dec_hidden, enc_output
    
        # enc_output shape == (batch_size, max_length, hidden_size)
        context_vector, attention_weights = self.attention(hidden, enc_output)
    
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
    
        # passing the concatenated vector to the GRU
        output, state = self.gru(x)
        
        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))
    
        # output shape == (batch_size, vocab)
        x = self.fc(output)
    
        return x, state, attention_weights


class Neural_Attention_Mechanism(tf.keras.Model):
    def __init__(self, model_id,feature_size_x , feature_size_y, window_length_x,window_length_y): 
        super(Neural_Attention_Mechanism, self).__init__()
        self.model_id = model_id
        
        self.model_directory = os.path.join(self.model_id, "__model__")

        self.feature_size_input = feature_size_x
        self.feature_size_target = feature_size_y
        self.backward_window_length = window_length_x
        self.forward_window_length = window_length_y
        
        self.set_hyperparameters()
        
    
    def set_hyperparameters(self,epoch_size = 10, batch_size = 112, number_of_hidden_neuron = None, dropout_rate_encoder = 0.1,dropout_rate_decoder=0.1, recurrent_dropout_rate_encoder = 0.1, recurrent_dropout_rate_decoder=0.1, learning_rate = 0.001, momentum_rate=0.1):
        self.epoch_size = epoch_size
        self.batch_size = batch_size
        if number_of_hidden_neuron is None:
            number_of_hidden_neuron = self.feature_size_input*self.backward_window_length*2
            
        self.number_of_hidden_neuron = number_of_hidden_neuron
        self.dropout_rate_encoder = dropout_rate_encoder
        self.dropout_rate_decoder = dropout_rate_decoder
        self.recurrent_dropout_rate_encoder = recurrent_dropout_rate_encoder
        self.recurrent_dropout_rate_decoder = recurrent_dropout_rate_decoder
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate= self.learning_rate, beta_1=self.momentum_rate)
        self.loss_function = tf.keras.losses.MeanAbsoluteError()
        self.activation_function = 'tanh'
        
        self.encoder = Encoder(self.feature_size_input, self.number_of_hidden_neuron, self.batch_size,self.activation_function, self.dropout_rate_encoder, self.recurrent_dropout_rate_encoder)
        self.decoder = Decoder(self.feature_size_target, self.number_of_hidden_neuron, self.batch_size, self.activation_function, self.dropout_rate_decoder, self.recurrent_dropout_rate_decoder)
    
    
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
          
        batch_loss = (loss / int(targ.shape[1]))
        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return batch_loss

    
    def train(self, input_tensor_train, target_tensor_train):        
        checkpoint_dir = os.path.join(self.model_id, "__training checkpoints__")
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")    
        checkpoint = tf.train.Checkpoint(optimizer=self.optimizer,encoder=self.encoder,decoder=self.decoder)
        
        steps_per_epoch = len(input_tensor_train)//self.batch_size
        buffer_size = len(input_tensor_train)
        
        dataset = (tf.data.Dataset.from_tensor_slices((input_tensor_train,target_tensor_train)))
        dataset = dataset.shuffle(buffer_size)
        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        
        for epoch in range(self.epoch_size):
            encoder_hidden = self.encoder.initialize_hidden_state()
            total_loss = 0.0
                    
            for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
                inp = tf.reshape(inp, (self.batch_size, self.backward_window_length, self.feature_size_input, 1))
                targ = tf.reshape(targ, (self.batch_size, self.forward_window_length, self.feature_size_target, 1 )) #since forward window length includes also t.
                
                inp = tf.reshape(inp, (self.batch_size, self.backward_window_length, self.feature_size_input))
                targ = tf.reshape(targ, (self.batch_size, self.forward_window_length,self.feature_size_target))
                
                batch_loss = self.train_step(inp, targ, encoder_hidden)
                
                total_loss += batch_loss
                
            if (epoch + 1) % 2 == 0 or epoch == 0:
                checkpoint.write(file_prefix = checkpoint_prefix)
            
            checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        
        self.save_weights(self.model_directory)
        
    
    def predict(self, input_tensor_test):
        self.load_weights(self.model_directory)
        
        batch_size_test = len(input_tensor_test)
        
        dataset_test = (tf.data.Dataset.from_tensor_slices((input_tensor_test)))
        
        dataset_test = dataset_test.batch(batch_size_test, drop_remainder=True)
        
        encoder_hidden = tf.zeros((batch_size_test, self.number_of_hidden_neuron))
    
        predictions = tf.zeros(1)
        for (batch, (inp_test)) in enumerate(dataset_test.take(-1)):
            inp_test = tf.reshape(inp_test, (batch_size_test, self.backward_window_length, self.feature_size_input, 1))
            inp_test = tf.reshape(inp_test, (batch_size_test, self.backward_window_length, self.feature_size_input))
    
            encoder_output, encoder_hidden = self.encoder(inp_test, encoder_hidden)
            
            decoder_hidden = encoder_hidden
            
            decoder_input = tf.expand_dims(np.zeros((batch_size_test,self.feature_size_target)) , 1)
            
            p2 = tf.zeros(1)
            for t in range(0, self.forward_window_length):
                pred, dec_hidden, _ = self.decoder(decoder_input, decoder_hidden, encoder_output) # passing enc_output to the decoder
                decoder_input = tf.expand_dims(pred, 1)
                if t == 0:
                    p2 = pred
                else:
                    p2 = tf.concat([p2, pred],1)
            
            predictions = p2
        
        
        
        return predictions