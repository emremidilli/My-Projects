# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 14:26:30 2020

@author: Yunus Emre Midilli
"""


# https://www.tensorflow.org/tutorials/text/nmt_with_attention#translate
import tensorflow as tf
import numpy as np
import os
import time
import shutil


#hyperparameters
epoch_size = 2
batch_size = 112
number_of_hidden_neuron = 80
dropout_rate = 0.2
recurrent_dropout_rate = 0.2
optimizer = tf.keras.optimizers.Adam(learning_rate= 0.001, beta_1=0.7)
loss_function = tf.keras.losses.MeanAbsoluteError()
activation_function = 'tanh'

checkpoint_dir = './training_checkpoints'
checkpoints_location = "C:\\Users\\yunus\\Documents\\My Project\\training_checkpoints"
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")


feature_size_input = 0
feature_size_target = 0
backward_window_length = 0
forward_window_length = 0


class Encoder(tf.keras.Model):
  def __init__(self, one_hot_size, enc_units, batch_sz):
    super(Encoder, self).__init__()
    self.batch_sz = batch_sz
    self.enc_units = enc_units
    
    self.gru = tf.keras.layers.GRU(self.enc_units,
                                    return_sequences=True,
                                    return_state=True,
                                    recurrent_initializer='glorot_uniform',activation = activation_function,dropout = dropout_rate , recurrent_dropout= recurrent_dropout_rate)
    
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
  def __init__(self, one_hot_size, dec_units, batch_sz): 
    super(Decoder, self).__init__()
    self.batch_sz = batch_sz
    self.dec_units = dec_units

    self.gru = tf.keras.layers.GRU(self.dec_units,
                                    return_sequences=True, #We use return_sequences=True here because we'd like to access the complete encoded sequence rather than the final summary state.
                                    return_state=True,
                                    recurrent_initializer='glorot_uniform',activation = activation_function,dropout = dropout_rate , recurrent_dropout= recurrent_dropout_rate)
    self.fc = tf.keras.layers.Dense(one_hot_size)

    # used for attention
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


@tf.function
def train_step(inp, targ, enc_hidden, p_encoder, p_decoder):
  loss = 0

  with tf.GradientTape() as tape:
    enc_output, enc_hidden = p_encoder(inp, enc_hidden)
    dec_hidden = enc_hidden
    
    # start one-hot vector is the one hot vector of t.
    # dec_input = tf.expand_dims(targ[:, 0], 1) 
    dec_input = tf.expand_dims(np.zeros((batch_size,feature_size_target)) , 1)
    
    # Teacher forcing - feeding the target as the next input
    for t in range(1, targ.shape[1]):
      # passing enc_output to the decoder
      predictions, dec_hidden, _ = p_decoder(dec_input, dec_hidden, enc_output)

      loss += loss_function(targ[:, t], predictions)

      # using teacher forcing
      dec_input = tf.expand_dims(predictions, 1)

  batch_loss = (loss / int(targ.shape[1]))

  variables = p_encoder.trainable_variables + p_decoder.trainable_variables

  gradients = tape.gradient(loss, variables)

  optimizer.apply_gradients(zip(gradients, variables))

  return batch_loss


def attention_model(input_tensor_train, target_tensor_train, input_tensor_test):
    global batch_size
    global encoder
    global decoder
    
    steps_per_epoch = len(input_tensor_train)//batch_size
    buffer_size = len(input_tensor_train)
    
    dataset = create_dataset(input_tensor_train, target_tensor_train).shuffle(buffer_size)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    
    encoder = Encoder(feature_size_input, number_of_hidden_neuron, batch_size)
    decoder = Decoder(feature_size_target, number_of_hidden_neuron, batch_size)
    
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,encoder=encoder,decoder=decoder)
    
    for epoch in range(epoch_size):
        start = time.time()
        
        encoder_hidden = encoder.initialize_hidden_state()
        total_loss = 0
        
        for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):         
            # normalize
            inp = tf.reshape(inp, (batch_size, backward_window_length, feature_size_input, 1))
            targ = tf.reshape(targ, (batch_size, forward_window_length, feature_size_target, 1 )) #since forward window length includes also t.
            
            inp = tf.reshape(inp, (batch_size, backward_window_length, feature_size_input))
            targ = tf.reshape(targ, (batch_size, forward_window_length,feature_size_target))

            batch_loss = train_step(inp, targ, encoder_hidden, encoder, decoder)
            total_loss += batch_loss
            # denormalize

        # saving (checkpoint) the model every 2 epochs
        if (epoch + 1) % 2 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)
            
        print('Epoch {} Loss {:.4f}'.format(epoch + 1,total_loss / steps_per_epoch))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
        
        # restoring the latest checkpoint in checkpoint_dir
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    
    return attention_predict(input_tensor_test)
    

def attention_predict(input_tensor_test):
    global encoder
    global decoder

    batch_size_test = len(input_tensor_test)
        
    dataset_test = (tf.data.Dataset.from_tensor_slices((input_tensor_test)))
    
    dataset_test = dataset_test.batch(batch_size_test, drop_remainder=True)
    
    encoder_hidden = [tf.zeros((batch_size_test, number_of_hidden_neuron))]
    steps_per_epoch_test = len(input_tensor_test)//batch_size_test
    
    for (batch, (inp_test)) in enumerate(dataset_test.take(steps_per_epoch_test)):  
        inp_test = tf.reshape(inp_test, (batch_size_test, backward_window_length, feature_size_input, 1))
     
        inp_test = tf.reshape(inp_test, (batch_size_test, backward_window_length, feature_size_input))
        
        encoder_output, encoder_hidden = encoder(inp_test, encoder_hidden)
        decoder_hidden = encoder_hidden

        for t in range(0, feature_size_target*forward_window_length):
            if t == 0:
                decoder_input = tf.expand_dims(np.zeros((batch_size_test,feature_size_target)) , 1)
            else:
                decoder_input = tf.expand_dims(p, 1)

            # passing enc_output to the decoder
            p, dec_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_output)
            
            if t == 0:
                p2 = p
            else:
                p2 = tf.concat([p2, p],1)

        if batch == 0:
            predictions = p2
            
        else:
            predictions = tf.concat([predictions, p2],0)
       
    predictions = np.array(predictions)

    return predictions


def create_dataset(p_arr_input, p_arr_target):
    ds = (tf.data.Dataset.from_tensor_slices((p_arr_input,p_arr_target)))
    return ds
        

def main(scaled_input_train, scaled_target_train, scaled_input_test, feature_size_x, feature_size_y):
    try:
        shutil.rmtree(checkpoints_location)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))
        
    global feature_size_input
    global feature_size_target
    global backward_window_length
    global forward_window_length
        
    feature_size_input = feature_size_x
    feature_size_target = feature_size_y
    backward_window_length = int(scaled_input_train.shape[1]/feature_size_x)
    forward_window_length = int(scaled_target_train.shape[1]/feature_size_y)
    
    predicted_result = attention_model(scaled_input_train, scaled_target_train, scaled_input_test)
    
    return predicted_result
