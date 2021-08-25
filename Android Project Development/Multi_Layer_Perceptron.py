import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers




class Multi_Layer_Perceptron(Sequential):
    def __init__(self, model_id,feature_size_x , feature_size_y, window_length_x,window_length_y): 
        super(Multi_Layer_Perceptron, self).__init__()
        self.model_id = model_id
        
        self.model_directory = os.path.join(self.model_id, "__model__")

        self.feature_size_input = feature_size_x
        self.feature_size_target = feature_size_y
        self.backward_window_length = window_length_x
        self.forward_window_length = window_length_y
        
        self.set_hyperparameters()
        
    
    def set_hyperparameters(self,epoch_size = 1000, batch_size = 128, number_of_hidden_neuron = None,  learning_rate = 0.001, momentum_rate=0.9):
        self.epoch_size = epoch_size
        self.batch_size = batch_size
        if number_of_hidden_neuron is None:
            number_of_hidden_neuron = self.feature_size_input*self.backward_window_length*2
            
        self.number_of_hidden_neuron = number_of_hidden_neuron

        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate= self.learning_rate, beta_1=self.momentum_rate)
        self.loss_function = tf.keras.losses.MeanSquaredError()
        self.activation_function = 'sigmoid'
        
        oKernelRegulizer = regularizers.l2(0.001)
        
        self.add(Dense(self.number_of_hidden_neuron, activation = self.activation_function, kernel_regularizer=oKernelRegulizer ,input_shape = (self.backward_window_length * self.feature_size_input,)))
        self.add(Dense(self.number_of_hidden_neuron, activation = self.activation_function, kernel_regularizer=oKernelRegulizer))
        self.add(Dense(self.number_of_hidden_neuron, activation = self.activation_function, kernel_regularizer=oKernelRegulizer))
        self.add(Dense((self.forward_window_length * self.feature_size_target), kernel_regularizer=oKernelRegulizer))

        self.compile(optimizer=self.optimizer, loss=self.loss_function)

    
    def train(self, aInputTrain, aOutputTrain, aInputValidation, aOutputValidation):        
        checkpoint_dir = os.path.join(self.model_id, "__training checkpoints__")
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")    
        checkpoint = tf.train.Checkpoint(optimizer=self.optimizer,modelMlp = self)
        
        oEarlyStop = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 0 , patience = 15)
        
        self.fit(aInputTrain, aOutputTrain, epochs=self.epoch_size, batch_size=self.batch_size, verbose=0, validation_data= (aInputValidation, aOutputValidation), callbacks=[oEarlyStop] )
            
        checkpoint.write(file_prefix = checkpoint_prefix)
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        self.save_weights(self.model_directory)
        
    
    def aPredict(self, aInputTest):
        self.load_weights(self.model_directory)
        
        aPredictions = self.predict(aInputTest)
        
        return aPredictions