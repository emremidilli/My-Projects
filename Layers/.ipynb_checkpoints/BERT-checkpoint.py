import tensorflow as tf
from PositionEncoder import PositionEncoder 
from TransformerEncoder import TransformerEncoder 


class BERT(tf.keras.Model):
    def __init__(
        self, 
        mlm_input_shape, 
        nsp_input_shape,
        nr_of_encoder_blocks,
        attention_key_dims,
        attention_nr_of_heads,
        attention_dense_dims,
        dropout_rate
    ):
        super(BERT, self).__init__()
 
        self.mlm_input_shape = mlm_input_shape
        self.nsp_input_shape = nsp_input_shape
        self.nr_of_encoder_blocks = nr_of_encoder_blocks
        self.attention_key_dims= attention_key_dims
        self.attention_nr_of_heads= attention_nr_of_heads
        self.attention_dense_dims = attention_dense_dims
        self.dropout_rate= dropout_rate
        
#         self.loss_tracker = tf.keras.metrics.Mean(name="loss")
#         self.mlm_loss_tracker = tf.keras.metrics.Mean(name="mlm_loss")
#         self.nsp_loss_tracker = tf.keras.metrics.Mean(name="nsp_loss")
        
        
        self.mlm_input = tf.keras.layers.Input(mlm_input_shape, name = 'mlm_input')
        self.nsp_input = tf.keras.layers.Input(nsp_input_shape, name = 'nsp_input')
        
        self.concatenate_inputs = tf.keras.layers.Concatenate(axis = 1, name = 'concatenate_inputs')
        
        # self.position_embedding = PositionEncoder(
        #     input_dim = mlm_input_shape[0] + nsp_input_shape[0] , 
        #     output_dim = mlm_input_shape[1]
        # )
        
        self.transformer_encoders = []
        
        for i in range(nr_of_encoder_blocks):
            self.transformer_encoders.append(TransformerEncoder(
                attention_key_dims, 
                attention_nr_of_heads, 
                attention_dense_dims,
                dropout_rate,
                mlm_input_shape[-1]
            ))
            
        self.mlm_output = tf.keras.layers.Dense(mlm_input_shape[1], activation = 'softmax', name = 'mlm_classifier')
        
        self.nsp_output = tf.keras.layers.Dense(1, activation = 'sigmoid', name = 'nsp_classifier')


        
#     def compute_loss(self, x, y, y_pred, sample_weight):
        
#         y_mlm = y[0]
#         y_nsp = y[1]
        
#         y_pred_mlm = y_pred[0]
#         y_pred_nsp = y_pred[1]
        
#         mlm_loss = mlm_custom_loss(y_mlm, y_pred_mlm) 
#         nsp_loss = nsp_custom_loss(y_nsp,y_pred_nsp)  
#         loss =mlm_loss +  nsp_loss
        
#         self.loss_tracker.update_state(loss)
#         self.mlm_loss_tracker.update_state(mlm_loss)
#         self.nsp_loss_tracker.update_state(nsp_loss)
#         return loss
    
#     def reset_metrics(self):
#         self.loss_tracker.reset_states()
#         self.mlm_loss_tracker.reset_states()
#         self.nsp_loss_tracker.reset_states()
        
#     @property
#     def metrics(self):
#         return [self.loss_tracker, self.mlm_loss_tracker, self.nsp_loss_tracker]
        
#         # Get output layer with `call` method
#         self.out = self.call([self.mlm_input, self.nsp_input])

#         # Reinitial
#         super(BERT, self).__init__(
#             inputs=[self.mlm_input, self.nsp_input],
#             outputs=self.out)

        
#     def train_step(self, data):
#         x, y = data
        
#         y_mlm = y[0]
#         y_nsp = y[1]
#         with tf.GradientTape() as tape:
#             y_pred = self(x, training=True)  # Forward pass
            
#             y_pred_mlm = y_pred[0]
#             y_pred_nsp = y_pred[1]
#             # Compute our own loss
#             mlm_loss = mlm_custom_loss(y_mlm, y_pred_mlm) 
#             nsp_loss = nsp_custom_loss(y_nsp,y_pred_nsp)  
#             loss =mlm_loss +  nsp_loss
            

#         # Compute gradients
#         trainable_vars = self.trainable_variables
#         gradients = tape.gradient(loss, trainable_vars)

#         # Update weights
#         self.optimizer.apply_gradients(zip(gradients, trainable_vars))

#         # Compute our own metrics
#         self.loss_tracker.update_state(loss)
#         self.mlm_loss_tracker.update_state(mlm_loss)
#         self.nsp_loss_tracker.update_state(nsp_loss)
#         return {"loss": self.loss_tracker.result(), "mlm_loss": self.mlm_loss_tracker.result(), "nsp_loss": self.nsp_loss_tracker.result()}


    def summary(self):
        x = [self.mlm_input, self.nsp_input]
        model = tf.keras.Model(
            inputs=x,
            outputs=self.call(x)
        )
        return model.summary()


        
     
    def call(self, inputs, training = True):
        mlm_input = inputs[0]
        nsp_input = inputs[1]
        
        x = self.concatenate_inputs([mlm_input, nsp_input])
        
#         x_position_embeddings = self.position_embedding()
        
#         x = tf.add(x, x_position_embeddings)
        
        for oEncoder in self.transformer_encoders:
            x = oEncoder(x)
            
        mlm_output = self.mlm_output(x)
        
        x = tf.keras.layers.Flatten()(x)
    
        nsp_output = self.nsp_output(x)
        
        return [mlm_output, nsp_output]

    
    
    def get_config(self):
        
        return {
            'mlm_input_shape': self.mlm_input_shape, 
            'nsp_input_shape': self.nsp_input_shape,
            'nr_of_encoder_blocks':self.nr_of_encoder_blocks, 
            'attention_key_dims':self.attention_key_dims ,
            'attention_nr_of_heads':self.attention_nr_of_heads,
            'attention_dense_dims':self.attention_dense_dims,
            'dropout_rate':self.dropout_rate
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    
def mlm_custom_loss( y_true, y_pred):
    # y_true for mlm is without cls token. So we should ignore the cls_token time step of y_pred.
    # since cls token is the latest latest token of prediction, we ignore that token.
    y_pred_without_special_tokens = y_pred[:, :-1, :]
    non_masks = tf.not_equal(y_true, tf.cast(tf.ones(y_true.shape[2])*-1, tf.dtypes.float32))[:,:,0]

    y_pred_non_mask = tf.boolean_mask(y_pred_without_special_tokens,non_masks)
    y_true_non_mask = tf.boolean_mask(y_true,non_masks)

    oBinCE = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

    loss = oBinCE(y_true_non_mask , y_pred_non_mask)
    loss = tf.reduce_mean(loss)

    return loss

def nsp_custom_loss( y_true, y_pred):
    oBinCE = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

    loss = oBinCE(y_true , y_pred)
    loss = tf.reduce_mean(loss)

    return  loss

