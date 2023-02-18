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
        
        self.mlm_input = tf.keras.layers.Input(mlm_input_shape, name = 'mlm_input')
        self.nsp_input = tf.keras.layers.Input(nsp_input_shape, name = 'nsp_input')
        
        self.concatenate_inputs = tf.keras.layers.Concatenate(axis = 1, name = 'concatenate_inputs')
        
        self.position_embedding = PositionEncoder(
            input_dim = mlm_input_shape[0] + nsp_input_shape[0] , 
            output_dim = mlm_input_shape[1]
        )
        
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
        
        
        # Get output layer with `call` method
        self.out = self.call([self.mlm_input, self.nsp_input])

        # Reinitial
        super(BERT, self).__init__(
            inputs=[self.mlm_input, self.nsp_input],
            outputs=self.out)
        

    def build(self):
        # Initialize the graph
        self._is_graph_network = True
        self._init_graph_network(
            inputs=[self.mlm_input, self.nsp_input],
            outputs=[self.out]
        )
        
        
     
    def call(self, inputs, training = True):
        mlm_input = inputs[0]
        nsp_input = inputs[1]
        
        x = self.concatenate_inputs([mlm_input, nsp_input])
        
        x_position_embeddings = self.position_embedding()
        
        x = tf.add(x, x_position_embeddings)
        
        for oEncoder in self.transformer_encoders:
            x = oEncoder(x)
            
        mlm_output = self.mlm_output(x)
        
        x = tf.keras.layers.Flatten()(x)
    
        nsp_output = self.nsp_output(x)
        
        
        return [mlm_output, nsp_output]



def mlm_custom_loss(y_true, y_pred):
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

def nsp_custom_loss(y_true, y_pred):
    oBinCE = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    
    loss = oBinCE(y_true , y_pred)
    loss = tf.reduce_mean(loss)
    
    return  loss
        
