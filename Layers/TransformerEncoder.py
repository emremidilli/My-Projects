import tensorflow as tf


class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(
        self, 
        key_dim, 
        num_heads, 
        ff_dim, 
        dropout,
        input_dim
    ):
        super(TransformerEncoder, self).__init__()
        self.key_dim = key_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.input_dim = input_dim
        
        
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon =1e-6, name = 'trans_enc_attn_layer_norm')
        self.multi_head_attn = tf.keras.layers.MultiHeadAttention(
            key_dim=self.key_dim,
            num_heads = self.num_heads,
            dropout = dropout,
            name = 'trans_enc_attn_mha'
        )
        
        self.droput_layer = tf.keras.layers.Dropout(dropout, name = 'trans_enc_attn_dropout')
        
        self.layer_norm_2 =  tf.keras.layers.LayerNormalization(epsilon = 1e-6, name = 'trans_enc_res_layer_norm')
        
        self.dense = tf.keras.layers.Conv1D(
            filters = self.ff_dim, 
            kernel_size=  1, 
            activation = 'relu',
            name = 'trans_enc_res_dense'
        )
        
        self.droput_layer_2 = tf.keras.layers.Dropout(
            dropout,
            name = 'trans_enc_res_dropout'
        )
        
        self.dense_2 = tf.keras.layers.Conv1D(
            filters=input_dim, 
            kernel_size = 1,
            name = 'trans_enc_res_dense_out'
            
        )
        
    def call(self, x):
        # Normalization and Attention
        y = self.layer_norm(x)
        
        y = self.multi_head_attn(y,y) # self attention
        
        y = self.droput_layer(y)
        
        res = y + x
        
        # Feed Forward Part
        y = self.layer_norm_2(res)
        y = self.dense(y)
        y = self.droput_layer_2 (y)
        y = self.dense_2(y)
        
        return y + res
    
    def get_config(self):
        return {
            'key_dim': self.key_dim, 
            'num_heads': self.num_heads,
            'ff_dim':self.ff_dim, 
            'dropout':self.dropout ,
            'input_dim':self.input_dim
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)
        