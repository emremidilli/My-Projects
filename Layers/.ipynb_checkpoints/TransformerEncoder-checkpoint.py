import tensorflow as tf


class TransformerEncoder(tf.Module):
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
        
        
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon =1e-6)
        self.multi_head_attn = tf.keras.layers.MultiHeadAttention(
            key_dim=self.key_dim,
            num_heads = self.num_heads,
            dropout = dropout
        )
        
        self.droput_layer = tf.keras.layers.Dropout(dropout)
        
        self.layer_norm_2 =  tf.keras.layers.LayerNormalization(epsilon = 1e-6)
        
        self.dense = tf.keras.layers.Conv1D(filters = self.ff_dim, kernel_size=  1, activation = 'relu')
        
        self.droput_layer_2 = tf.keras.layers.Dropout(dropout)
        
        self.dense_2 = tf.keras.layers.Conv1D(filters=input_dim, kernel_size = 1)
        
    def __call__(self, x):
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
        