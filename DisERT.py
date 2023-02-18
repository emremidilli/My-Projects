import tensorflow as tf


# class PositionEncoder(tf.Module):
#     def __init__(self, input_dim, output_dim ):
#         super(PositionEncoder, self).__init__()
        
#         self.input_dim = input_dim
#         self.output_dim = output_dim
        
#         self.pos_enc = self.get_pos_encoding_matrix(input_dim, output_dim)
        
        
#     def __call__(self, x):
#         y = tf.keras.layers.Embedding(
#             input_dim= self.input_dim,
#             output_dim=self.output_dim,
#             weights=[self.pos_enc],
#             name="position_embedding",
#         )(x)
        
#         return y

        
#     def get_pos_encoding_matrix(self, max_len,d_emb ):
#         pos_enc = np.array(
#             [
#                 [pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)]
#                 if pos != 0
#                 else np.zeros(d_emb)
#                 for pos in range(max_len)
#             ]
#         )
#         pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])  # dim 2i
#         pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])  # dim 2i+1
#         return pos_enc



# class TransformerEncoder(tf.Module):
#     def __init__(self, key_dim, num_heads, ff_dim, dropout):
#         super(TransformerEncoder, self).__init__()
#         self.key_dim = key_dim
#         self.num_heads = num_heads
#         self.ff_dim = ff_dim
#         self.dropout = dropout
        
        
#     def __call__(self, x):
#         # Normalization and Attention
#         y = tf.keras.layers.LayerNormalization(epsilon =1e-6)(x)
        
#         y = tf.keras.layers.MultiHeadAttention(
#             key_dim=self.key_dim,
#             num_heads = self.num_heads,
#             dropout = self.dropout
#         )(y,y) # self attention
        
        
#         y = tf.keras.layers.Dropout(self.dropout)(y)
#         res = y + x
        
#         # Feed Forward Part
#         y = tf.keras.layers.LayerNormalization(epsilon = 1e-6)(res)
#         y = tf.keras.layers.Conv1D(filters = self.ff_dim, kernel_size=  1, activation = 'relu')(y)
#         y = tf.keras.layers.Dropout(self.dropout)(y)
#         y = tf.keras.layers.Conv1D(filters=x.shape[-1], kernel_size = 1)(y)
        
#         return y + res
        

# mlm_input = tf.keras.layers.Input((X_MLM.shape[1], X_MLM.shape[2]), name = 'mlm_input')
# nsp_input = tf.keras.layers.Input((X_NSP.shape[1], X_NSP.shape[2]), name = 'nsp_input')

# x = tf.keras.layers.Concatenate(axis = 1)([mlm_input, nsp_input])

# x_positions = (tf.range(start=0, limit=x.shape[1], delta=1))
# x_position_embeddings = PositionEncoder(
#     input_dim = x.shape[1] , 
#     output_dim = x.shape[2]
# )(x_positions)
# x = tf.add(x, x_position_embeddings)

# for i in range(NR_OF_ENCODER_BLOCKS):
#     x = TransformerEncoder(ATTENTION_KEY_DIMS, ATTENTION_NR_OF_HEADS, ENCODER_DENSE_DIMS, DROPOUT_RATE)(x)

# mlm_output = tf.keras.layers.Dense(X_MLM.shape[2], activation = 'softmax', name = 'mlm_classifier')(x)

# x = tf.keras.layers.Flatten()(x)
# nsp_output = tf.keras.layers.Dense(1, activation = 'sigmoid', name = 'nsp_classifier')(x)

# DisERT_model = tf.keras.Model(inputs = [mlm_input, nsp_input], outputs= [mlm_output, nsp_output])

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

def nsp__custom_loss(y_true, y_pred):
    oBinCE = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    
    loss = oBinCE(y_true , y_pred)
    loss = tf.reduce_mean(loss)
    
    return  loss
        
# DisERT_model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
#     loss = [mlm_custom_loss, nsp__custom_loss]
# )

# print(DisERT_model.summary())
# tf.keras.utils.plot_model(DisERT_model, show_shapes=True)