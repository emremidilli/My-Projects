import tensorflow as tf
import numpy as np

class PositionEncoder(tf.Module):
    def __init__(self, input_dim, output_dim ):
        super(PositionEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.pos_enc = self.get_pos_encoding_matrix(input_dim, output_dim)
        
        self.x_positions = tf.range(start=0, limit=input_dim, delta=1)
        
        self.position_embedding = tf.keras.layers.Embedding(
            input_dim= self.input_dim,
            output_dim=self.output_dim,
            weights=[self.pos_enc],
            name="position_embedding",
        )
        
    def __call__(self):
        y = self.position_embedding(self.x_positions)

        return y

        
    def get_pos_encoding_matrix(self, max_len,d_emb ):
        pos_enc = np.array(
            [
                [pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)]
                if pos != 0
                else np.zeros(d_emb)
                for pos in range(max_len)
            ]
        )
        pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])  # dim 2i
        pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])  # dim 2i+1
        return pos_enc

