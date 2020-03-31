import tensorflow as tf
from tensorflow.keras import backend as K

def triplet_hardestneg(batch_size=64):
    """ Computes triplet loss (eq.1 in "On Identification and Retrieval of Near-Duplicate Biological Images: a New Dataset and Baseline")
        with hardest negative selection from within the batch
    """
    def triplet_loss(y_true,y_pred):
        """ y_pred = output of dense_embedding layer in models.py"""
        dense_1=y_pred[0] # from dense_embedding (dense layer from left branch)
        dense_2=y_pred[1] # (dense layer from right branch)
        
        # Compute L2 distance matrix

        x = tf.matmul(tf.expand_dims(tf.reduce_sum(tf.square(tf.math.l2_normalize(dense_1,axis=1)), 1), 1),
                       tf.ones(shape=(1, batch_size))) # compute sum of squares of l2 norm of dense_1
        y = tf.transpose(tf.matmul(tf.reshape(tf.reduce_sum(tf.square(tf.math.l2_normalize(dense_2, axis=1)), 1), 
                        shape=[-1, 1]),tf.ones(shape=(batch_size, 1)),transpose_b=True)) # compute sum of squares of l2 norm of dense_2
        l2_matrix = tf.sqrt(tf.math.maximum(tf.constant( K.epsilon(),dtype='float'),tf.add(x, y)-2*tf.matmul(tf.math.l2_normalize(dense_1,axis=1),
                            tf.math.l2_normalize(dense_2,axis=1),transpose_b= True) + tf.constant(K.epsilon())))
        """ Set diagonal elements to large values"""
        l2_batch = tf.linalg.set_diag(l2_matrix,tf.linalg.tensor_diag_part(l2_matrix) + tf.constant(10, dtype='float'))
        """ Select hardest negative examples from within the batch i.e. closest different image pairs"""
        l2_batch_min = tf.reduce_min([tf.reduce_min(l2_batch,axis=0), tf.reduce_min(l2_batch,axis=1)],axis=0)
        """Compute triplet loss"""
        loss = tf.math.maximum(tf.constant(0 + K.epsilon(),dtype='float'),tf.linalg.tensor_diag_part(l2_matrix) + tf.constant(1,dtype='float') - l2_batch_min)

        return loss

    return triplet_loss


def em_loss(ytrue,y_pred):
    """ Ranking loss where positive and negative pairs are randomly seleted, y_pred = output of l2_norm_d layer in models.py"""
    loss=(ytrue*(y_pred)) + ((1-ytrue)*((K.maximum(K.constant([0]),K.constant([2])-y_pred))))
    
    return loss

