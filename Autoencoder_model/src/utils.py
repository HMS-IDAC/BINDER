from sklearn.metrics import precision_recall_curve
import sys
import tensorflow as tf
from tensorflow.keras import backend as K

## get precision at recall
def f_acc_ROC(ytrue,dist):

    # calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(ytrue, 1/(dist+sys.float_info.epsilon))
    #precision, recall, thresholds = precision_recall_curve(ytrue, dist)
    rec_90=lambda recall : recall >= 0.90
    pres_rec90=precision[rec_90(recall)][-1]
    thresh_90=thresholds[rec_90(recall[:-1])][-1]
   
    rec_95=lambda recall : recall >= 0.95
    pres_rec95=precision[rec_95(recall)][-1]
    thresh_95=thresholds[rec_95(recall[:-1])][-1]
   
    rec_99=lambda recall : recall >= 0.99
    pres_rec99=precision[rec_99(recall)][-1]
    thresh_99=thresholds[rec_99(recall[:-1])][-1]
   
    print(' Recall = 90 % \n', 'Precision = ' + str(pres_rec90*100 )+'% \n', 'Threshold =' + str(1/(thresh_90+sys.float_info.epsilon)))
    print(' Recall = 95 % \n', 'Precision = ' + str(pres_rec95*100 )+'% \n', 'Threshold =' + str(1/(thresh_95+sys.float_info.epsilon)))
    print(' Recall = 99 % \n', 'Precision = ' + str(pres_rec99*100 )+'% \n', 'Threshold =' + str(1/(thresh_99+sys.float_info.epsilon)))
    return(precision,recall,thresholds)

## get recall at precisions

def f_acc_pres_recall(ytrue,dist):

    # calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(ytrue, 1/(dist+sys.float_info.epsilon))
    #precision, recall, thresholds = precision_recall_curve(ytrue, dist)
    rec_90=lambda precision : precision >= 0.90
    pres_rec90=recall[rec_90(precision)][0]
    thresh_90=thresholds[rec_90(precision[:-1])][0]
   
    rec_95=lambda precision : precision >= 0.95
    pres_rec95=recall[rec_95(precision)][0]
    thresh_95=thresholds[rec_95(precision[:-1])][0]
   
    rec_99=lambda precision : precision >= 0.99
    pres_rec99=recall[rec_99(precision)][0]
    thresh_99=thresholds[rec_99(precision[:-1])][0]
   
    print(' Precision = 90 % \n', 'Recall = ' + str(pres_rec90*100 )+'% \n', 'Threshold =' + str(1/(thresh_90+sys.float_info.epsilon)))
    print(' Precision = 95 % \n', 'Recall = ' + str(pres_rec95*100 )+'% \n', 'Threshold =' + str(1/(thresh_95+sys.float_info.epsilon)))
    print(' Precision = 99 % \n', 'Recall = ' + str(pres_rec99*100 )+'% \n', 'Threshold =' + str(1/(thresh_99+sys.float_info.epsilon)))
    return(precision,recall,thresholds)

def l2dist_matrix(y_true,y_pred):
        """ compute l2 normalized euclidean distance matrix""" 
        dense_1=y_pred[0]
        dense_2=y_pred[1]

        x = tf.matmul(tf.expand_dims(tf.reduce_sum(tf.square(tf.math.l2_normalize(dense_1,axis=1)), 1), 1),
                       tf.ones(shape=(1, dense_1.shape[0]))) # compute sum of sqaures of l2 norm of dense_1
        y = tf.transpose(tf.matmul(tf.reshape(tf.reduce_sum(tf.square(tf.math.l2_normalize(dense_2, axis=1)), 1), 
                        shape=[-1, 1]),tf.ones(shape=(dense_1.shape[0], 1)),transpose_b=True))
        
        l2_matrix = tf.sqrt(tf.math.maximum(tf.constant( K.epsilon(),dtype='float'),tf.add(x, y)-2*tf.matmul(tf.math.l2_normalize(dense_1,axis=1),
                            tf.math.l2_normalize(dense_2,axis=1),transpose_b= True) + tf.constant(K.epsilon())))
        l2_batch = tf.linalg.set_diag(l2_matrix,tf.linalg.tensor_diag_part(l2_matrix) + tf.constant(10, dtype='float'))
        l2_batch_min = tf.reduce_min([tf.reduce_min(l2_batch,axis=0), tf.reduce_min(l2_batch,axis=1)],axis=0)
        
        return l2_matrix, l2_batch_min
