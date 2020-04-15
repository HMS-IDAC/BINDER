import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, MaxPooling2D, AveragePooling2D,Dense,Flatten,BatchNormalization, Activation,Lambda,UpSampling2D, LeakyReLU, ReLU
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import ModelCheckpoint,TensorBoard
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers, regularizers, metrics, initializers
from losses import * 
 
def Autoencoder(input_size=(128,128,1),pretrained_weights=False,weights_path=None, lr = 1e-4): #size is batchXlengthXwidthXchannels
    """
    left and right branch input (anchor and same/different image)
    """
    Input_l=Input(shape=input_size,name='Input_l') # anchor
    Input_r=Input(shape=input_size,name='Input_r') # twin
    
    """ ENCODER E1/E2 """
    
    x=Sequential()
    x.add(Conv2D(32,(3,3),strides=(1,1),padding='same',name='l_1'))
    x.add(BatchNormalization())
    x.add(Activation('relu'))
    x.add(MaxPooling2D((2,2),padding='valid'))
    
    x.add(Conv2D(32,(3,3),strides=(1,1),padding='same',name='l_2'))
    x.add(BatchNormalization())
    x.add(Activation('relu'))
    x.add(MaxPooling2D((2,2),padding='valid'))
    
    x.add(Conv2D(32,(3,3),strides=(1,1),padding='same',name='r_3'))
    x.add(BatchNormalization())
    x.add(Activation('relu'))
    x.add(MaxPooling2D((2,2),padding='valid'))
    
    """output of encoder i.e. output embedding"""
    output_l=x(Input_l)
    output_r=x(Input_r)
    
    """ DECODER D1 & D2 """
    
    y=Sequential()
    y.add(Conv2D(32,(3,3),strides=(1,1),padding='same',name='r_8'))
    y.add(BatchNormalization())
    y.add(Activation('relu'))
    y.add(UpSampling2D((2,2)))
    
    y.add(Conv2D(32,(3,3),strides=(1,1),padding='same',name='l_3'))
    y.add(BatchNormalization())
    y.add(Activation('relu'))
    y.add(UpSampling2D((2,2)))
    
    y.add(Conv2D(32,(3,3),strides=(1,1),padding='same',name='l_4'))
    y.add(BatchNormalization())
    y.add(Activation('relu'))
    y.add(UpSampling2D((2,2)))
    
    y.add(Conv2D(1,(3,3),strides=(1,1),padding='same',name='l_5'))
    y.add(BatchNormalization())
    
    z=Sequential()
    z.add(Conv2D(32,(3,3),strides=(1,1),padding='same',name='r_11'))
    z.add(BatchNormalization())
    z.add(Activation('relu'))
    z.add(UpSampling2D((2,2)))
    
    z.add(Conv2D(32,(3,3),strides=(1,1),padding='same',name='l_6'))
    z.add(BatchNormalization())
    z.add(Activation('relu'))
    z.add(UpSampling2D((2,2)))
    
    z.add(Conv2D(32,(3,3),strides=(1,1),padding='same',name='l_7'))
    z.add(BatchNormalization())
    z.add(Activation('relu'))
    z.add(UpSampling2D((2,2)))
    
    z.add(Conv2D(1,(3,3),strides=(1,1),padding='same',name='l_8'))
    z.add(BatchNormalization())
    
    y_l=y(output_l)
    output_1=Activation('sigmoid',name='output_1')(y_l) # reconstructed from Input_l
    z_r=z(output_r)
    output_2=Activation('sigmoid',name='output_2')(z_r) # reconstrcted fromInput_r
    
    fc_l=Flatten()(output_l)
    fc_r=Flatten()(output_r)
    
    """calculate L2 norm distance, uncomment 84,85, 86 & 90 (and comment 88 & 91) to minimize (reconstrcution loss + distance between embedding)  """
    
    #abs_d = Lambda(lambda tensors: K.abs(K.l2_normalize(tensors[0]) - K.l2_normalize(tensors[1])), name='abs_d')([fc_l, fc_r])
    #l2_norm_d = Lambda(lambda tensors: K.sqrt(K.sum(K.square(tensors[0]),axis=1)), name='l2_norm_d',dtype='float')([abs_d])
    #model = Model([Input_l, Input_r],[output_1, output_2, l2_norm_d])
    
    model = Model([Input_l,Input_r],[output_1,output_2])
    Adam  =optimizers.Adam(lr = lr)
    #model.compile(optimizer=Adam,loss={'output_1':'mse','output_2':'mse','l2_norm_d':em_loss })
    model.compile(optimizer=Adam,loss={'output_1':'mse','output_2':'mse'})
    
    if pretrained_weights:
        model.load_weights(weights_path)
    
    return model


def Embedding(input_size=(128,128,1),pretrained_weights=False,weights_path=None): #size is batchXlengthXwidthXchannels
    """  Model with embedding as output i.e. ENCODER from Autoencoder
         left and right branch input (anchor and same/different image)
    """
    Input_l = Input(shape=input_size,name='Input_l') # anchor
    Input_r = Input(shape=input_size,name='Input_r') # twin
    
    x=Sequential()
    x.add(Conv2D(32,(3,3),strides=(1,1),padding='same',name='l_1'))
    x.add(BatchNormalization())
    x.add(Activation('relu'))
    x.add(MaxPooling2D((2,2),padding='valid'))
    
    x.add(Conv2D(32,(3,3),strides=(1,1),padding='same',name='l_2'))
    x.add(BatchNormalization())
    x.add(Activation('relu'))
    x.add(MaxPooling2D((2,2),padding='valid'))
    
    x.add(Conv2D(32,(3,3),strides=(1,1),padding='same',name='r_3'))
    x.add(BatchNormalization())
    x.add(Activation('relu'))
    x.add(MaxPooling2D((2,2),padding='valid'))
    
    output_l=x(Input_l)
    output_r=x(Input_r)
    """ Weights of the encoder are frozen and weights are loaded from the autoencoder model, to train set to True"""   
    model= Model([Input_l,Input_r],[output_l,output_r])
    model.trainable=False
    print('Model architecture (encoder)', x.summary())

    
    return model



# include metric learning by freezing model weights, and taking output of encoder = 16 x 16 x32  and pass throughh pooling (average pool and max pool) and dense layers
def Autoencoder_top(input_size=(128,128,1), pretrained_weights = False, weights_path = None,
                    batch_size = 64, pretrained_weights_base = False, base_weights = None, lr = 1e-4):
    """ 
         left and right branch input (anchor and same/different image)
    """
    Input_l = Input(shape=input_size,name='Input_l') # anchor
    Input_r = Input(shape=input_size,name='Input_r') # twin 
    
    weights_list = Autoencoder(pretrained_weights=pretrained_weights_base, weights_path = base_weights).get_weights() # get the weights of autoencoder pretrained on coco for 15 epochs (bs = 64)
    
    # load the pretrained weights into the encoder model that outputs the embedding    
    model_embedding = Embedding()
    model_embedding.set_weights(weights_list[0:18])
    embedding_out = model_embedding([Input_l,Input_r])
    output_embedding_l= embedding_out[0] # left
    output_embedding_r= embedding_out[1] # right
    
    # Maxpool and average pool (2,2) kernel ( 16 x 16 x 32  --> 8 x 8 x 32)
    #TODO: Generalized mean pooling
    max_l=MaxPooling2D((2,2),padding='valid')(output_embedding_l)
    max_r=MaxPooling2D((2,2),padding='valid')(output_embedding_r)
    avg_l=AveragePooling2D((2,2),padding='valid')(output_embedding_l)
    avg_r=AveragePooling2D((2,2),padding='valid')(output_embedding_r)
    flatmax_l=Flatten()(max_l) # (batchsize x 2048)
    flatmax_r=Flatten()(max_r)
    flatavg_l=Flatten()(avg_l)
    flatavg_r=Flatten()(avg_r)
    
    # concatenate average and max pool (batchsize x 4096)
    fc_l=Lambda(lambda tensors: K.concatenate([tensors[0], tensors[1]], axis = 1))([flatmax_l,flatavg_l])
    fc_r=Lambda(lambda tensors: K.concatenate([tensors[0], tensors[1]], axis = 1))([flatmax_r,flatavg_r])

    # Fully connected layers 
    dense_1=Dense(1024,activation='relu')
    dense1_l=dense_1(fc_l)
    dense1_r=dense_1(fc_r)
    dense_2=Dense(256,activation='linear')
    dense2_l=dense_2(dense1_l)
    dense2_r=dense_2(dense1_r)
    
    # output embeddings from left and right branch stacked
    dense_embedding = Lambda(lambda tensors: tf.stack([tensors[0],tensors[1]]), name='dense_embedding')([dense2_l,dense2_r])
    
    # output embeddings from left and right branch find l2 distance
    abs_d = Lambda(lambda tensors: K.abs(tf.math.l2_normalize(tensors[0],axis = 1) - tf.math.l2_normalize(tensors[1],axis = 1)), name='abs_d')([dense2_l, dense2_r])
    l2_norm_d = Lambda(lambda tensors: K.sqrt(K.sum(K.square(tensors[0]),axis=1) + K.epsilon()), name='l2_norm_d',dtype='float')([abs_d])
  
    # Instantiate the model
    model=Model([Input_l,Input_r],[dense_embedding, l2_norm_d])
    
    Adam=optimizers.Adam(lr=lr,clipnorm=1.)
     
    model.compile(optimizer=Adam,loss={'dense_embedding':triplet_hardestneg(batch_size)})
    #model.compile(optimizer=Adam,loss={'l2_norm_d':em_loss}) # to train using ranking loss
    
    if pretrained_weights==True:
        model.load_weights(weights_path)
    
    return model 

