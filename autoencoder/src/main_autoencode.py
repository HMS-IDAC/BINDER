import argparse
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from data_gen import *
from Autoencoder_models import *
import os

## set the GPU 
os.environ["CUDA_VISIBLE_DEVICES"]="0"


parser=argparse.ArgumentParser(description='Model Info.')

parser.add_argument('-t', '--t_path', dest = 'train_path', help = "Path to training data.")
parser.add_argument('-v', '--v_path', dest = 'val_path', help = "Path to validation data.")
parser.add_argument('-m', '--m_path', dest = 'model_path', help = "Path to saving model logs", default = './Model_logs/')
parser.add_argument('-bs', '--batch_size', dest = 'batch_size', help = "Batch size to use", default = 64)
parser.add_argument('-n_channels', '--number_channels', dest = 'n_channels', help = "Number of channels, = 3 for RGB; = 1 for gray", default = 1)
parser.add_argument('-ep', '--epochs', dest = 'epochs', help = "Number of Epochs", default = 15)
parser.add_argument('-lr', '--learning_rate', dest = 'lr', help = "learning rate", default = 1e-4)
parser.add_argument('--output_weights_path', dest = 'output_weights_path', help = 'path to save model weights', default='./weights/weights_AutoencoderModel.hdf5')
parser.add_argument('--input_weights_path', dest = 'input_weights_path', help = 'path to load model weights')
parser.add_argument('--model', dest = 'model', help = 'Model to load, = "Autoencoder"/"Autoencoder_top"', default = 'Autoencoder_top')
parser.add_argument('--dataset', dest = 'dataset', help = 'Dataset to load, = "COCO"/"BINDER"')

arguments = parser.parse_args()

if not arguments.train_path:   # if folder is not specified
	parser.error('Error: path to training data must be specified. Pass --t_path or -t to command line')

if not arguments.val_path:   # if folder is not specified
	parser.error('Error: path to validation data must be specified. Pass --v_path or -v to command line')
    
print('Setting path variables')

train_path = arguments.train_path
val_path = arguments.val_path
model_path = arguments.model_path
weights_path = arguments.output_weights_path
epochs = arguments.epochs
lr = arguments.lr
batch_size = int(arguments.batch_size)
n_channels = arguments.n_channels

if arguments.input_weights_path:   # if filename is given
	in_weights_path=arguments.input_weights_path

base_weights = './weights/Autoencoder_base_pretrained_COCO.hdf5' # Autoencoder pretrained on COCO to load into base encoder network
"""
Create Model, the base of the network is the Autoencoder pretrained on COCO, and top FC layers are pretrained on COCO followed by fine-tuning on BINDER

To train Autoencoder (base), --model = Autoencoder
To train top fully-connected layers, --model = Autoencoder_top, while the base layers are frozen with pretrained weights 

if pretrained model weights are defined in --in_weights_path, loads pretrained model

"""
if arguments.model == 'Autoencoder':   
    """load Autoencoder model to pretrain on COCO; else load pretrained Autoencoder; this is the bottom(base) network"""
    if arguments.input_weights_path: 
        Model = Autoencoder(pretrained_weights = True, weights_path = in_weights_path)
    else:
        Model = Autoencoder()
    prob_neg_pos = 0.5 # probability of negative and positive pair generation
    print( arguments.model=='Autoencoder')

elif arguments.model == 'Autoencoder_top': 
    """load Autoencoder_top model to  fine-tune on BINDER; else load pretrained Autoencoder_top"""
    if arguments.input_weights_path: # load pretrained model
        Model = Autoencoder_top(pretrained_weights = True, weights_path = in_weights_path, batch_size = batch_size)
    else:
        Model = Autoencoder_top(batch_size = batch_size, pretrained_weights_base = True, base_weights = base_weights)
    prob_neg_pos = 0 # probability of negative pair generation = 0 ; using triplet loss with hardest neg selection from within batch



## data generator to generate batches of train & validation sets  (from datagen_manipulate.py)
if arguments.dataset == 'COCO':   # if structure of dataset is image files in a folder
    train_generator = DataGenerator(train_path, batch_size = batch_size, n_channels = n_channels, prob_neg_pos = prob_neg_pos)
    val_generator = DataGenerator(val_path, batch_size = batch_size, n_channels = n_channels, prob_neg_pos = prob_neg_pos)
elif arguments.dataset == 'BINDER':  # if structure of dataset is images pairs in individual folders within a directory
    train_generator = DataGenerator(train_path,batch_size = batch_size,n_channels = n_channels, prob_neg_pos = prob_neg_pos)
    val_generator = DataGenerator(val_path, batch_size = batch_size, n_channels = n_channels, dataset = 'valid')

## Callback- save logs and vizualize on tensorboard
logdir = model_path + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir,histogram_freq=1,batch_size=batch_size,write_images=True)

print('Start training {} model'.format(arguments.model))

history_metric=Model.fit_generator(train_generator, steps_per_epoch = len(os.listdir(train_path))/batch_size,
                                           epochs = epochs, verbose = 1, validation_data=val_generator,
                                           validation_steps=len(os.listdir(val_path))/batch_size,
                                           callbacks = [ModelCheckpoint(weights_path, monitor = 'loss', verbose = 0, save_best_only = True, 
                                           save_weights_only = True), tensorboard_callback],
                                           max_queue_size = 32,workers = 8,use_multiprocessing = True)


