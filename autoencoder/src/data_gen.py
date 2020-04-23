#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 10:31:00 2020

@author: spandana
"""

import os
import numpy as np
import cv2
from matplotlib import image as mpimg
import matplotlib.pyplot as plt
from skimage import transform
from skimage import exposure
from PIL import Image
import warnings
import tensorflow
import random
from manipulate import Manipulator
from PIL import Image as pil

"""
Image Augment class and functions to apply transformations 
"""

def imcrop(im, imcropsize=128):
    """resize to 256x256, then crop"""
    im = cv2.resize(im, (imcropsize*2, imcropsize*2), interpolation=cv2.INTER_CUBIC)
    row0 = int(imcropsize / 2)  # <- int((imcropsize*2 - imcropsize) / 2)
    col0 = row0
    return im[row0:row0 + imcropsize, col0:col0 + imcropsize]

def img_resize(im, imresize=256):
    """resize image"""
    im = cv2.resize(im, (imresize, imresize), interpolation=cv2.INTER_CUBIC)
    return im

def img_convert(im): 
    """convert images to gray if RGB (take average of 3 channels) otherwise pass, so (h,w,c) -> (h,w)"""
    if len(im.shape) == 3:
        return np.mean(im,axis=2)
    elif len(im.shape) == 2:
        return im
    else:
        raise TypeError('Length of Image size cannot be {}'.format(len(im.shape)))
        
      
# MAnipulator for train set -real time synthetic manipulations generator 
""" Applies Synthetic manipulations, Generates same & same-source manipulated image pairs"""
X = Manipulator('./src/manipulations.yaml')

## Sequence genertor (use multiprocessing=True and multiple threads for faster GPU computation)
class DataGenerator(tensorflow.keras.utils.Sequence):
    
    """
    Generates positive and negative pairs if prob_neg_pos > 0, 
    else geneartes only positve pairs
       
       Input - path to directory with images
       Ouput - Batch of Image pairs and labels: 
               1 = 'same'(postive pair) | 0 = 'different' (negative_pair) 
       
       returns - {dict1- input to model} , {dict2- Ground Truth for supervised-learning}  
       dict1:
           'Input_l' - anchor image
           'Input_r' - same src manipulated or random different img
       dict2:
           'output_1' - same as 'Input_l' (for training autoencoder base network)
           'output_2' - same as 'Input_r' (for training autoencoder base network)
           'l2_norm_d' - batch labels (0 or 1 for different or same respectively)
           'dense_embedding - batch labels (0 or 1 for different or same respectively)
       
       'Initialization'
           dir_path: path to directory
           batch_size: batch size
           input_size: image size (e.g. = 128, i.e. height or width b/c aspect ratio =1) 
           n_channels: number of channels (= 1 or 3) 
           prob_neg_pos:probability of negative samples [0,1]
           dataset: 'train' / 'valid'
                    if directory has img files choose 'train', 
                    if directory has subfolders with img files choose'valid'
                  
    """
    
    def __init__(self, dir_path, batch_size=1, input_size=128, n_channels=1, prob_neg_pos=0.5, dataset='train',
                 shuffle=True,seed_img=None):
        'Initialization'
        self.input_size = input_size
        self.batch_size = batch_size
        self.dir_path = dir_path
        self.list_IDs = os.listdir(self.dir_path)
        self.n_channels = n_channels
        self.prob_neg_pos = prob_neg_pos
        self.dataset=dataset
        self.shuffle = shuffle
        self.on_epoch_end()
        self.n = 0
        self.max = self.__len__()
        self.seed_img = seed_img
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        #print(indexes)
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        img_anchor = np.empty((self.batch_size,self.input_size,self.input_size,self.n_channels)) # to store anchor for left branch of siamese net
        img_twin = np.empty((self.batch_size,self.input_size,self.input_size,self.n_channels)) # to store twin for right branch of siamese net.
        batch_labels = np.empty(self.batch_size,dtype='float') 
        
        if self.dataset == 'train':  # use when dir_path has image files
            # Generate data
            for i, ID in enumerate(list_IDs_temp):
                img_anc=pil.open(self.dir_path+ID)
                anchor = img_resize(img_convert(np.asarray(img_anc))) # anchor image
                #print('rand',random.random())
                if random.random() < self.prob_neg_pos: # select different image
                    list_idx = self.list_IDs.copy()
                    list_idx.remove(ID)
                    img_diff_idx = random.choice(list_idx)
                    img_twin[i,] = np.expand_dims(imcrop(img_resize(img_convert(plt.imread(self.dir_path+img_diff_idx))/255)),axis=2)                            
                    img_anchor[i,] = np.expand_dims(imcrop(anchor)/255,axis=2) # scale pxintensity to [0 1]
                    batch_labels[i] = 0

                else: # create manipulation
                    anc,same=X(img_anc)
                    img_twin[i,] = np.expand_dims(np.asarray(same)/255,axis=2)
                    img_anchor[i,] = np.expand_dims(np.asarray(anc)/255,axis=2)# scale pxl intensity to [0 1]
                    batch_labels[i] = 1
                    
        elif self.dataset == 'valid': # use when dir_path has subfolders with image pairs
            # Generate data
            for i, ID in enumerate(list_IDs_temp):
                test_names = random.sample(os.listdir(self.dir_path + '/' + ID), k=2)
                anchor = plt.imread(self.dir_path + '/' + ID + '/' + test_names[0])
                img_twin[i,:,:,:] = np.expand_dims((plt.imread(self.dir_path+'/'+ ID + '/' + test_names[1])/255),axis=2)
                img_anchor[i,:,:,:] = np.expand_dims((anchor)/255,axis=2)# scale pxl intensity to [0 1]
                batch_labels[i] = 1

        return {'Input_l':img_anchor, 'Input_r':img_twin}, {'output_1':img_anchor, 'output_2':img_twin, 'dense_embedding':batch_labels, 'l2_norm_d':batch_labels} 
    
    def __next__(self):
	# generate batches for each iteration within an epoch
        if self.n >= self.max:
            self.n = 0
        result = self.__getitem__(self.n)
        self.n += 1
        return result
  
""" Data Generator for validation set """
# for BINDER/PUBPEER 
def valid_generator(dir_path,batch_size=2,img_size=128,n_channels=3,seed_dir=None,seed_img=None):

        # check that the directory path is correct and directory is not empty
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            if not os.listdir(dir_path):
                print("Directory {} is empty".format(dir_path))
            else:    
                print("Directory {} is not empty".format(dir_path))
        else:
            print("Given Directory {} doesn.t exists".format(dir_path)) 
        
        Image_filenames = os.listdir(dir_path) # list of file names of images in directory
        img_anchor = np.empty((batch_size,img_size,img_size,n_channels)) # to store anchor for left branch of siamese net
        img_twin = np.empty((batch_size,img_size,img_size,n_channels)) # to store twin for right branch of siamese net.
        batch_labels = np.empty(batch_size) 
        
        # input seed
        if seed_dir:
            np.random.seed(30)
        
        while True: 
          # generate the batches of data : img_anchor (original img), img_twin (manipulation of original or different img)
            img_anchor_idx = np.arange(len(os.listdir(dir_path))) # index of all images
            random.shuffle(img_anchor_idx)
            
            for i,idx in enumerate(img_anchor_idx):
                test_names = random.sample(os.listdir(dir_path + '/' + Image_filenames[idx]), k=2)
                anchor = plt.imread(dir_path + '/' + Image_filenames[idx] + '/' + test_names[0])
                
                if True: # create manipulation
                    img_twin[1,:,:,:] = np.expand_dims((plt.imread(dir_path+'/'+Image_filenames[idx]+'/' + test_names[1])/255),axis=2)
                    img_anchor[1,:,:,:] = np.expand_dims((anchor)/255,axis=2)# scale pxl intensity to [0 1]
                    batch_labels[1] = 1
            
                yield {'Input_l':img_anchor,'Input_r':img_twin} , {'l2_norm_d':batch_labels, 'dense_embedding':batch_labels}
                
    
""" Data Generators for test set """
# for BINDER/PUBPEER with random selection 
def test_generator_random(dir_path,batch_size=2,img_size=128,n_channels=3,seed_dir=None,seed_img=None):

        # check that the directory path is correct and directory is not empty
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            if not os.listdir(dir_path):
                print("Directory {} is empty".format(dir_path))
            else:    
                print("Directory {} is not empty".format(dir_path))
        else:
            print("Given Directory {} doesn.t exists".format(dir_path)) 
        
        Image_filenames = os.listdir(dir_path) # list of file names of images in directory
        img_anchor = np.empty((batch_size,img_size,img_size,n_channels)) # to store anchor for left branch of siamese net
        img_twin = np.empty((batch_size,img_size,img_size,n_channels)) # to store twin for right branch of siamese net.
        batch_labels = np.empty(batch_size) 
        
        # input seed
        if seed_dir:
            np.random.seed(30)
        
        while True: 
          # generate the batches of data : img_anchor (original img), img_twin (manipulation of original or different img)
            img_anchor_idx = np.arange(len(os.listdir(dir_path))) # index of all images
            random.shuffle(img_anchor_idx)
            
            for idx in img_anchor_idx:
                test_names = random.sample(os.listdir(dir_path + '/' + Image_filenames[idx]), k=2)
                anchor = plt.imread(dir_path + '/' + Image_filenames[idx] + '/' + test_names[0])

                if True: # select different image
                    list_idx = list(range(0,len(os.listdir(dir_path))))
                    list_idx.remove(idx)
                    img_diff_idx = random.choice(list_idx)
                    img_diff_name = random.choice(os.listdir(dir_path + '/' + Image_filenames[img_diff_idx]))
                    ##**TODO: add an assertion for num chnanels and shape for diff
                    img_twin[0,:,:,:] = np.expand_dims((plt.imread(dir_path+'/'+Image_filenames[img_diff_idx]+'/' + img_diff_name)/255),axis=2)
                    img_anchor[0,:,:,:] = np.expand_dims((anchor)/255,axis=2) # scale pxintensity to [0 1]
                    batch_labels[0] = 0
                
                if True: # create manipulation
                    img_twin[1,:,:,:] = np.expand_dims((plt.imread(dir_path+'/'+Image_filenames[idx]+'/' + test_names[1])/255),axis=2)
                    img_anchor[1,:,:,:] = np.expand_dims((anchor)/255,axis=2)# scale pxl intensity to [0 1]
                    batch_labels[1] = 1
            
                yield {'Input_l':img_anchor,'Input_r':img_twin} , {'l2_norm_d':batch_labels}
                
# for mfnd with resize and random selection, mfnd needs to be resized to 256 x 256 and cropped to 128 x 128
def test_imageconvert_generator_random(dir_path,batch_size=2,img_size=128,n_channels=3,seed_dir=None,seed_img=None):

        # check that the directory path is correct and directory is not empty
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            if not os.listdir(dir_path):
                print("Directory {} is empty".format(dir_path))
            else:    
                print("Directory {} is not empty".format(dir_path))
        else:
            print("Given Directory {} doesn.t exists".format(dir_path)) 
        
        Image_filenames = os.listdir(dir_path) # list of file names of images in directory
        img_anchor = np.empty((batch_size,img_size,img_size,n_channels)) # to store anchor for left branch of siamese net
        img_twin = np.empty((batch_size,img_size,img_size,n_channels)) # to store twin for right branch of siamese net.
        batch_labels = np.empty(batch_size) 
        
        # input seed
        if seed_dir:
            np.random.seed(30)
        
        while True: 
          # generate the batches of data : img_anchor (original img), img_twin (manipulation of original or different img)
            img_anchor_idx = np.arange(len(os.listdir(dir_path))) # index of all images

            for idx in img_anchor_idx:
                
                test_names = random.sample(os.listdir(dir_path + '/' + Image_filenames[idx]), k=2)
                anchor = img_resize(img_convert(plt.imread(dir_path + '/' + Image_filenames[idx] + '/' + test_names[0])))
            
                if True: # select different image
                    list_idx = list(range(0,len(os.listdir(dir_path))))
                    list_idx.remove(idx)
                    img_diff_idx = random.choice(list_idx)
                    img_diff_name = random.choice(os.listdir(dir_path + '/' + Image_filenames[img_diff_idx]))

                    ##**TODO: add an assertion for num chnanels and shape for diff
                    img_twin[0,:,:,:] = np.expand_dims(resize(img_resize(img_convert(plt.imread(dir_path+'/'+
                                        Image_filenames[img_diff_idx]+'/' + img_diff_name)/255)),(128,128)),axis=2)
                    img_anchor[0,:,:,:] = np.expand_dims(resize(anchor/255,(128,128)),axis=2) # scale pxintensity to [0 1]
                    batch_labels[0] = 0
                
                if True: # create manipulation
                    img_twin[1,:,:,:] = np.expand_dims(resize(img_resize(img_convert(plt.imread(dir_path
                        +'/'+Image_filenames[idx]+'/' + test_names[1])/255)),(128,128)),axis=2)
                    img_anchor[1,:,:,:] = np.expand_dims(resize(anchor/255,(128,128)),axis=2)# scale pxl intensity to [0 1]
                    batch_labels[1] = 1
            
                yield {'Input_l':img_anchor,'Input_r':img_twin} , {'l2_norm_d':batch_labels}

# for BINDER/PUBPEER with harneg selection, mfnd needs to be resized to 256 x 256 and cropped to 128 x 128
def test_generator_hardneg(dir_path,batch_size=2,img_size=128,n_channels=3,seed_dir=None,seed_img=None):

        # check that the directory path is correct and directory is not empty
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            if not os.listdir(dir_path):
                print("Directory {} is empty".format(dir_path))
            else:    
                print("Directory {} is not empty".format(dir_path))
        else:
            print("Given Directory {} doesn.t exists".format(dir_path)) 
        
        Image_filenames = os.listdir(dir_path) # list of file names of images in directory
        img_anchor = np.empty((batch_size,img_size,img_size,n_channels)) # to store anchor for left branch of siamese net
        img_twin = np.empty((batch_size,img_size,img_size,n_channels)) # to store twin for right branch of siamese net.
        batch_labels = np.empty(batch_size) 
        
        # input seed
        if seed_dir:
            np.random.seed(30)
        
        while True: 
          # generate the batches of data : img_anchor (original img), img_twin (manipulation of original or different img)
            img_anchor_idx = np.arange(len(os.listdir(dir_path))) # index of all images
            n=0
            
            for idx in img_anchor_idx:
                
                test_names = random.sample(os.listdir(dir_path + '/' + Image_filenames[idx]), k=2)
                anchor = plt.imread(dir_path + '/' + Image_filenames[idx] + '/' + test_names[0])
                
                img_twin[n,:,:,:] = np.expand_dims(plt.imread(dir_path
                        +'/'+Image_filenames[idx]+'/' + test_names[1])/255,axis=2)
                img_anchor[n,:,:,:] = np.expand_dims(anchor/255,axis=2)# scale pxl intensity to [0 1]
                batch_labels[n] = 1
                
                n += 1
            
            yield {'Input_l':img_anchor,'Input_r':img_twin} , {'l2_norm_d':batch_labels}

# for mfnd with resize and hardneg selection
def test_imageconvert_generator_hardneg(dir_path,batch_size=2,img_size=128,n_channels=3,seed_dir=None,seed_img=None):

        # check that the directory path is correct and directory is not empty
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            if not os.listdir(dir_path):
                print("Directory {} is empty".format(dir_path))
            else:    
                print("Directory {} is not empty".format(dir_path))
        else:
            print("Given Directory {} doesn.t exists".format(dir_path)) 
       
        Image_filenames = os.listdir(dir_path) # list of file names of images in directory
        img_anchor = np.empty((batch_size,img_size,img_size,n_channels)) # to store anchor for left branch of siamese net
        img_twin = np.empty((batch_size,img_size,img_size,n_channels)) # to store twin for right branch of siamese net.
        batch_labels = np.empty(batch_size) 
        
        # input seed
        if seed_dir:
            np.random.seed(30)
        
        while True: 
          # generate the batches of data : img_anchor (original img), img_twin (manipulation of original or different img)
            img_anchor_idx = np.arange(len(os.listdir(dir_path))) # index of all images
            n=0
            
            for idx in img_anchor_idx:
                
                test_names = random.sample(os.listdir(dir_path + '/' + Image_filenames[idx]), k=2)
                anchor = img_resize(img_convert(plt.imread(dir_path + '/' + Image_filenames[idx] + '/' + test_names[0])))
                
                img_twin[n,:,:,:] = np.expand_dims(resize(img_resize(img_convert(plt.imread(dir_path
                        +'/'+Image_filenames[idx]+'/' + test_names[1])/255)),(128,128)),axis=2)
                img_anchor[n,:,:,:] = np.expand_dims(resize(anchor/255,(128,128)),axis=2)# scale pxl intensity to [0 1]
                batch_labels[n] = 1
                
                n += 1
            
            yield {'Input_l':img_anchor,'Input_r':img_twin} , {'l2_norm_d':batch_labels}

class test_gen():
    """
    Generates positive and negative pairs of images 
    for testing/validating the model performance
       
       Input - path to directory with images
       Ouput - Batch of Image pairs and labels: 
               1 = 'same'(postive pair) | 0 = 'different' (negative_pair) 
       
       returns - {dict1 - input to model} , {dict2 - Ground Truth labels}  
       
       dict1:
           'Input_l' - anchor image
           'Input_r' - same src manipulated or random different img
       dict2:
           'l2_norm_d' - batch labels (0 or 1 for different or same respectively)
       
       'Initialization'
           dir_path: path to directory
           test_dataset_name: PUBPEER | MFND | BINDER 
           neg_selection: 
           batch_size: batch size
           input_size: image size (e.g. = 128, i.e. height or width b/c aspect ratio =1) 
           n_channels: number of channels (= 1 or 3) 
           prob_neg_pos:probability of negative samples [0,1]
           dataset: 'train' / 'valid'
                    if directory has img files choose 'train', 
                    if directory has subfolders with img files choose'valid'
                  
    """
    def __init__(self, dir_path, test_dataset_name, neg_selection,  batch_size, n_channels, img_size=128, seed_dir=None,seed_img=None):
        self.dir_path=dir_path
        self.test_dataset_name=test_dataset_name
        self.neg_selection=neg_selection
        self.img_size=img_size
        self.batch_size=batch_size
        self.n_channels=n_channels
        self.seed_dir=seed_dir
        self.seed_img=seed_img
        
    def select_generator(self):
        if self.test_dataset_name=='PUBPEER' or self.test_dataset_name=='BINDER':
            if self.neg_selection=='random':
                data_generator= test_generator_random(dir_path=self.dir_path,batch_size=self.batch_size,n_channels=self.n_channels)
            elif self.neg_selection=='hardneg':
                data_generator= test_generator_hardneg(dir_path=self.dir_path,batch_size=self.batch_size,n_channels=self.n_channels)
            else:
                raise Exception('ValueError: Invalid str for neg_selection: {}; should be either "random" or "hardneg" '.format(self.neg_selection))
        elif self.test_dataset_name=='MFND':
            if self.neg_selection=='random':
                data_generator= test_imageconvert_generator_random(dir_path=self.dir_path,batch_size=self.batch_size,n_channels=self.n_channels)
            elif self.neg_selection=='hardneg':
                data_generator= test_imageconvert_generator_hardneg(dir_path=self.dir_path,batch_size=self.batch_size,n_channels=self.n_channels)
            else:
                raise Exception('ValueError: Invalid str for neg_selection: {}; should be either "random" or "hardneg" '.format(self.neg_selection))
        else:
            raise Exception('ValueError: Invalid str for test_dataset: {}; should be either "PUBPEER" or "BINDER" or "MFND"'.format(self.test_dataset))

        return data_generator