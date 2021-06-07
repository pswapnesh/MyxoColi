from skimage.io import imread,imsave
import numpy as np
import matplotlib.pyplot as plt
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf    
from skimage.feature import shape_index
from skimage.transform import rescale,resize
from skimage.morphology import remove_small_objects
from skimage.filters import laplace, gaussian
from skimage.segmentation import watershed
from skimage.segmentation import clear_border
from utils import *
from skimage.exposure import adjust_gamma
from scipy.ndimage import label


class MyxoColi():
    def __init__(self):
        
        model_name = tf.keras.utils.get_file("mxec31072020","https://github.com/pswapnesh/Models/raw/master/mxec31072020.h5")
        self.model = tf.keras.models.load_model(model_name,compile=False)  
        self.size = 256        
    
    def segment(self,im,exclude = 32):                        
        im = 1.0 - normalize2max(im)
        tiles,params = extract_tiles(im,size = self.size,exclude = exclude)               
        yp = self.model.predict(tiles)
        return stitch_tiles(yp,params)
    
    def postprocess(self,im,yp):
        mask = adjust_gamma( yp[:,:,0]+yp[:,:,1],0.75) > 0.5
        th_img = yp[:,:,0] > 0.9
        tmp = clear_border(th_img)
        border = np.logical_xor(th_img,tmp)
        marker_myxo,_ = label(remove_small_objects(tmp,210) + border) 
        ws = watershed(im, markers=marker_myxo,watershed_line = True,mask = mask,compactness = 1,connectivity = 8)                        
        #ws = remove_small_objects(ws,256)
        res = np.zeros((yp.shape[0],yp.shape[1],3))
        res[:,:,0] = ws
        res[:,:,1] = mask - 1.0*(ws>0)
        return res
        