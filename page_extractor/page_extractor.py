#! /usr/bin/env python3

__version__ = '1.0'

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sys import getsizeof
import random
from tqdm import tqdm
from keras.models import model_from_json
from keras.models import load_model

from sklearn.cluster import KMeans
import gc
from keras import backend as K
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import xml.etree.ElementTree as ET
import warnings
import click
import time
from matplotlib import pyplot, transforms
import imutils

warnings.filterwarnings('ignore')
#
__doc__ = \
    """
    tool to extract table form data from alto xml data
    """
class page_extractor:
    def __init__(self, image_dir,dir_out, dir_models , co_image = False, out_co = False):
        self.image_dir = image_dir  # XXX This does not seem to be a directory as the name suggests, but a file
        self.dir_out = dir_out
        self.kernel = np.ones((5, 5), np.uint8)
        
        self.image = cv2.imread(self.image_dir)
        self.model_page_dir = dir_models
        self.co_image = co_image
        if co_image:
            self.image_co = cv2.imread(self.co_image)
        self.out_co = out_co
            
    def resize_image(self, img_in, input_height, input_width):
        return cv2.resize(img_in, (input_width, input_height), interpolation=cv2.INTER_NEAREST)

            
    def start_new_session_and_model(self, model_dir):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        session = tf.InteractiveSession()
        model = load_model(model_dir, compile=False)

        return model, session
    
    def do_prediction(self,patches,img,model,marginal_of_patch_percent=0.1):
        
        img_height_model = model.layers[len(model.layers) - 1].output_shape[1]
        img_width_model = model.layers[len(model.layers) - 1].output_shape[2]
        n_classes = model.layers[len(model.layers) - 1].output_shape[3]
        


        if patches:
            if img.shape[0]<img_height_model:
                img=self.resize_image(img,img_height_model,img.shape[1])
                
            if img.shape[1]<img_width_model:
                img=self.resize_image(img,img.shape[0],img_width_model)
            
            #print(img_height_model,img_width_model)
            #margin = int(0.2 * img_width_model)
            margin = int(marginal_of_patch_percent * img_height_model)

            width_mid = img_width_model - 2 * margin
            height_mid = img_height_model - 2 * margin


            img = img / float(255.0)
            #print(sys.getsizeof(img))
            #print(np.max(img))
            
            img=img.astype(np.float16)
            
            #print(sys.getsizeof(img))

            img_h = img.shape[0]
            img_w = img.shape[1]

            prediction_true = np.zeros((img_h, img_w, 3))
            mask_true = np.zeros((img_h, img_w))
            nxf = img_w / float(width_mid)
            nyf = img_h / float(height_mid)

            if nxf > int(nxf):
                nxf = int(nxf) + 1
            else:
                nxf = int(nxf)

            if nyf > int(nyf):
                nyf = int(nyf) + 1
            else:
                nyf = int(nyf)

            for i in range(nxf):
                for j in range(nyf):

                    if i == 0:
                        index_x_d = i * width_mid
                        index_x_u = index_x_d + img_width_model
                    elif i > 0:
                        index_x_d = i * width_mid
                        index_x_u = index_x_d + img_width_model

                    if j == 0:
                        index_y_d = j * height_mid
                        index_y_u = index_y_d + img_height_model
                    elif j > 0:
                        index_y_d = j * height_mid
                        index_y_u = index_y_d + img_height_model

                    if index_x_u > img_w:
                        index_x_u = img_w
                        index_x_d = img_w - img_width_model
                    if index_y_u > img_h:
                        index_y_u = img_h
                        index_y_d = img_h - img_height_model
                        
                    

                    img_patch = img[index_y_d:index_y_u, index_x_d:index_x_u, :]

                    label_p_pred = model.predict(
                        img_patch.reshape(1, img_patch.shape[0], img_patch.shape[1], img_patch.shape[2]))

                    seg = np.argmax(label_p_pred, axis=3)[0]

                    seg_color = np.repeat(seg[:, :, np.newaxis], 3, axis=2)

                    if i==0 and j==0:
                        seg_color = seg_color[0:seg_color.shape[0] - margin, 0:seg_color.shape[1] - margin, :]
                        seg = seg[0:seg.shape[0] - margin, 0:seg.shape[1] - margin]

                        mask_true[index_y_d + 0:index_y_u - margin, index_x_d + 0:index_x_u - margin] = seg
                        prediction_true[index_y_d + 0:index_y_u - margin, index_x_d + 0:index_x_u - margin,
                        :] = seg_color
                        
                    elif i==nxf-1 and j==nyf-1:
                        seg_color = seg_color[margin:seg_color.shape[0] - 0, margin:seg_color.shape[1] - 0, :]
                        seg = seg[margin:seg.shape[0] - 0, margin:seg.shape[1] - 0]

                        mask_true[index_y_d + margin:index_y_u - 0, index_x_d + margin:index_x_u - 0] = seg
                        prediction_true[index_y_d + margin:index_y_u - 0, index_x_d + margin:index_x_u - 0,
                        :] = seg_color
                        
                    elif i==0 and j==nyf-1:
                        seg_color = seg_color[margin:seg_color.shape[0] - 0, 0:seg_color.shape[1] - margin, :]
                        seg = seg[margin:seg.shape[0] - 0, 0:seg.shape[1] - margin]

                        mask_true[index_y_d + margin:index_y_u - 0, index_x_d + 0:index_x_u - margin] = seg
                        prediction_true[index_y_d + margin:index_y_u - 0, index_x_d + 0:index_x_u - margin,
                        :] = seg_color
                        
                    elif i==nxf-1 and j==0:
                        seg_color = seg_color[0:seg_color.shape[0] - margin, margin:seg_color.shape[1] - 0, :]
                        seg = seg[0:seg.shape[0] - margin, margin:seg.shape[1] - 0]

                        mask_true[index_y_d + 0:index_y_u - margin, index_x_d + margin:index_x_u - 0] = seg
                        prediction_true[index_y_d + 0:index_y_u - margin, index_x_d + margin:index_x_u - 0,
                        :] = seg_color
                        
                    elif i==0 and j!=0 and j!=nyf-1:
                        seg_color = seg_color[margin:seg_color.shape[0] - margin, 0:seg_color.shape[1] - margin, :]
                        seg = seg[margin:seg.shape[0] - margin, 0:seg.shape[1] - margin]

                        mask_true[index_y_d + margin:index_y_u - margin, index_x_d + 0:index_x_u - margin] = seg
                        prediction_true[index_y_d + margin:index_y_u - margin, index_x_d + 0:index_x_u - margin,
                        :] = seg_color
                        
                    elif i==nxf-1 and j!=0 and j!=nyf-1:
                        seg_color = seg_color[margin:seg_color.shape[0] - margin, margin:seg_color.shape[1] - 0, :]
                        seg = seg[margin:seg.shape[0] - margin, margin:seg.shape[1] - 0]

                        mask_true[index_y_d + margin:index_y_u - margin, index_x_d + margin:index_x_u - 0] = seg
                        prediction_true[index_y_d + margin:index_y_u - margin, index_x_d + margin:index_x_u - 0,
                        :] = seg_color
                        
                    elif i!=0 and i!=nxf-1 and j==0:
                        seg_color = seg_color[0:seg_color.shape[0] - margin, margin:seg_color.shape[1] - margin, :]
                        seg = seg[0:seg.shape[0] - margin, margin:seg.shape[1] - margin]

                        mask_true[index_y_d + 0:index_y_u - margin, index_x_d + margin:index_x_u - margin] = seg
                        prediction_true[index_y_d + 0:index_y_u - margin, index_x_d + margin:index_x_u - margin,
                        :] = seg_color
                        
                    elif i!=0 and i!=nxf-1 and j==nyf-1:
                        seg_color = seg_color[margin:seg_color.shape[0] - 0, margin:seg_color.shape[1] - margin, :]
                        seg = seg[margin:seg.shape[0] - 0, margin:seg.shape[1] - margin]

                        mask_true[index_y_d + margin:index_y_u - 0, index_x_d + margin:index_x_u - margin] = seg
                        prediction_true[index_y_d + margin:index_y_u - 0, index_x_d + margin:index_x_u - margin,
                        :] = seg_color

                    else:
                        seg_color = seg_color[margin:seg_color.shape[0] - margin, margin:seg_color.shape[1] - margin, :]
                        seg = seg[margin:seg.shape[0] - margin, margin:seg.shape[1] - margin]

                        mask_true[index_y_d + margin:index_y_u - margin, index_x_d + margin:index_x_u - margin] = seg
                        prediction_true[index_y_d + margin:index_y_u - margin, index_x_d + margin:index_x_u - margin,
                        :] = seg_color

            prediction_true = prediction_true.astype(np.uint8)
            del img
            del mask_true
            del seg_color
            del seg
            del img_patch
                
        if not patches:
            img_h_page=img.shape[0]
            img_w_page=img.shape[1]
            img = img /float( 255.0)
            img = self.resize_image(img, img_height_model, img_width_model)

            label_p_pred = model.predict(
                img.reshape(1, img.shape[0], img.shape[1], img.shape[2]))

            seg = np.argmax(label_p_pred, axis=3)[0]
            seg_color =np.repeat(seg[:, :, np.newaxis], 3, axis=2)
            prediction_true = self.resize_image(seg_color, img_h_page, img_w_page)
            prediction_true = prediction_true.astype(np.uint8)
            
            
            del img
            del seg_color
            del label_p_pred
            del seg
        del model
        gc.collect()
        
        return prediction_true
    
    def crop_image_inside_box(self, box, img_org_copy):
        image_box = img_org_copy[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]
        return image_box, [box[1], box[1] + box[3], box[0], box[0] + box[2]]
            
    def extract_page(self):
        patches=False
        model_page, session_page = self.start_new_session_and_model(self.model_page_dir)
        ###img = self.otsu_copy(self.image)
        for ii in range(1):
            img = cv2.GaussianBlur(self.image, (5, 5), 0)

        
        img_page_prediction=self.do_prediction(patches,img,model_page)
        
        imgray = cv2.cvtColor(img_page_prediction, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(imgray, 0, 255, 0)

        thresh = cv2.dilate(thresh, self.kernel, iterations=3)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cnt_size = np.array([cv2.contourArea(contours[j]) for j in range(len(contours))])

        cnt = contours[np.argmax(cnt_size)]

        x, y, w, h = cv2.boundingRect(cnt)
        

        if x<=30:
            w=w+x
            x=0
        if (self.image.shape[1]-(x+w) )<=30:
            w=w+(self.image.shape[1]-(x+w) )
        
        if y<=30:
            h=h+y
            y=0
        if (self.image.shape[0]-(y+h) )<=30:
            h=h+(self.image.shape[0]-(y+h) )
            
            

        box = [x, y, w, h]

        croped_page, page_coord = self.crop_image_inside_box(box, self.image)
        
        if self.co_image:
            co_image_page = self.image_co[page_coord[0]:page_coord[1] , page_coord[2]:page_coord[3] , :]
        
        else:
            co_image_page = None
        
        self.cont_page=[]
        self.cont_page.append( np.array( [ [ page_coord[2] , page_coord[0] ] , 
                                                    [ page_coord[3] , page_coord[0] ] ,
                                                    [ page_coord[3] , page_coord[1] ] ,
                                                [ page_coord[2] , page_coord[1] ]] ) )

        session_page.close()
        del model_page
        del session_page
        del contours
        del thresh
        del img
        del imgray

        gc.collect()
        return croped_page, page_coord, co_image_page
    def run(self):


        
        image_page,page_coord, co_image_page=self.extract_page()
        
        cv2.imwrite(self.dir_out,image_page)
        

        if self.out_co:
            cv2.imwrite(self.out_co,co_image_page)
        

    
        


        

@click.command()
@click.option('--image', '-i', help='image filename')
@click.option('--out', '-o', help='output image name with directory')
@click.option('--model', '-m', help='directory of model')
@click.option('--co_image', '-ci', help='corresponding image file name that will be cropped as main image. In the case that you dont have any co image this option is not needed.')
@click.option('--out_co', '-co', help='output name for corresponding image name with directory.  In the case that you dont have any co image this option is not needed.')


def main(image,out, model, co_image, out_co):
    possibles = globals()  # XXX unused?
    possibles.update(locals())
    x = page_extractor(image, out, model, co_image, out_co)
    x.run()


if __name__ == "__main__":
    main()
  
