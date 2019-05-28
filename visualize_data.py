import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
#from keras.models import load_model
import pandas as pd
from PIL import Image
from pathlib import Path
import sys, os

print('sys.argv[0] =', sys.argv[0])             
pathname = os.path.dirname(sys.argv[0])        
print('path =', pathname)
print('full path =', os.path.abspath(pathname)) 



#Folder path where I keep the original folders data from kaggle ("train", "test", and csv's)
data_fullpath = os.path.abspath(pathname)+'/data'
print(data_fullpath) 
print(os.listdir(data_fullpath))

train_df = pd.read_csv(data_fullpath+'/train_relationships.csv')
print(train_df.head())
train_list = os.listdir(data_fullpath+'/train')
print('Number of Family Folders: ', len(train_list))

test_df = pd.read_csv(data_fullpath+'/sample_submission.csv')
test_list = os.listdir(data_fullpath+'/test')
print('Number of faces in teh test folder: ',len(test_list))


img_path = Path(data_fullpath+'/train/')
img_list = os.listdir(img_path / train_df.p1[0])




def convert_image_to_array(image_dir):
    try:
        image = cv2.imread(image_dir)
        if image is not None :
            return list(image)
        else :
            return np.array([])
    except Exception as e:
        print(f"Error : {e}")
        return None

train_image_list = os.listdir(data_fullpath+'/train/'+train_df.p1[0])
train_image_list1 = os.listdir(data_fullpath+'/train/'+train_df.p2[0])

p1 = []
p2 = []
for image in train_image_list:
    p1.append(convert_image_to_array(data_fullpath+'/train/'+train_df.p1[0]+"/"+image))
for image in train_image_list1:
    p2.append(convert_image_to_array(data_fullpath+'/train/'+train_df.p2[0]+"/"+image))



plt.figure(figsize=(16,10))
for i in range(1,10):
    plt.subplot(2,5,i)
    plt.grid(False)
    plt.imshow(p1[i])
    #plt.xlabel(label_list[i])
plt.show()
#plt.pause(3)

plt.figure(figsize=(16,10))
for i in range(1,8):
    plt.subplot(2,5,i)
    plt.grid(False)
    plt.imshow(p2[i])
    #plt.xlabel(label_list[i])
plt.show()
#plt.pause(3)

