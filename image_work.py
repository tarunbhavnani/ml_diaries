#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 12:51:43 2019

@author: tarun.bhavnani@dev.smecorner.com
"""
a=[]
b= open("image1.jpg","rb").read()  
print(b)
a.append(b)
b=open("image2.jpg","rb").read()
a.append(b)
b=open("image3.jpg","rb").read()
a.append(b)
len(a)

a1=set(a)
len(a1)




#####blurr image

import cv2
import numpy as np
import argparse
 

from PIL import Image
Image.open('image1.jpg')


# create the argument parser and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument('-i', '--image', required = True, help = 'image1.jpg')
#args = vars(ap.parse_args())
 
# read the image
#image = cv2.imread(args['image'])
image= cv2.imread('image1.jpg')
# apply the 3x3 mean filter on the image
kernel = np.ones((3,3),np.float32)/9

processed_image = cv2.filter2D(image,-1,kernel)

# display image
cv2.imshow('Mean Filter Processing', processed_image)
# save image to disk
cv2.imwrite('processed_image.png', processed_image)
# pause the execution of the script until a key on the keyboard is pressed
cv2.waitKey(0)

Image.open('processed_image.png')
#################################3
#denoise image



import cv2
import numpy as np
import argparse
 
from PIL import Image
Image.open('sky.jpg')


# create the argument parser and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument('-i', '--image', required = True, help = 'image1.jpg')
#args = vars(ap.parse_args())
 
# read the image
#image = cv2.imread(args['image'])
image_sky= cv2.imread("sky.jpg")
# apply the 3x3 mean filter on the image

processed_image = cv2.medianBlur(image_sky, 3)
# display image
cv2.imshow('Mean Filter Processing', processed_image)
# save image to disk
cv2.imwrite('processed_image.png', processed_image)
# pause the execution of the script until a key on the keyboard is pressed
cv2.waitKey(0)

#see image
Image.open('processed_image.png')




#
kernel= np.asarray([[1,0,1],[1,0,1],[1,0,1]])
image_sky= cv2.imread("sky.jpg")
kernel= np.asarray([[1,0,1],[1,0,1],[1,0,1]])
#kernel= np.asarray([[1,1,1],[0,0,0],[1,1,1]])

pimage= cv2.filter2D(image_sky,-1,kernel)
cv2.imwrite('processed_image.png', pimage)

Image.open('processed_image.png')

