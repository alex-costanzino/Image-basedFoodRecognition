'''
Alex Costanzino, Marco Costante
MSc student in Artificial Intelligence
@ Alma Mater Studiorum, University of Bologna
March, 2021
'''

# We'll check the dimension of the various images in order to choose a trad-eoff between upsampling and downsampling of the images
import os
import cv2
import numpy as np
from tqdm import tqdm

# Choose the right directory
image_directory = './images/'

width = []
height = []
image_dataset = [] 

images = os.listdir(image_directory)

for i, image_name in tqdm(enumerate(images), total = len(images), position = 0, leave = True):
  image = cv2.imread(image_directory + image_name, 0)
  width.append(image.shape[0])
  height.append(image.shape[1])

print(np.mean(width))
print(np.mean(height))

print(np.median(width))
print(np.median(height))