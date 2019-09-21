import os ,sys
import numpy as np
from PIL import Image
import random
from random import *


select_amt = 1
ROOT_DIR = os.getcwd()
align_image_path = os.path.join(ROOT_DIR,'img_align_celeba')
origin_image_path = os.path.join(ROOT_DIR,'img_celeba/img_celeba')
bbox_path = os.path.join(ROOT_DIR,'list_bbox_celeba.csv')
landmark_path = os.path.join(ROOT_DIR,'list_landmarks_align_celeba.csv')     #  create path

origin_image = os.listdir( origin_image_path )
oimage_amount = len(origin_image)

select_file = [[0] for i in range(0,select_amt)]

for i in range(0,select_amt):
	select_file[i][0]= origin_image[i]  # random choose  numbers of pic
	
	im = Image.open(origin_image_path + '/' + str(select_file[i][0]))
	img=im.resize((10,10))  
	img.save('./test.jpg')
	image = np.asarray(img)
	print(image)

	

