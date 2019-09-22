import tensorflow as tf
tf.enable_eager_execution()
AUTOTUNE = tf.data.experimental.AUTOTUNE
import pathlib
import os
import random
import json
from os import listdir
from os.path import isfile, join
from tensorflow import keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # block the warning message on tensorflow
L2_WEIGHT_DECAY = 0.001
BATCH_NORM_DECAY = 0.001
BATCH_NORM_EPSILON = 0.005
from resnet  import resnet50


data_root = pathlib.Path('/home/yiiiiiach/MTCNN_TRY/image_celeba/')

ROOT_DIR = os.getcwd()
#align_image_path = os.path.join(ROOT_DIR,'img_align_celeba/')

with open('img10000.json', 'r') as f:
	data = json.load(f)	
	tuple_of_data = len(data)	
	print('tuple_of_data',tuple_of_data )

all_image_paths = []

for i in range(0,tuple_of_data):
	all_image_paths.append( '/home/yiiiiiach/MTCNN_TRY/img_celeba/img_celeba/'  + data[i][0]) 

def preprocess_image(image):
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.resize(image, [48, 48])                              #  CONTROL THE INPUT SIZE OF MODEL
  image /= 255.0  # normalize to [0,1] range

  return image

def load_and_preprocess_image(path):
  image = tf.read_file(path)
  return preprocess_image(image)

#import matplotlib.pyplot as plt

										      #--------------------------------------------------------#

model = resnet50( (48,48,3) )

model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
              loss={'bbox':'mean_squared_error'},loss_weights={'bbox':0.5  },metrics=["accuracy"])
#model.summary()

#steps_per_epoch=tf.ceil(len(all_image_paths)/BATCH_SIZE).numpy()
model.load_weights('weight_file/resnet_final_weight.h5')
rand_seed = random.randint(1,tuple_of_data)
test_image_path = all_image_paths[rand_seed]                  # the file to be test
list_img = []
list_img.append(test_image_path)

print('test_image_path', test_image_path)

test_path = tf.data.Dataset.from_tensor_slices(list_img )
test_image = test_path.map(load_and_preprocess_image , num_parallel_calls=AUTOTUNE)
test_image = test_image.batch(1)

result = model.predict(test_image)


bbox_result = result[0]



print('-------------------------------------------------------------------------')
print('\n\n\n')
print('test_image_path', test_image_path)
print('bbox result =======>> :   ', bbox_result)

from PIL import Image , ImageDraw
im = Image.open(test_image_path) 


bbox_result[0] = (bbox_result[0]/48)*im.size[0]
bbox_result[1] = (bbox_result[1]/48)*im.size[1]
bbox_result[2] = (bbox_result[2]/48)*im.size[0]
bbox_result[3] = (bbox_result[3]/48)*im.size[1]
print('bbox result =======>> :   ', bbox_result)

exact_bbox = []
#print('lm result =======>> :   ', lm_result)
with open('img10000.json', 'r') as f:
	data = json.load(f)	
	print('load json: ' ,data[rand_seed])
	exact_bbox = 	data[rand_seed][1:5]

exact_bbox[0] = int(exact_bbox[0])/48*im.size[0]
exact_bbox[1] = int(exact_bbox[1])/48*im.size[1]
exact_bbox[2] = int(exact_bbox[2])/48*im.size[0]
exact_bbox[3] = int(exact_bbox[3])/48*im.size[1]


for i in range(0,4):
	exact_bbox[i] = int(exact_bbox[i])

print('im.size : ' ,im.size)
print('exact_bbox : ' , exact_bbox)
drawSurface = ImageDraw.Draw(im)
d = 20
drawSurface.line(( (exact_bbox[0],exact_bbox[1]) , (exact_bbox[0]+exact_bbox[2],exact_bbox[1]) , (exact_bbox[0]+exact_bbox[2],exact_bbox[1]+exact_bbox[3]) , (exact_bbox[0],exact_bbox[1]+exact_bbox[3]) , (exact_bbox[0],exact_bbox[1]) ),fill = (0,255,0), width = 3)
drawSurface.line(( (bbox_result[0],bbox_result[1]) , (bbox_result[0]+bbox_result[2],bbox_result[1]) , (bbox_result[0]+bbox_result[2],bbox_result[1]+bbox_result[3]) , (bbox_result[0],bbox_result[1]+bbox_result[3]) , (bbox_result[0],bbox_result[1]) ),fill = (255,0,0), width = 3)



im.show()
del drawSurface
#im.save('test.jpeg',format='jpeg')

