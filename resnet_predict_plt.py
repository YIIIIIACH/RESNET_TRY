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
from resnet  import small_resnet

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # block the warning message on tensorflow

##                        TRAINING  HYPERPARAMETER
L2_WEIGHT_DECAY = 0.001
BATCH_NORM_DECAY = 0.001
BATCH_NORM_EPSILON = 0.005
# # #                   #################################################

test_path = './test_img'

files = listdir(test_path)               ###
test_file = []       
for f in files:
  test_file.append(join(test_path, f))


#data_root = pathlib.Path('/home/yiiiiiach/MTCNN_TRY/image_celeba/')
#ROOT_DIR = os.getcwd()
'''
with open('img10000.json', 'r') as f:
	data = json.load(f)	
	tuple_of_data = len(data)	
	print('tuple_of_data',tuple_of_data )
'''
all_image_paths = []
'''
for i in range(0,tuple_of_data):
	all_image_paths.append( '/home/yiiiiiach/MTCNN_TRY/img_celeba/img_celeba/'  + data[i][0]) 
'''
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

model = small_resnet( (48,48,3) )
model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
              loss={'bbox':'mean_squared_error'},loss_weights={'bbox':0.5  },metrics=["accuracy"])
#model.summary()
model.load_weights('weight_file/small_resnet_70weight.h5')
#rand_seed = random.randint(1,tuple_of_data)
#test_image_path =  '/home/yiiiiiach/MTCNN_TRY/img_celeba/img_celeba/'  + data[rand_seed][0]                 # the file to be test   #all_image_paths[rand_seed]

rand_seed = random.randint(0,len(test_file))
list_img = []
list_img.append(test_file[ rand_seed-1 ])

print('test_image_path', test_file[ rand_seed-1 ])

test_path = tf.data.Dataset.from_tensor_slices(list_img )
test_image = test_path.map(load_and_preprocess_image , num_parallel_calls=AUTOTUNE)
test_image = test_image.batch(1)

result = model.predict(test_image , steps =  1)


bbox_result = result[0]

print('-------------------------------------------------------------------------')
print('\n\n\n')
print('test_image_path', test_path)
print('bbox result =======>> :   ', bbox_result)

from PIL import Image , ImageDraw

im = Image.open(test_file[ rand_seed-1 ]) 

bbox_result[0] = (bbox_result[0]/48)*im.size[0]
bbox_result[1] = (bbox_result[1]/48)*im.size[1]
bbox_result[2] = (bbox_result[2]/48)*im.size[0]
bbox_result[3] = (bbox_result[3]/48)*im.size[1]
print('bbox result =======>> :   ', bbox_result)



print('im.size : ' ,im.size)

drawSurface = ImageDraw.Draw(im)
d = 20



drawSurface.line(( (bbox_result[0],bbox_result[1]) , (bbox_result[0]+bbox_result[2],bbox_result[1]) , (bbox_result[0]+bbox_result[2],bbox_result[1]+bbox_result[3]) , (bbox_result[0],bbox_result[1]+bbox_result[3]) , (bbox_result[0],bbox_result[1]) ),fill = (255,0,0), width = 3)



im.show()
del drawSurface
#im.save('test.jpeg',format='jpeg')

