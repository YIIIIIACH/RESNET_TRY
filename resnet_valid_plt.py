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

def identity_block(input_tensor, kernel_size, filters):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
    """
    filters1, filters2, filters3 = filters
    if tf.keras.backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
 
    x = tf.keras.layers.Conv2D(filters1, (1, 1), use_bias=False,
                      kernel_initializer='he_normal',
                      kernel_regularizer=tf.keras.regularizers.l2(L2_WEIGHT_DECAY))(input_tensor)
 
    x = tf.keras.layers.BatchNormalization(axis=bn_axis,
                                  momentum=BATCH_NORM_DECAY,
                                  epsilon=BATCH_NORM_EPSILON)(x)
    x = tf.keras.layers.Activation('relu')(x)
 
    x = tf.keras.layers.Conv2D(filters2, kernel_size,
                      padding='same', use_bias=False,
                      kernel_initializer='he_normal',
                      kernel_regularizer=tf.keras.regularizers.l2(L2_WEIGHT_DECAY))(x)
 
    x = tf.keras.layers.BatchNormalization(axis=bn_axis,
                                  momentum=BATCH_NORM_DECAY,
                                  epsilon=BATCH_NORM_EPSILON)(x)
 
    x = tf.keras.layers.Activation('relu')(x)
 
    x = tf.keras.layers.Conv2D(filters3, (1, 1), use_bias=False,
                      kernel_initializer='he_normal',
                      kernel_regularizer=tf.keras.regularizers.l2(L2_WEIGHT_DECAY))(x)
 
    x = tf.keras.layers.BatchNormalization(axis=bn_axis,
                                  momentum=BATCH_NORM_DECAY,
                                  epsilon=BATCH_NORM_EPSILON)(x)
 
    x = tf.keras.layers.add([x, input_tensor])
    x = tf.keras.layers.Activation('relu')(x)
    return x

#------------------------------------------------------------------------------------------------------------

def conv_block(input_tensor, kernel_size, filters, strides=(2, 2)):
    """A block that has a conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
    # Returns
        Output tensor for the block.
    Note that from stage 3,
    the second conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
 
    filters1, filters2, filters3 = filters
 
    if tf.keras.backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
 
    x = tf.keras.layers.Conv2D(filters1, (1, 1), use_bias=False,
                      kernel_initializer='he_normal',
                      kernel_regularizer=tf.keras.regularizers.l2(L2_WEIGHT_DECAY))(input_tensor)
    x = tf.keras.layers.BatchNormalization(axis=bn_axis,
                                  momentum=BATCH_NORM_DECAY,
                                  epsilon=BATCH_NORM_EPSILON)(x)
    x = tf.keras.layers.Activation('relu')(x)
 
 
    x = tf.keras.layers.Conv2D(filters2, kernel_size, strides=strides, padding='same',
                      use_bias=False, kernel_initializer='he_normal',
                      kernel_regularizer=tf.keras.regularizers.l2(L2_WEIGHT_DECAY))(x)
    x = tf.keras.layers.BatchNormalization(axis=bn_axis,
                                  momentum=BATCH_NORM_DECAY,
                                  epsilon=BATCH_NORM_EPSILON)(x)
    x = tf.keras.layers.Activation('relu')(x)
 
    x = tf.keras.layers.Conv2D(filters3, (1, 1), use_bias=False,
                      kernel_initializer='he_normal',
                      kernel_regularizer=tf.keras.regularizers.l2(L2_WEIGHT_DECAY))(x)
    x = tf.keras.layers.BatchNormalization(axis=bn_axis,
                                  momentum=BATCH_NORM_DECAY,
                                  epsilon=BATCH_NORM_EPSILON)(x)
 
    shortcut = tf.keras.layers.Conv2D(filters3, (1, 1), strides=strides, use_bias=False,
                             kernel_initializer='he_normal',
                             kernel_regularizer=tf.keras.regularizers.l2(L2_WEIGHT_DECAY))(input_tensor)
    shortcut = tf.keras.layers.BatchNormalization(axis=bn_axis,
                                         momentum=BATCH_NORM_DECAY,
                                         epsilon=BATCH_NORM_EPSILON)(shortcut)
 
    x = tf.keras.layers.add([x, shortcut])
    x = tf.keras.layers.Activation('relu')(x)
    return x

#-------------------------------------------------------------------------------------------------------------

def resnet50( input_shape):
    img_input = tf.keras.layers.Input(shape=input_shape)
 
    if tf.keras.backend.image_data_format() == 'channels_first':
        x = tf.keras.layers.Lambda(lambda x: tf.keras.backend.permute_dimensions(x, (0, 3, 1, 2)),
                          name='transpose')(img_input)
        bn_axis = 1
    else:  # channels_last
        x = img_input
        bn_axis = 3
 
    # Conv1 (7x7,64,stride=2)
    x = tf.keras.layers.ZeroPadding2D(padding=(3, 3))(x)
 
    x = tf.keras.layers.Conv2D(64, (7, 7),
                      strides=(2, 2),
                      padding='valid', use_bias=False,
                      kernel_initializer='he_normal',
                      kernel_regularizer=tf.keras.regularizers.l2(L2_WEIGHT_DECAY))(x)
    x = tf.keras.layers.BatchNormalization(axis=bn_axis,
                                  momentum=BATCH_NORM_DECAY,
                                  epsilon=BATCH_NORM_EPSILON)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(x)
 
    # 3x3 max pool,stride=2
    x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
 
    # Conv2_x
 
    # 1×1, 64
    # 3×3, 64
    # 1×1, 256
 
    x = conv_block(x, 3, [64, 64, 256], strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256])
    x = identity_block(x, 3, [64, 64, 256])
 
    # Conv3_x
    #
    # 1×1, 128
    # 3×3, 128
    # 1×1, 512
 
    x = conv_block(x, 3, [128, 128, 512])
    x = identity_block(x, 3, [128, 128, 512])
    x = identity_block(x, 3, [128, 128, 512])
    x = identity_block(x, 3, [128, 128, 512])
 
    # Conv4_x
    # 1×1, 256
    # 3×3, 256
    # 1×1, 1024
    x = conv_block(x, 3, [256, 256, 1024])
    x = identity_block(x, 3, [256, 256, 1024])
    x = identity_block(x, 3, [256, 256, 1024])
    x = identity_block(x, 3, [256, 256, 1024])
    x = identity_block(x, 3, [256, 256, 1024])
    x = identity_block(x, 3, [256, 256, 1024])
 
    # 1×1, 512
    # 3×3, 512
    # 1×1, 2048
    x = conv_block(x, 3, [512, 512, 2048])
    x = identity_block(x, 3, [512, 512, 2048])
    x = identity_block(x, 3, [512, 512, 2048])
 
    # average pool, 1000-d fc, softmax
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    bbox = tf.keras.layers.Dense(  4,name='bbox', kernel_regularizer=tf.keras.regularizers.l2(L2_WEIGHT_DECAY),bias_regularizer=tf.keras.regularizers.l2(L2_WEIGHT_DECAY))(x)                                                 
    landmark = tf.keras.layers.Dense(  10,name='landmark' , kernel_regularizer=tf.keras.regularizers.l2(L2_WEIGHT_DECAY),bias_regularizer=tf.keras.regularizers.l2(L2_WEIGHT_DECAY))(x) 
    # Create model.
    return tf.keras.models.Model(inputs=[img_input], outputs=[bbox,landmark], name='resnet50')






#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
data_root = pathlib.Path('/home/yiiiiiach/MTCNN_TRY/image_align_celeba/')

ROOT_DIR = os.getcwd()
align_image_path = os.path.join(ROOT_DIR,'img_align_celeba/')

with open('10000img.json', 'r') as f:
	data = json.load(f)	
	tuple_of_data = len(data)	
	print('tuple_of_data',tuple_of_data )

all_image_paths = []

for i in range(0,tuple_of_data):
	all_image_paths.append(align_image_path + data[i][0]) 

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
              loss={'bbox':'mean_squared_error','landmark':'mean_squared_error'},loss_weights={'bbox':0.5 , 'landmark':0.5 },metrics=["accuracy"])
#model.summary()

#steps_per_epoch=tf.ceil(len(all_image_paths)/BATCH_SIZE).numpy()
model.load_weights('resnet_40weight.h5')
rand_seed = random.randint(0,tuple_of_data)
test_image_path = all_image_paths[rand_seed]                  # the file to be test
list_img = []
list_img.append(test_image_path)

print('test_image_path', test_image_path)

test_path = tf.data.Dataset.from_tensor_slices(list_img )
test_image = test_path.map(load_and_preprocess_image , num_parallel_calls=AUTOTUNE)
test_image = test_image.batch(1)

result = model.predict(test_image)


bbox_result = result[0][0]
lm_result = result[1][0]


print('-------------------------------------------------------------------------')
print('\n\n\n')
print('test_image_path', test_image_path)
print('bbox result =======>> :   ', bbox_result)
print('lm result =======>> :   ', lm_result)

bbox_result[0] = (bbox_result[0]/128)*178
bbox_result[1] = (bbox_result[1]/128)*218
bbox_result[2] = (bbox_result[2]/128)*178
bbox_result[3] = (bbox_result[3]/128)*218
for i in range(0,5):
	lm_result[2*i] = (lm_result[2*i]/128)*178
	lm_result[2*i+1] = (lm_result[2*i+1]/128)*218

exact_bbox = []
#print('lm result =======>> :   ', lm_result)
with open('10000img.json', 'r') as f:
	data = json.load(f)	
	print('load json: ' ,data[rand_seed])
	exact_bbox = 	data[rand_seed][1:5]
print('exact_bbox' , exact_bbox)
exact_bbox[0] = (exact_bbox[0]/128)*178
exact_bbox[1] = (exact_bbox[1]/128)*218
exact_bbox[2] = (exact_bbox[2]/128)*178
exact_bbox[3] = (exact_bbox[3]/128)*218


from PIL import Image , ImageDraw
im = Image.open(test_image_path) 
drawSurface = ImageDraw.Draw(im)
d = 20
drawSurface.line(( (exact_bbox[0],exact_bbox[1]) , (exact_bbox[0]+exact_bbox[2],exact_bbox[1]) , (exact_bbox[0]+exact_bbox[2],exact_bbox[1]+exact_bbox[3]) , (exact_bbox[0],exact_bbox[1]+exact_bbox[3]) , (exact_bbox[0],exact_bbox[1]) ),fill = (0,255,0), width = 5)
drawSurface.line(( (bbox_result[0],bbox_result[1]) , (bbox_result[0]+bbox_result[2],bbox_result[1]) , (bbox_result[0]+bbox_result[2],bbox_result[1]+bbox_result[3]) , (bbox_result[0],bbox_result[1]+bbox_result[3]) , (bbox_result[0],bbox_result[1]) ),fill = (255,0,0), width = 5)
'''
for i in range(0,5):

	drawSurface.ellipse((lm_result[2*i]+ d,lm_result[2*i+1]- d,lm_result[2*i]+ d,lm_result[2*i+1]-d),fill = (0, 255, 0))
'''

im.show()
del drawSurface
#im.save('test.jpeg',format='jpeg')

