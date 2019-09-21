import tensorflow as tf
tf.enable_eager_execution()
AUTOTUNE = tf.data.experimental.AUTOTUNE
import pathlib
import os
import random
import json
from os import listdir
from os.path import isfile, join

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
from tensorflow import keras

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
											# # # #											
											#PART 1#
											# # # # 
data_root = pathlib.Path('/home/yiiiiiach/MTCNN_TRY/image_align_celeba/')          #  打開  ./img_of_all.json 、image directory 將所有 landmark 與 bbox 標注 、圖片path 
										   #  儲存到 1. all_image_paths  2. all_image_bbox 3. all_image_landmark 三個list 中

ROOT_DIR = os.getcwd()
align_image_path = os.path.join(ROOT_DIR,'img_align_celeba/')

with open('./10000img.json', 'r') as f:
	data = json.load(f)	
	tuple_of_data = len(data)	
	print('tuple_of_data',tuple_of_data )


all_image_paths = []
all_image_bbox = []
all_image_landmark = []


for i in range(0,tuple_of_data):
	all_image_paths.append(align_image_path + data[i][0]) 
	all_image_bbox.append(data[i][1:5])
	all_image_landmark.append(data[i][5:15])
#random.shuffle(all_image_paths)          C O N S I D E R I N G   T O   U S E
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
											# # # #											
											#PART 2#
											# # # # 
def preprocess_image(image):
  image = tf.image.decode_jpeg(image, channels=3)											
  image = tf.image.resize(image, [48, 48])                             		#宣告能讀取圖片並將像素轉為適合 tensorflow.Data  Class 的格式  的函式
  image /= 255.0  # normalize to [0,1] range

  return image

def load_and_preprocess_image(path):
  image = tf.read_file(path)
  return preprocess_image(image)
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
											# # # #											
											#PART 3#
											# # # # 
import matplotlib.pyplot as plt


path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)               #  tf.data.Dataset.from_tensor_slices(  <list>  ) ==> 將list 中的資料切成片段 以方便以類似串流的方式 丟進模型裡訓練
image_ds = path_ds.map(load_and_preprocess_image , num_parallel_calls=AUTOTUNE) 
								            #  tf.data.Dataset.map(  <function_name>  )  ==> 將 Class Dataset 裡面的資料使用function操作
									    #  tf.cast ( <object> , <dtype> )  ==> 轉換資料格式
all_image_bbox_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_bbox, tf.int64))
all_image_landmark_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_landmark, tf.int64))


label_ds = tf.data.Dataset.zip((all_image_bbox_ds , all_image_landmark_ds))#  tf.data.Dataset.zip(( <Dataset> , <Dataset> , ........... ))   將Dataset 結合在一起 (要注意是否與 model 合適）
image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))                 #  image_label_ds 結構解析 ==>>
									#     [  [image]  , [ bbox ,  landmark ]  ]
									#        輸入         label_1  label_2
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
											# # # #											
											#PART 4#
											# # # # 
print('image_ds',image_ds)

BATCH_SIZE = 36
image_label_ds = image_label_ds.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=1000))    # 將 image_label_ds 進行 1. 重排 2. batch化  等預處理
image_label_ds  = image_label_ds.batch(BATCH_SIZE)
image_label_ds = image_label_ds.prefetch(buffer_size=AUTOTUNE)


#img_input = tf.keras.Input(shape=(48,48,3,) )                                                      #  tf.keras.Input( ）宣告 input 的 shape 


model = resnet50( (48,48,3) )
#model = tf.keras.models.Model(inputs=[img_input], outputs=[bbox,landmark])                       #宣告model 的輸入與輸出 對應到哪個 layer  , 其中 tf.keras.models.Model() 的參數 要與輸入輸出layer 的name 一樣 

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
											# # # #											
											#PART 5#
											# # # #
 
                  
#  Optimizer() change ??   tf.train.AdamOptimizer()  OR   tf.keras.optimizers.RMSprop(0.001)   #因為輸出為 regression 故使用 RMSoptimizer      
model.compile(optimizer=tf.keras.optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.00001),
              loss={'bbox':'mean_squared_error','landmark':'mean_squared_error'},loss_weights={'bbox':1 , 'landmark':0.5 },metrics=["accuracy"])
											      # 宣告 每個輸出使用的loss function 以及各自權重 、 可以發現到其 資料結構與當初 宣告 image_label_ds 結構呼應
model.summary()										      # echo model 的架構與參數量
#model.load_weights('resnet_final_weight.h5')
#steps_per_epoch=tf.ceil(len(all_image_paths)/BATCH_SIZE).numpy()
for times in range(0 , int(epochs/10)):
	model.fit(image_label_ds, epochs=10, steps_per_epoch=100)                                    # 丟入 model 訓練
	
	model.save_weights('resnet_' + times*10 + 'weight.h5')   #save our weight


model.save_weights('resnet_final_weight.h5')
