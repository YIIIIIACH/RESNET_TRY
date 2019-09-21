import tensorflow as tf

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
    
    # Create model.
    return tf.keras.models.Model(inputs=[img_input], outputs=[bbox], name='resnet50')

