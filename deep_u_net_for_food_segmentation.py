'''
Alex Costanzino, Marco Costante
MSc student in Artificial Intelligence
@ Alma Mater Studiorum, University of Bologna
March, 2021
'''

''' U-net parameters'''
FILTER = 32

''' Input layer '''
inputs = tf.keras.layers.Input((IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS))
s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs) # Normalization

''' Contractive path '''
### Layer 1
c1 = tf.keras.layers.Conv2D(FILTER, (3, 3),
                            kernel_initializer = 'he_normal',
                            padding = 'same')(s)

c1 = tf.keras.layers.BatchNormalization()(c1) # Batch normalization instead of dropout (note: I had to split the conv2D and ReLU)

c1 = tf.keras.layers.Activation('selu')(c1) 

c1 = tf.keras.layers.Conv2D(FILTER, (3, 3),
                            activation = 'selu',
                            kernel_initializer = 'he_normal',
                            padding = 'same')(c1)

p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

### Layer 2
c2 = tf.keras.layers.Conv2D(FILTER*2, (3, 3),
                            kernel_initializer = 'he_normal',
                            padding = 'same')(p1)

c2 = tf.keras.layers.BatchNormalization()(c2) # Batch normalization instead of dropout (note: I had to split the conv2D and ReLU)

c2 = tf.keras.layers.Activation('selu')(c2)

c2 = tf.keras.layers.Conv2D(FILTER*2, (3, 3),
                            activation = 'selu',
                            kernel_initializer = 'he_normal',
                            padding = 'same')(c2)

p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

### Layer 3
c3 = tf.keras.layers.Conv2D(FILTER*4, (3, 3),
                            kernel_initializer = 'he_normal',
                            padding = 'same')(p2)

c3 = tf.keras.layers.BatchNormalization()(c3) # Batch normalization instead of dropout (note: I had to split the conv2D and ReLU)

c3 = tf.keras.layers.Activation('selu')(c3)

c3 = tf.keras.layers.Conv2D(FILTER*4, (3, 3),
                            activation = 'selu',
                            kernel_initializer = 'he_normal',
                            padding = 'same')(c3)

p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

### Layer 4
c4 = tf.keras.layers.Conv2D(FILTER*8, (3, 3),
                            kernel_initializer = 'he_normal',
                            padding = 'same')(p3)

c4 = tf.keras.layers.BatchNormalization()(c4) # Batch normalization instead of dropout (note: I had to split the conv2D and ReLU)

c4 = tf.keras.layers.Activation('selu')(c4)

c4 = tf.keras.layers.Conv2D(FILTER*8, (3, 3),
                            activation = 'selu',
                            kernel_initializer = 'he_normal',
                            padding = 'same')(c4)

p4 = tf.keras.layers.MaxPooling2D((2, 2))(c4)

### Layer 5
c5 = tf.keras.layers.Conv2D(FILTER*16, (3, 3),
                            kernel_initializer = 'he_normal',
                            padding = 'same')(p4)

c5 = tf.keras.layers.BatchNormalization()(c5) # Batch normalization instead of dropout (note: I had to split the conv2D and ReLU)

c5 = tf.keras.layers.Activation('selu')(c5)

c5 = tf.keras.layers.Conv2D(FILTER*16, (3, 3),
                            activation = 'selu',
                            kernel_initializer = 'he_normal',
                            padding = 'same')(c5)

p5 = tf.keras.layers.MaxPooling2D((2, 2))(c5)

### Layer 6
c6 = tf.keras.layers.Conv2D(FILTER*32, (3, 3),
                            kernel_initializer = 'he_normal',
                            padding = 'same')(p5)

c6 = tf.keras.layers.BatchNormalization()(c6) # Batch normalization instead of dropout (note: I had to split the conv2D and ReLU)

c6 = tf.keras.layers.Activation('selu')(c6)

c6 = tf.keras.layers.Conv2D(FILTER*32, (3, 3),
                            activation = 'selu',
                            kernel_initializer = 'he_normal',
                            padding = 'same')(c6)

''' Expansive path '''
# Layer 7
u7 = tf.keras.layers.Conv2DTranspose(FILTER*16, (2, 2),
                                     strides = (2, 2),
                                     padding = 'same')(c6)

u7 = tf.keras.layers.concatenate([u7, c5])

c7 = tf.keras.layers.Conv2D(FILTER*16, (3, 3),
                            kernel_initializer = 'he_normal',
                            padding = 'same')(u7)

c7 = tf.keras.layers.BatchNormalization()(c7) # Batch normalization instead of dropout (note: I had to split the conv2D and ReLU)

c7 = tf.keras.layers.Activation('selu')(c7)

c7 = tf.keras.layers.Conv2D(FILTER*16, (3, 3),
                            activation = 'selu',
                            kernel_initializer = 'he_normal',
                            padding = 'same')(c7)

# Layer 8
u8 = tf.keras.layers.Conv2DTranspose(FILTER*8, (2, 2),
                                     strides = (2, 2),
                                     padding = 'same')(c7)

u8 = tf.keras.layers.concatenate([u8, c4])

c8 = tf.keras.layers.Conv2D(FILTER*8, (3, 3),
                            kernel_initializer = 'he_normal',
                            padding = 'same')(u8)

c8 = tf.keras.layers.BatchNormalization()(c8) # Batch normalization instead of dropout (note: I had to split the conv2D and ReLU)

c8 = tf.keras.layers.Activation('selu')(c8)

c8 = tf.keras.layers.Conv2D(FILTER*8, (3, 3),
                            activation = 'selu',
                            kernel_initializer = 'he_normal',
                            padding = 'same')(c8)

# Layer 9
u9 = tf.keras.layers.Conv2DTranspose(FILTER*4, (2, 2),
                                     strides = (2, 2),
                                     padding = 'same')(c8)

u9 = tf.keras.layers.concatenate([u9, c3])

c9 = tf.keras.layers.Conv2D(FILTER*4, (3, 3),
                            kernel_initializer = 'he_normal',
                            padding = 'same')(u9)

c9 = tf.keras.layers.BatchNormalization()(c9) # Batch normalization instead of dropout (note: I had to split the conv2D and ReLU)

c9 = tf.keras.layers.Activation('selu')(c9)

c9 = tf.keras.layers.Conv2D(FILTER*4, (3, 3),
                            activation = 'selu',
                            kernel_initializer = 'he_normal',
                            padding = 'same')(c9)

# Layer 10
u10 = tf.keras.layers.Conv2DTranspose(FILTER*2, (2, 2),
                                      strides = (2, 2),
                                      padding = 'same')(c9)

u10 = tf.keras.layers.concatenate([u10, c2])

c10 = tf.keras.layers.Conv2D(FILTER*2, (3, 3),
                             kernel_initializer = 'he_normal',
                             padding = 'same')(u10)

c10 = tf.keras.layers.BatchNormalization()(c10) # Batch normalization instead of dropout (note: I had to split the conv2D and ReLU)

c10 = tf.keras.layers.Activation('selu')(c10)

c10 = tf.keras.layers.Conv2D(FILTER*2, (3, 3),
                             activation = 'selu',
                             kernel_initializer = 'he_normal',
                             padding = 'same')(c10)

# Layer 11
u11 = tf.keras.layers.Conv2DTranspose(FILTER, (2, 2),
                                      strides = (2, 2),
                                      padding = 'same')(c10)

u11 = tf.keras.layers.concatenate([u11, c1], axis = 3)

c11 = tf.keras.layers.Conv2D(FILTER, (3, 3),
                             kernel_initializer = 'he_normal',
                             padding = 'same')(u11)

c11 = tf.keras.layers.BatchNormalization()(c11) # Batch normalization instead of dropout (note: I had to split the conv2D and ReLU)

c11 = tf.keras.layers.Activation('selu')(c11) # Dropout

c11 = tf.keras.layers.Conv2D(FILTER, (3, 3),
                             activation = 'selu',
                             kernel_initializer = 'he_normal',
                             padding = 'same')(c11)

''' Output layer '''
outputs = tf.keras.layers.Conv2D(CLASSES, (1, 1),
                                 activation = 'softmax')(c11)

''' Model building '''
model = tf.keras.Model(inputs = [inputs], outputs = [outputs])
model.summary()