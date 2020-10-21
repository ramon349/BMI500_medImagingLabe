import tensorflow as tf 
# m This code is taken from the kaggle competiion example notebook 


#here we build a traditional convolution block using batchnormalization and maxppoooling 
def conv_block(filters):
    block = tf.keras.Sequential([
        tf.keras.layers.SeparableConv2D(filters, 3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D()
    ]
    )
    
    return block
#estbalishes a dense block to be built with batchnormalziaiton 
def dense_block(units, dropout_rate):
    block = tf.keras.Sequential([
        tf.keras.layers.Dense(units, activation='relu'),
        tf.keras.layers.BatchNormalization(),
    ])
    
    return block

#the actual model to be built 
def build_model(IMAGE_SIZE):
    model = tf.keras.Sequential([ 
        tf.keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
        tf.keras.layers.Conv2D(8, 3, activation='relu', padding='same'),
        tf.keras.layers.MaxPool2D(),
        conv_block(64),
        tf.keras.layers.Flatten(),
        dense_block(32, 0.3),
        
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model