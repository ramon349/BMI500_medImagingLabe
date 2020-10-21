
import tensorflow as tf 

def conv_block(filters):
    block = tf.keras.Sequential([
        tf.keras.layers.SeparableConv2D(filters, 3,  padding='same'),
        tf.keras.layers.LeakyReLU(alpha=0.3),
        tf.keras.layers.SeparableConv2D(filters, 3, padding='same'),
        tf.keras.layers.LeakyReLU(alpha=0.3),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D()
    ]
    )
    
    return block

def dense_block(units, dropout_rate):
    block = tf.keras.Sequential([
        tf.keras.layers.Dense(units),
        tf.keras.layers.LeakyReLU(alpha=0.3),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(dropout_rate)
    ])
    
    return block


def build_model(IMAGE_SIZE):
    model = tf.keras.Sequential([ 
        tf.keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
        
        tf.keras.layers.Conv2D(16, 3,  padding='same'),
        tf.keras.layers.Conv2D(16, 3, padding='same'),
        tf.keras.layers.LeakyReLU(alpha=0.3),
        tf.keras.layers.MaxPool2D(),
        
        conv_block(32),
        conv_block(64),
        
        conv_block(128),
        tf.keras.layers.Dropout(0.2),
        
        conv_block(256),
        tf.keras.layers.Dropout(0.2),
        
        tf.keras.layers.Flatten(),
        dense_block(512, 0.7),
        dense_block(128, 0.5),
        dense_block(64, 0.3),
        
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model