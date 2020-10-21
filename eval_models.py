import re
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from leaky_model_builder import build_model 
from preprocessing import process_path,prepare_for_training

def show_batch( iterator):
    plt.figure(figsize=(10,10))
    for n,(image,label) in enumerate( iterator.take(25)):
        ax = plt.subplot(5,5,n+1)
        plt.imshow(image)
        if label:
            plt.title("PNEUMONIA")
        else:
            plt.title("NORMAL")
        plt.axis("off")
    plt.show()

AUTOTUNE = tf.data.experimental.AUTOTUNE 

GCS_PATH = "/Users/ramoncorrea/Desktop/BMI599/correa_imaging_experiments"
#GCS_PATH = "/labs/colab/BMI500-Fall2020/"

BATCH_SIZE = 16 * 10
IMAGE_SIZE = [180, 180]
EPOCHS = 25 
train_dataset_path = os.path.abspath(os.path.join(GCS_PATH,'chest_xray/train/*/*'))
print(train_dataset_path)
test_dataset_path = os.path.abspath(os.path.join(GCS_PATH,'chest_xray/val/*/*'))
filenames = tf.io.gfile.glob(train_dataset_path) 
filenames.extend(tf.io.gfile.glob(test_dataset_path) ) 
train_filenames, val_filenames = train_test_split(filenames, test_size=0.2 )

val_list_ds = tf.data.Dataset.from_tensor_slices(val_filenames)
val_ds = val_list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.batch(BATCH_SIZE)
VAL_IMG_COUNT = tf.data.experimental.cardinality(val_list_ds).numpy()
print("Validating images count: " + str(VAL_IMG_COUNT))
model = build_model(IMAGE_SIZE)
METRICS = [
    tf.keras.metrics.AUC(name='auc', curve='ROC'),
    tf.keras.metrics.AUC(name='pr', curve='PR')
]

#load model as usual 
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=METRICS
)
weight_path = ["path1","path2"] # jason plug in actual paths 
print("ROC AUC \t  PR AUC")
for w in weight_path: 
    #model.load_weights(w) # Jason uncomment this 
    loss,ROC_auc,PR_AUC =  model.evaluate(val_ds,verbose=0) #eliminate verbosity so my output looks preety 
    print("{} \t {}".format(ROC_auc,PR_AUC) )