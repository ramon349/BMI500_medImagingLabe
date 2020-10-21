import re
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from leaky_model_builder import build_model  as leaky_builder 
from model_builder import build_model as reg_builder
from preprocessing import process_path,prepare_for_training
from sklearn.metrics import roc_curve
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
weight_path = ["/labs/colab/BMI500-Fall2020/BMI500_jjeong/leaky_model.cpt.data-00000-of-00001","/labs/colab/BMI500-Fall2020/BMI500_jjeong/base_model.cpt.data-00000-of-00001"] # jason plug in actual paths   first leaky then regular 
model_builders  = [leaky_builder,reg_builder]
print("ROC AUC \t  PR AUC")


for i,w in enumerate(weight_path): 
    model = model_builders[i](IMAGE_SIZE)
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
    y_pred = list() 
    t_labels =list() 
    for img,labels in val_ds.take(-1): 
    #model.load_weights(w) # Jason uncomment this 
        y_pred_keras = model.predict(img).ravel() 
        y_pred.extend(y_pred_keras)
        t_labels.extend(labels) 
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(t_labels,y_pred) 
    plt.plot(fpr_keras,tpr_keras)
    loss,ROC_auc,PR_AUC =  model.evaluate(val_ds,verbose=0) #eliminate verbosity so my output looks preety 
    print("{} \t {}".format(ROC_auc,PR_AUC) )
plt.legend(['leaky','default'])
plt.title("AUC of models") 
plt.xlabel("FPR") 
plt.ylabel("TPR")
plt.savefig("AUC chart")