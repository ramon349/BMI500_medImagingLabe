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
GCS_PATH = "/labs/colab/BMI500-Fall2020/"

BATCH_SIZE = 16 * 10
IMAGE_SIZE = [180, 180]
EPOCHS = 25 
train_dataset_path = os.path.abspath(os.path.join(GCS_PATH,'chest_xray/train/*/*'))
print(train_dataset_path)
test_dataset_path = os.path.abspath(os.path.join(GCS_PATH,'chest_xray/val/*/*'))
filenames = tf.io.gfile.glob(train_dataset_path) 
filenames.extend(tf.io.gfile.glob(test_dataset_path) ) 

train_filenames, val_filenames = train_test_split(filenames, test_size=0.2)
print(tf.__version__) 
COUNT_NORMAL = len([filename for filename in train_filenames if "NORMAL" in filename])
print("Normal images count in training set: " + str(COUNT_NORMAL))

COUNT_PNEUMONIA = len([filename for filename in train_filenames if "PNEUMONIA" in filename])
print("Pneumonia images count in training set: " + str(COUNT_PNEUMONIA))

train_list_ds = tf.data.Dataset.from_tensor_slices(train_filenames)
val_list_ds = tf.data.Dataset.from_tensor_slices(val_filenames)


train_ds = train_list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
train_ds = prepare_for_training(train_ds)
val_ds = val_list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.batch(BATCH_SIZE)


test_list_ds = tf.data.Dataset.list_files(str(GCS_PATH + '/chest_xray/test/*/*'))
TEST_IMAGE_COUNT = tf.data.experimental.cardinality(test_list_ds).numpy()
test_ds = test_list_ds.map(process_path, num_parallel_calls=AUTOTUNE) 
test_ds = test_ds.batch(BATCH_SIZE)

TRAIN_IMG_COUNT = tf.data.experimental.cardinality(train_list_ds).numpy()
print("Training images count: " + str(TRAIN_IMG_COUNT))

VAL_IMG_COUNT = tf.data.experimental.cardinality(val_list_ds).numpy()
print("Validating images count: " + str(VAL_IMG_COUNT))


weight_for_0 = (1 / COUNT_NORMAL)*(TRAIN_IMG_COUNT)/2.0 
weight_for_1 = (1 / COUNT_PNEUMONIA)*(TRAIN_IMG_COUNT)/2.0

class_weight = {0: weight_for_0, 1: weight_for_1}

print('Weight for class 0: {:.2f}'.format(weight_for_0))
print('Weight for class 1: {:.2f}'.format(weight_for_1))

initial_bias = np.log([COUNT_PNEUMONIA/COUNT_NORMAL])




model = build_model(IMAGE_SIZE)

METRICS = [
    'accuracy',
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
    tf.keras.metrics.AUC(name='auc', curve='ROC'),
    tf.keras.metrics.AUC(name='pr', curve='PR')
]

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=METRICS
)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath='/labs/colab/BMI500-Fall2020/BMI500_jjeong/leaky_model.cpt',
                                                 save_weights_only=True,
                                                 verbose=1)

history = model.fit(
    train_ds,
    steps_per_epoch=TRAIN_IMG_COUNT // BATCH_SIZE,
    epochs=25,
    validation_data=val_ds,
    validation_steps=VAL_IMG_COUNT // BATCH_SIZE,
    class_weight=class_weight,
    callbacks=[cp_callback]
)

fig, ax = plt.subplots(1, 4, figsize=(20, 3))
ax = ax.ravel()
for i, met in enumerate(['precision', 'recall', 'accuracy', 'auc', 'pr', 'loss']):
    ax[i].plot(history.history[met])
    ax[i].plot(history.history['val_' + met])
    ax[i].set_title('Model {}'.format(met))
    ax[i].set_xlabel('epochs')
    ax[i].set_ylabel(met)
    ax[i].legend(['train', 'val'])
plt.savefig('leaky.png')
