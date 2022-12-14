import os
import random
from numpy.random import seed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import pickle

from scipy import signal
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
import mne.io

import aux_functions
from generate_keys import generate_data_keys_subsample
from generator_ds import SegmentedGenerator
import ChronoNet

random_seed = 1
seed(random_seed)
tf.random.set_seed(random_seed)


dataPath = 'D:\Datasets\SeizeIT1\Data' # Path to data

annotation_type = 'a1'  # Annotation file type (a1 or a2)

model_name = 'ChronoNet_test'   # name of the model


# Configuration for the data generator, model and training routine:

batch_size = 128        # batch size
frame = 2               # segment window size in seconds
stride = 1              # stride between segments in seconds
stride_s = 0.5          # specific stride between seizure segments in seconds for upsampling method
factor = 5              # balancing factor between classes (N non-seiz segments = factor * N seiz segments)
weights = {0: 1, 1: 5}  # class weights
lr = 0.0001             # learning rate
act = 'elu'             # activation function
dropoutRate = 0.5       # dropout rate
nb_epochs = 100         # Nr of epochs for training

load_offline = True

normalize = True


# Different possible montages:

bhe_montage = [('OorLiTop', 'OorReTop'), ('OorLiTop', 'OorLiAchter'), ('OorReTop', 'OorReAchter')]

Longitudinal_montage = [('fp1', 'f7'), ('f7', 't3'), ('t3', 't5'), ('t5', 'o1'),
                        ('fp1', 'f3'), ('f3', 'c3'), ('c3', 'p3'), ('p3', 'o1'),
                        ('fp2', 'f4'), ('f4', 'c4'), ('c4', 'p4'), ('p4', 'o2'),
                        ('fp2', 'f8'), ('f8', 't4'), ('t4', 't6'), ('t6', 'o2')]

Longitudinal_transverse_montage = [('fp1', 'f7'), ('f7', 't3'), ('t3', 't5'), ('t5', 'o1'),
                                   ('fp1', 'f3'), ('f3', 'c3'), ('c3', 'p3'), ('p3', 'o1'),
                                   ('fp2', 'f4'), ('f4', 'c4'), ('c4', 'p4'), ('p4', 'o2'),
                                   ('fp2', 'f8'), ('f8', 't4'), ('t4', 't6'), ('t6', 'o2'),
                                   ('t3', 'c3'), ('c3', 'cz'), ('cz', 'c4'), ('c4', 't4')]

Longitudinal_transverse_montage_2 = [('fp1', 'f7'), ('f7', 't3'), ('t3', 't5'), ('t5', 'o1'),
                                     ('fp1', 'f3'), ('f3', 'c3'), ('c3', 'p3'), ('p3', 'o1'),
                                     ('fp2', 'f4'), ('f4', 'c4'), ('c4', 'p4'), ('p4', 'o2'),
                                     ('fp2', 'f8'), ('f8', 't4'), ('t4', 't6'), ('t6', 'o2'),
                                     ('a1', 't3'), ('t3', 'c3'), ('c3', 'cz'), ('cz', 'c4'), ('c4', 't4'), ('t4', 'a2')]


montage = bhe_montage


#######################################################################################################################
### Divide train/val/test ###
#######################################################################################################################

patient_list = [x for x in os.listdir(dataPath) if 'P_ID' in x]

train_pats_list = random.sample(patient_list, int(np.round(len(patient_list)*0.7)))
val_pats_list = random.sample(patient_list, int(np.round(len(patient_list)*0.15)))
test_pats_list = random.sample(patient_list, int(np.round(len(patient_list)-len(train_pats_list)-len(val_pats_list))))


#######################################################################################################################
### Load Passive Data ###
#######################################################################################################################

train_file_list = []
train_montages = []
val_file_list = []
val_montages = []
test_file_list = []
test_montages = []

for pat in patient_list:
    rec_list = [x for x in os.listdir(os.path.join(dataPath, pat)) if '.edf' in x]

    seiz_hem = aux_functions.get_hemisphere(os.path.join(dataPath, pat))

    save_mont = True
    for rec in rec_list:
        print(rec)
        rec_path = os.path.join(dataPath, pat, rec)

        channels = set([x for sub in montage for x in sub])
        add_on_mont = ''

        raw = mne.io.read_raw_edf(rec_path, include=channels, preload=False, verbose=False)
        if not raw.ch_names:
            add_on_mont = 'EEG '
            channels = [add_on_mont + ch for ch in channels]
            raw = mne.io.read_raw_edf(rec_path, include=channels, preload=False, verbose=False)
        

        if seiz_hem == 'left':
            rec_montage = [[add_on_mont + 'OorLiTop', add_on_mont + 'OorLiAchter'], [add_on_mont + 'OorLiTop', add_on_mont + 'OorReTop']]
        elif seiz_hem == 'right':
            rec_montage = [[add_on_mont + 'OorReTop', add_on_mont + 'OorReAchter'], [add_on_mont + 'OorLiTop', add_on_mont + 'OorReTop']]
        
        if pat in train_pats_list:
            train_file_list.append(raw)
            train_montages.append(rec_montage)
        elif pat in val_pats_list:
            val_file_list.append(raw)
            val_montages.append(rec_montage)
        elif pat in test_pats_list:
            test_file_list.append(raw)
            test_montages.append(rec_montage)


#######################################################################################################################
### Get segments for training ###
#######################################################################################################################

train_segments = generate_data_keys_subsample(train_file_list, frame, stride, stride_s, factor)
val_segments = generate_data_keys_subsample(val_file_list, frame, stride, stride_s, factor)

#######################################################################################################################
#######################################################################################################################

gen_train = SegmentedGenerator(train_file_list, train_segments[0:32], train_montages, load_offline=True, batch_size=32, shuffle=True)

gen_val = SegmentedGenerator(val_file_list, val_segments[0:32], val_montages, load_offline=True, batch_size=32, shuffle=True)

#######################################################################################################################
#######################################################################################################################

aux_functions.set_gpu()

K.set_image_data_format('channels_last')

#######################################################################################################################
#######################################################################################################################

model = ChronoNet.net()

model.summary()

optimizer = Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, amsgrad=False)

loss = [aux_functions.focal_loss]

metrics = ['accuracy', aux_functions.sens, aux_functions.spec]
cf_metrics = ['accuracy', 'sens', 'spec']

custom_objs = {'focal_loss': aux_functions.focal_loss, 'sens': aux_functions.sens, 'spec': aux_functions.spec}
cf_custom_objs = custom_objs

monitor = 'val_sens'
monitor_mode = 'max'

early_stopping = False
patience = 50

#######################################################################################################################
#######################################################################################################################


if not os.path.exists('Models'):
    os.mkdir('Models')

if not os.path.exists('Models/Callbacks'):
    os.mkdir('Models/Callbacks')

if not os.path.exists('Models/History'):
    os.mkdir('Models/History')

if not os.path.exists('Models/Saved_models'):
    os.mkdir('Models/Saved_models')

if not os.path.exists('Models/Saved_models/Graphs'):
    os.mkdir('Models/Saved_models/Graphs')

if not os.path.exists('Models/Saved_models/Weights'):
    os.mkdir('Models/Saved_models/Weights')

if not os.path.exists('Models/Saved_models/Models'):
    os.mkdir('Models/Saved_models/Models')


def run_training(model):
    cp_model = 'Models/Callbacks/' + model_name

    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=metrics)

    mc = ModelCheckpoint(cp_model,
                         monitor=monitor,
                         verbose=1,
                         save_best_only=True,
                         mode=monitor_mode)

    csv_logger = CSVLogger(os.path.join('Models/History', model_name + '.csv'), append=True)

    if early_stopping:
        es = EarlyStopping(monitor='val_loss',
                           patience=patience,
                           verbose=1,
                           mode='min')


    if early_stopping:
        callbacks_list = [mc, es, csv_logger]
    else:
        callbacks_list = [mc, csv_logger]

    hist = model.fit(gen_train, validation_data=gen_val,
                     epochs=nb_epochs,
                     callbacks=callbacks_list,
                     shuffle=False,
                     verbose=1,
                     class_weight=weights)

    saved_model = load_model(cp_model, custom_objects=custom_objs)

    plt.plot(hist.history['accuracy'])
    plt.plot(hist.history['val_accuracy'])
    plt.title("Model accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Accuracy", "Validation Accuracy"])
    plt.savefig('Models/Saved_models/Graphs/' + model_name + '_acc.png',
                bbox_inches='tight')
    plt.close()

    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title("Model loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Loss", "Validation Loss"])
    plt.savefig('Models/Saved_models/Graphs/' + model_name + '_loss.png',
                bbox_inches='tight')
    plt.close()


    model_json = saved_model.to_json()
    with open("Models/Saved_models/Models/" + model_name + ".json",
              "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    saved_model.save_weights(
        "Models/Saved_models/Weights/" + model_name + ".h5")
    print("Saved model to disk")


start_train = time.time()
data = run_training(model)
end_train = time.time() - start_train

print('Total train duration = ', end_train / 60)

print('============END============')
