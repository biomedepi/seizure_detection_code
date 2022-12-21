import os
from numpy.random import seed
import numpy as np
import pandas as pd
import tensorflow as tf
import h5py

import aux_functions
import routines
from generator_ds import SegmentedGenerator
from config import Settings

# Random seed set for reproducibility
random_seed = 1
seed(random_seed)
tf.random.set_seed(random_seed)

#######################################################################################################################
### Initialization ###
#######################################################################################################################

data_path = 'Data' # Path to data
config_path = 'Configs' # path to configuration files

annotation_type = 'a1'  # Annotation file type (a1 or a2)

model_name = 'ChronoNet_test'   # name of the model


# Configuration for the data generator, model and training routine:
config = Settings(name=model_name)

config.batch_size = 16         # batch size
config.frame = 2               # segment window size in seconds
config.stride = 1              # stride between segments in seconds
config.stride_s = 0.5          # specific stride between seizure segments in seconds for upsampling method
config.factor = 5              # balancing factor between classes (N non-seiz segments = factor * N seiz segments)
config.class_weights = {0: 1, 1: 5}  # class weights
config.lr = 0.0001             # learning rate
config.nr_epochs = 10         # Nr of epochs for training

# You can choose to save and load a configuration for future experiments
config.save(path=config_path, filename=model_name + '.cfg')

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


#######################################################################################################################
### Divide train/val/test + load passive data ###
#######################################################################################################################

train_file_list, train_montages, val_file_list, val_montages, test_file_list, test_montages = routines.split_sets(data_path, bhe_montage)


#######################################################################################################################
### Get training and validation segments ###
#######################################################################################################################

train_segments = routines.get_data_keys_subsample(train_file_list, config)
val_segments = routines.get_data_keys_subsample(val_file_list, config)


#######################################################################################################################
### Build data generators ###
#######################################################################################################################

gen_train = SegmentedGenerator(train_file_list, train_segments[0:32], train_montages, batch_size=32, shuffle=True)
gen_val = SegmentedGenerator(val_file_list, val_segments[0:32], val_montages, batch_size=32, shuffle=True)


#######################################################################################################################
### Initialize and train model ###
#######################################################################################################################

aux_functions.set_gpu()

routines.train_net(config, gen_train, gen_val)


#######################################################################################################################
### Get model's predictions on test set ###
#######################################################################################################################

file_names, y_preds, y_true = routines.predict_net(test_file_list, test_montages, config)


# Saving predictions
dt_fl = h5py.vlen_dtype(np.dtype('float32'))
dt_int = h5py.vlen_dtype(np.dtype('uint8'))
dt_str = h5py.special_dtype(vlen=str)

if not os.path.exists('Predictions'):
    os.mkdir('Predictions')

with h5py.File(os.path.join('Predictions', config.name + '.h5'), 'w') as f:
    dset_signals = f.create_dataset('predictions', (len(y_preds),), dtype=dt_fl)
    dset_labels = f.create_dataset('labels', (len(y_true),), dtype=dt_int)
    dset_file_names = f.create_dataset('filenames', (len(file_names),), dtype=dt_str)

    for i in range(len(file_names)):
        dset_signals[i] = y_preds[i]
        dset_labels[i] = y_true[i]
        dset_file_names[i] = file_names[i]


#######################################################################################################################
### Get metrics and score ###
#######################################################################################################################

pred_file = 'Predictions/ChronoNet_test.h5'

score, sens_ovlp, FA_epoch = routines.get_metrics_scoring(pred_file)

print('Score: ', str(score))