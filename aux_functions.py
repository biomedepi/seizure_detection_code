## Custom functions ##
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K
from sklearn.metrics import roc_auc_score
from scipy import signal



def wrangle_tsv_sz1(tsv_path, only_visible_bhe=False):
    """
    Function to process a given .tsv annotation file (from the SeizeIT1 dataset) into a time series of labels.
    tsv_path: Path to .tsv file
    only_visible_bhe: if True, only considers seizures that have criteria "bhe = 1" in the annotation file -> meaning
    the seizure is visible in the bhe channels
    """
    df = pd.read_csv(tsv_path, sep='\t', header=None, names=[0, 1, 2, 3])

    nb_events = df.shape[0] - 9

    events_times = []
    seiz_type = []
    hemisphere = []

    for i in range(nb_events):
        if only_visible_bhe:
            visible_bhe = int(df[3][9 + i][-1])
            if visible_bhe == 1:
                start_sec = int(df[0][9 + i])
                if df[1][9 + i] == 'None':
                    stop_sec = 'None'
                else:
                    stop_sec = int(df[1][9 + i])
                events_times.append([start_sec, stop_sec])
                seiz_type.append(df[2][9 + i])
                hemisphere.append(df[3][9 + i][4])
        else:
            start_sec = int(df[0][9 + i])
            if df[1][9 + i] == 'None':
                stop_sec = 'None'
            else:
                stop_sec = int(df[1][9 + i])
            events_times.append([start_sec, stop_sec])
            seiz_type.append(df[2][9 + i])
            hemisphere.append(df[3][9 + i][4])

    return events_times, seiz_type, hemisphere


def get_hemisphere(pat_path):
    """
    Function to get the hemisphere of the seizure source. Usually, a subject only has seizures with source on the
    same hemisphere. If this is not the case, the hemisphere is set to the one that more seizures were originated
    from.
    pat_path: Path to patient's data folder
    """

    all_files = [x for x in os.listdir(pat_path) if x.endswith('.tsv')]

    count_left = 0
    count_right = 0
    for f in all_files:
        [_, _, hemisphere] = wrangle_tsv_sz1(os.path.join(pat_path, f))

        for i in range(len(hemisphere)):
            if hemisphere[i] == 'L':
                count_left += 1
            elif hemisphere[i] == 'R':
                count_right += 1
                
    if count_right > count_left:
        main_event_hem = 'right'
    else:
        main_event_hem = 'left'

    return main_event_hem


def pre_process_ch(ch_data, fs):

    fs_resamp = 200
    
    b, a = signal.butter(4, 0.5/(fs/2), 'high')
    ch_data = signal.filtfilt(b, a, ch_data)

    b, a = signal.butter(4, 60/(fs/2), 'low')
    ch_data = signal.filtfilt(b, a, ch_data)

    b, a = signal.butter(4, [49.5/(fs/2), 50.5/(fs/2)], 'bandstop')
    ch_data = signal.filtfilt(b, a, ch_data)

    ch_data = signal.resample(ch_data, int(fs_resamp*len(ch_data)/fs))

    return ch_data, fs_resamp



def apply_montage(raw, montage, normalize):

    ch_focal = raw.get_data(montage[0][0])[0] - raw.get_data(montage[0][1])[0]
    ch_cross = raw.get_data(montage[1][0])[0] - raw.get_data(montage[1][1])[0]

    ch_focal, fs_resamp = pre_process_ch(ch_focal, raw.info['sfreq'])
    ch_cross, _ = pre_process_ch(ch_cross, raw.info['sfreq'])
    
    if normalize:
        ch_focal = (ch_focal - np.mean(ch_focal))/np.std(ch_focal)
        ch_cross = (ch_cross - np.mean(ch_cross))/np.std(ch_cross)

    return [ch_focal, ch_cross], fs_resamp


def set_gpu():
    """
    Detects GPUs and (currently) sets automatic memory growth
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')

    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')

            print(len(gpus), 'Physical GPUs, ', len(logical_gpus), 'Logical GPUs')
        except RuntimeError as e:
            print(e)


def focal_loss(y_true, y_pred):
    """
    :param y_true: A tensor of the same shape as `y_pred`
    :param y_pred:  A tensor resulting from a sigmoid
    :return: Output tensor.
    """
    gamma = 2.0
    alpha = 0.25

    y_true = tf.cast(y_true, tf.float32)
    # Define epsilon so that the back-propagation will not result in NaN for 0 divisor case
    epsilon = K.epsilon()
    # Add the epsilon to prediction value
    # y_pred = y_pred + epsilon
    # Clip the prediciton value
    y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
    # Calculate p_t
    p_t = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
    # Calculate alpha_t
    alpha_factor = K.ones_like(y_true) * alpha
    alpha_t = tf.where(K.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
    # Calculate cross entropy
    cross_entropy = -K.log(p_t)
    weight = alpha_t * K.pow((1 - p_t), gamma)
    # Calculate focal loss
    loss = weight * cross_entropy
    # Sum the losses in mini_batch
    loss = K.mean(K.sum(loss, axis=1))
    return loss


def aucc(y_true, y_pred):
    return tf.py_function(roc_auc_score, (y_true, y_pred), tf.double)


def sens(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true[:, 1] * y_pred[:, 1], 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true[:, 1], 0, 1)))
    return true_positives / (possible_positives + K.epsilon())


def spec(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1 - y_true[:, 1]) * (1 - y_pred[:, 1]), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1 - y_true[:, 1], 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())