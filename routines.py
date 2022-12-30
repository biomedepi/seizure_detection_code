import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
import mne.io
import matplotlib.pyplot as plt
import h5py

import aux_functions
import ChronoNet
from generator_ds import SegmentedGenerator



def split_sets(data_path, montage):
    ''' Splits the patients into training, validation and test sets

    Args:
        data_path: the path where the data is stored
        montage: a montage list (see examples in the main.py script) - this function was made with the
                 bhe_montage in mind. Adapt for other montages.
    
    Returns:
        train_file_list: a list of raw instances (mne package) of each recording for the training set
        train_montages: a list of montages associated to each recording (index corresponds to the file list)
        (analogous to the other returning items for the validation and test sets)
    '''
    ### Divide train/val/test ###

    patient_list = [x for x in os.listdir(data_path) if 'P_ID' in x]

    train_pats_list = random.sample(patient_list, int(np.round(len(patient_list)*0.75)))
    val_pats_list = random.sample(patient_list, int(np.round(len(patient_list)*0.15)))
    test_pats_list = random.sample(patient_list, int(np.round(len(patient_list)-len(train_pats_list)-len(val_pats_list))))

    ### Load Passive Data ###

    train_file_list = []
    train_montages = []
    val_file_list = []
    val_montages = []
    test_file_list = []
    test_montages = []

    for pat in patient_list:
        rec_list = [x for x in os.listdir(os.path.join(data_path, pat)) if '.edf' in x]

        seiz_hem = aux_functions.get_hemisphere(os.path.join(data_path, pat))

        for rec in rec_list:
            rec_path = os.path.join(data_path, pat, rec)

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

    return train_file_list, train_montages, val_file_list, val_montages, test_file_list, test_montages



def get_data_keys_subsample(file_list, config):
    ''' Create the list of segment keys for the data generator via the subsampling method. Each key [1x4]
    contains the index of the recording within the file_list, the start and stop times of the segment in
    seconds and the label of the segment.

    Args:
        file_list: a list of raw instances (mne package) of the recordings
        config: configuration object containing the parameters, namely the window frame, stride for non-
                -seizure segments, stride for seizure segments (different since an upsampling strategy
                was used) and a factor related to the number of non-seizure segments to include (N non-
                seizure segments = factor * N seizure segments)
    
    Returns:
        segments_S: a list of keys
    '''
    segments_S = []
    segments_NS = []

    for i, file in enumerate(file_list):
        # Load annotation file
        rec_path = file.filenames[0]
        ann_path_S = rec_path[0:-4] + '_a2.tsv'
        ann_path_NS = rec_path[0:-4] + '_a1.tsv'

        [events_times_S, _, _] = aux_functions.wrangle_tsv_sz1(ann_path_S, only_visible_bhe=True)
        [events_times_NS, _, _] = aux_functions.wrangle_tsv_sz1(ann_path_NS, only_visible_bhe=False)

        if not events_times_S and not events_times_NS:
            n_segs =  int(np.floor((np.floor(file.tmax) - config.frame)/config.stride))
            seg_start = np.arange(0, n_segs)*config.stride
            seg_stop = seg_start + config.frame

            segments_NS.extend(np.column_stack(([i]*n_segs, seg_start, seg_stop, np.zeros(n_segs))))
        else:
            for e, ev in enumerate(events_times_S):
                n_segs = int(np.floor((ev[1] - ev[0])/config.stride_s)-1)
                seg_start = np.arange(0, n_segs)*config.stride_s + ev[0]
                seg_stop = seg_start + config.frame
                segments_S.extend(np.column_stack(([i]*n_segs, seg_start, seg_stop, np.ones(n_segs))))

            for e, ev in enumerate(events_times_NS):
                if ev[1] == 'None':
                    ev[1] = ev[0] + 10

                if e == 0:
                    n_segs = int(np.floor((ev[0])/config.stride)-1)
                    seg_start = np.arange(0, n_segs)*config.stride
                    seg_stop = seg_start + config.frame
                    segments_NS.extend(np.column_stack(([i]*n_segs, seg_start, seg_stop, np.zeros(n_segs))))
                else:
                    n_segs = int(np.floor((ev[0] - events_times_NS[e-1][1])/config.stride)-1)
                    seg_start = np.arange(0, n_segs)*config.stride + events_times_NS[e-1][1]
                    seg_stop = seg_start + config.frame
                    segments_NS.extend(np.column_stack(([i]*n_segs, seg_start, seg_stop, np.zeros(n_segs))))
                if e == len(events_times_NS)-1:
                    n_segs = int(np.floor((np.floor(file.tmax) - ev[1])/config.stride)-1)
                    seg_start = np.arange(0, n_segs)*config.stride + ev[1]
                    seg_stop = seg_start + config.frame
                    segments_NS.extend(np.column_stack(([i]*n_segs, seg_start, seg_stop, np.zeros(n_segs))))

    segments_S.extend(random.sample(segments_NS, config.factor*len(segments_S)))
    random.shuffle(segments_S)

    return segments_S



def get_data_keys_sequential(file_list, config):
    ''' Create the list of sequential segment keys for the data generator. Each key [1x4] contains the
    index of the recording within the file_list, the start and stop times of the segment in seconds and
    the label of the segment. The segments are consecutive, with config.frame length (in seconds) without
    overlap -> these are the parameters used in the test set for the challenge.

    Args:
        file_list: a list of raw instances (mne package) of the recordings
        config: configuration object containing the parameters, namely the window frame and stride.
    
    Returns:
        segments: a list of keys
    '''
    segments = []

    for i, file in enumerate(file_list):
        # Load annotation file
        rec_path = file.filenames[0]
        ann_path = rec_path[0:-4] + '_a1.tsv'

        [events_times, _, _] = aux_functions.wrangle_tsv_sz1(ann_path, only_visible_bhe=False)

        if not events_times:
            n_segs =  int(np.floor(np.floor(file.tmax)/config.frame))
            seg_start = np.arange(0, n_segs)*config.frame
            seg_stop = seg_start + config.frame

            segments.extend(np.column_stack(([i]*n_segs, seg_start, seg_stop, np.zeros(n_segs))))
        else:
            for e, ev in enumerate(events_times):
                if ev[1] == 'None':
                    ev[1] = ev[0] + 10

                if e == 0:
                    n_segs = int(np.floor((ev[0])/config.frame))
                    seg_start = np.arange(0, n_segs)*config.frame
                    seg_stop = seg_start + config.frame
                    segments.extend(np.column_stack(([i]*n_segs, seg_start, seg_stop, np.zeros(n_segs))))

                    n_seiz_segs = int(np.floor((ev[1]-segments[-1][2])/config.frame))
                    seg_start_seiz = (np.arange(0, n_seiz_segs))*config.frame + segments[-1][2]
                    seg_stop_seiz = seg_start_seiz + config.frame
                    segments.extend(np.column_stack(([i]*n_seiz_segs, seg_start_seiz, seg_stop_seiz, np.ones(n_seiz_segs))))
                else:
                    n_segs = int(np.floor((ev[0] - events_times[e-1][1])/config.frame))
                    seg_start = np.arange(0, n_segs)*config.frame + events_times[e-1][1]
                    seg_stop = seg_start + config.frame
                    segments.extend(np.column_stack(([i]*n_segs, seg_start, seg_stop, np.zeros(n_segs))))

                    n_seiz_segs = int(np.floor((ev[1]-segments[-1][2])/config.frame))
                    seg_start_seiz = (np.arange(0, n_seiz_segs))*config.frame + segments[-1][2]
                    seg_stop_seiz = seg_start_seiz + config.frame
                    segments.extend(np.column_stack(([i]*n_seiz_segs, seg_start_seiz, seg_stop_seiz, np.ones(n_seiz_segs))))
                if e == len(events_times)-1:
                    n_segs = int(np.floor((np.floor(file.tmax) - segments[-1][2])/config.frame))
                    seg_start = np.arange(0, n_segs)*config.frame + segments[-1][2]
                    seg_stop = seg_start + config.frame
                    segments.extend(np.column_stack(([i]*n_segs, seg_start, seg_stop, np.zeros(n_segs))))

    return segments



def train_net(config, gen_train, gen_val):
    ''' Routine for training the model. This routine saves the trained model in various formats, the
    training history and learning curves. It uses an Adam optimizer, focal loss and tracks the validation
    sensitivity as the metric to save the best model. The saved metrics in the history are the training 
    and validation accuracy, sensitivity and specificity.

    Args:
        config: a configuration object with the defined parameters
        gen_train: data generator with the training data
        gen_val: data generator with the validation data
    '''

    K.set_image_data_format('channels_last') 

    model = ChronoNet.net()

    model.summary()

    optimizer = Adam(learning_rate=config.lr, beta_1=0.9, beta_2=0.999, amsgrad=False)

    loss = [aux_functions.focal_loss]

    metrics = ['accuracy', aux_functions.sens, aux_functions.spec]

    custom_objs = {'focal_loss': aux_functions.focal_loss, 'sens': aux_functions.sens, 'spec': aux_functions.spec}

    monitor = 'val_sens'
    monitor_mode = 'max'

    early_stopping = False
    patience = 50


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


    cp_model = 'Models/Callbacks/' + config.name

    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=metrics)

    mc = ModelCheckpoint(cp_model,
                         monitor=monitor,
                         verbose=1,
                         save_best_only=True,
                         mode=monitor_mode)

    csv_logger = CSVLogger(os.path.join('Models/History', config.name + '.csv'), append=True)

    if early_stopping:
        es = EarlyStopping(monitor=monitor,
                           patience=patience,
                           verbose=1,
                           mode='min')

    if early_stopping:
        callbacks_list = [mc, es, csv_logger]
    else:
        callbacks_list = [mc, csv_logger]

    hist = model.fit(gen_train, validation_data=gen_val,
                     epochs=config.nr_epochs,
                     callbacks=callbacks_list,
                     shuffle=False,
                     verbose=1,
                     class_weight=config.class_weights)

    saved_model = load_model(cp_model, custom_objects=custom_objs)

    plt.plot(hist.history['accuracy'])
    plt.plot(hist.history['val_accuracy'])
    plt.title("Model accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Accuracy", "Validation Accuracy"])
    plt.savefig('Models/Saved_models/Graphs/' + config.name + '_acc.png',
                bbox_inches='tight')
    plt.close()

    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title("Model loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Loss", "Validation Loss"])
    plt.savefig('Models/Saved_models/Graphs/' + config.name + '_loss.png',
                bbox_inches='tight')
    plt.close()


    model_json = saved_model.to_json()
    with open("Models/Saved_models/Models/" + config.name + ".json",
              "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    saved_model.save_weights(
        "Models/Saved_models/Weights/" + config.name + ".h5")
    print("Saved model to disk")

    print('============ END ============')



def predict_net(file_list, montages, config):
    ''' Routine to obtain predictions from the trained model with the desired configurations.

    Args:
        file_list: list of raw instances (mne package) of the recordings to be classified
        montages: list of montages of the recordings in the file_list
        config: configuration object containing all parameters

    Returns:
        file_names: list of the recording files names that were classified
        y_probas: list of arrays with the probability of seizure occurences (0 to 1) of each consecutive
                  window of the recording. Each array has the same index correspondent to the file_list
        y_true: analogous to y_probas, the arrays contain the label of each segment (0 or 1)    
    '''

    model_weights_path = os.path.join('Models','Saved_models','Weights')      # path to saved weights

    K.set_image_data_format('channels_last')

    model = ChronoNet.net()
    model.load_weights(os.path.join(model_weights_path, config.name + '.h5'))

    y_probas = []
    y_true = []
    file_names = []

    for i, file in enumerate(file_list):
        print(file.filenames[0])
        
        file_segments = get_data_keys_sequential([file], config)

        montage = montages[i]

        gen_test = SegmentedGenerator([file], file_segments, [montage], batch_size=32, shuffle=False)

        y_aux = []
        for j in range(len(gen_test)):
            _, y = gen_test[j]
            y_aux.append(y)
        true_labels = np.vstack(y_aux)


        prediction = model.predict(gen_test)

        pred = np.empty(len(prediction), dtype='float32')
        for j in range(len(pred)):
            pred[j] = prediction[j][1]

        true = np.empty(len(true_labels), dtype='uint8')
        for j in range(len(true)):
            true[j] = true_labels[j][1]

        y_probas.append(pred)
        y_true.append(true)
        file_names.append(file.filenames[0])

    return file_names, y_probas, y_true


# ## EVENT & MASK MANIPULATION ###

def eventList2Mask(events, totalLen, fs):
    """Convert list of events to mask.
    
    Returns a logical array of length totalLen.
    All event epochs are set to True
    
    Args:
        events: list of events times in seconds. Each row contains two
                columns: [start time, end time]
        totalLen: length of array to return in samples
        fs: sampling frequency of the data in Hertz
    Return:
        mask: logical array set to True during event epochs and False the rest
              if the time.
    """
    mask = np.zeros((totalLen,))
    for event in events:
        for i in range(min(int(event[0]*fs), totalLen-1), min(int(event[1]*fs), totalLen-1)):
            mask[i] = 1
    return mask


def mask2eventList(mask, fs):
    """Convert mask to list of events.
        
    Args:
        mask: logical array set to True during event epochs and False the rest
          if the time.
        fs: sampling frequency of the data in Hertz
    Return:
        events: list of events times in seconds. Each row contains two
                columns: [start time, end time]
    """
    events = list()
    tmp = []
    start_i = np.where(np.diff(np.array(mask, dtype=int)) == 1)[0]
    end_i = np.where(np.diff(np.array(mask, dtype=int)) == -1)[0]
    
    if len(start_i) == 0 and mask[0]:
        events.append([0, (len(mask)-1)/fs])
    else:
        # Edge effect
        if mask[0]:
            events.append([0, end_i[0]/fs])
            end_i = np.delete(end_i, 0)
        # Edge effect
        if mask[-1]:
            if len(start_i):
                tmp = [[start_i[-1]/fs, (len(mask)-1)/fs]]
                start_i = np.delete(start_i, len(start_i)-1)
        for i in range(len(start_i)):
            events.append([start_i[i]/fs, end_i[i]/fs])
        events += tmp
    return events


def merge_events(events, distance):
    """ Merge events.
    
    Args:
        events: list of events times in seconds. Each row contains two
                columns: [start time, end time]
        distance: maximum distance (in seconds) between events to be merged
    Return:
        events: list of events (after merging) times in seconds.
    """
    i = 1
    tot_len = len(events)
    while i < tot_len:
        if events[i][0] - events[i-1][1] < distance:
            events[i-1][1] = events[i][1]
            events.pop(i)
            tot_len -= 1
        else:
            i += 1
    return events


def get_events(events, margin):
    ''' Converts the unprocessed events to the post-processed events based on physiological constrains:
    - seizure alarm events distanced by 0.2*margin (in seconds) are merged together
    - only events with a duration longer than margin*0.8 are kept
    (for more info, check: K. Vandecasteele et al., “Visual seizure annotation and automated seizure detection using
    behind-the-ear elec- troencephalographic channels,” Epilepsia, vol. 61, no. 4, pp. 766–775, 2020.)

    Args:
        events: list of events times in seconds. Each row contains two
                columns: [start time, end time]
        margin: float, the desired margin in seconds

    Returns:
        ev_list: list of events times in seconds after merging and discarding short events.
    '''
    events_merge = merge_events(events, margin*0.2)
    ev_list = []
    for i in range(len(events_merge)-1):
        if events_merge[i][1] - events_merge[i][0] >= margin*0.8:
            ev_list.append(events_merge[i])

    return ev_list



def post_processing(y_pred, fs, th, margin):
    ''' Post process the predictions given by the model based on physiological constraints: a seizure is
    not shorter than 10 seconds and events separated by 2 seconds are merged together.

    Args:
        y_pred: array with the seizure classification probabilties (of each segment)
        fs: sampling frequency of the y_pred array (1/window length - in this challenge fs = 1/2)
        th: threshold value for seizure probability (float between 0 and 1)
        margin: float, the desired margin in seconds (check get_events)
    
    Returns:
        pred: array with the processed classified labels by the model
    '''
    pred = (y_pred > th)
    events = mask2eventList(pred, fs)
    events = get_events(events, margin)
    pred = eventList2Mask(events, len(y_pred), fs)

    return pred


def getOverlap(a, b):
    ''' If > 0, the two intervals overlap.
    a = [start_a, end_a]; b = [start_b, end_b]
    '''
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def perf_measure_epoch(y_true, y_pred):
    ''' Calculate the performance metrics based on the EPOCH method.
    
    Args:
        y_true: array with the ground-truth labels of the segments
        y_pred: array with the predicted labels of the segments

    Returns:
        TP: true positives
        FP: false positives
        TN: true negatives
        FN: false negatives
    '''

    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_pred)): 
        if y_true[i] == y_pred[i] == 1:
           TP += 1
        if y_pred[i] == 1 and y_true[i] != y_pred[i]:
           FP += 1
        if y_true[i] == y_pred[i] == 0:
           TN += 1
        if y_pred[i] == 0 and y_true[i] != y_pred[i]:
           FN += 1

    return TP, FP, TN, FN


def perf_measure_ovlp(y_true, y_pred, fs):
    ''' Calculate the performance metrics based on the any-overlap method.
    
    Args:
        y_true: array with the ground-truth labels of the segments
        y_pred: array with the predicted labels of the segments
        fs: sampling frequency of the predicted and ground-truth label arrays
            (in this challenge, fs = 1/2)

    Returns:
        TP: true positives
        FP: false positives
        FN: false negatives
    '''
    true_events = mask2eventList(y_true, fs)
    pred_events = mask2eventList(y_pred, fs)

    TP = 0
    FP = 0
    FN = 0

    for pr in pred_events:
        found = False
        for tr in true_events:
            if getOverlap(pr, tr) > 0:
                TP += 1
                found = True
        if not found:
            FP += 1
    for tr in true_events:
        found = False
        for pr in pred_events:
            if getOverlap(tr, pr) > 0:
                found = True
        if found:
            FN += 1

    return TP, FP, FN


def get_metrics_scoring(pred_file):
    ''' Get the score for the challenge.

    Args:
        pred_file: path to the prediction file containing the objects 'filenames',
                   'predictions' and 'labels' (as returned by 'predict_net' function)
    
    Returns:
        score: the score of the challenge
        sens_ovlp: sensitivity calculated with the any-overlap method
        FA_epoch: false alarm rate (false alarms per hour) calculated with the EPOCH method
    '''
    with h5py.File(pred_file, 'r') as f:
        file_names_preds = []
        y_preds = []
        y_trues = []

        file_names_ds = f['filenames']
        y_preds_ds = f['predictions']
        y_true_ds = f['labels']

        for i in range(len(file_names_ds)):
            file_names_preds.append(file_names_ds[i])
            y_preds.append(y_preds_ds[i])
            y_trues.append(y_true_ds[i])


    total_N = 0
    total_TP_epoch = 0
    total_FP_epoch = 0
    total_FN_epoch = 0
    total_TP_ovlp = 0
    total_FP_ovlp = 0
    total_FN_ovlp = 0
    total_seiz = 0

    for i, y_pred in enumerate(y_preds):
        total_N += len(y_pred)
        total_seiz += np.sum(y_trues[i])

        # Post process predictions (merge predicted events separated by 2 seconds and discard events smaller than 10 seconds)
        y_pred = post_processing(y_pred, fs=1/2, th=0.5, margin=10)

        TP_epoch, FP_epoch, TN_epoch, FN_epoch = perf_measure_epoch(y_trues[i], y_pred)
        total_TP_epoch += TP_epoch
        total_FP_epoch += FP_epoch
        total_FN_epoch += FN_epoch

        TP_ovlp, FP_ovlp, FN_ovlp = perf_measure_ovlp(y_trues[i], y_pred, fs=1/2)
        total_TP_ovlp += TP_ovlp
        total_FP_ovlp += FP_ovlp
        total_FN_ovlp += FN_ovlp

    if total_seiz == 0:
        sens_ovlp = float("nan")
    else:
        sens_ovlp = total_TP_ovlp/(total_TP_ovlp + total_FN_ovlp)
        
    FA_epoch = total_FP_epoch*3600/total_N

    score = sens_ovlp - 0.4*FA_epoch

    print('Final score: ' + str(score))

    return score, sens_ovlp, FA_epoch