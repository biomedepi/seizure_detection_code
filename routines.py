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

import aux_functions
import ChronoNet
from generator_ds import SegmentedGenerator


def split_sets(data_path, montage):

    ### Divide train/val/test ###

    patient_list = [x for x in os.listdir(data_path) if 'P_ID' in x]

    train_pats_list = random.sample(patient_list, int(np.round(len(patient_list)*0.7)))
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

    segments_S = []
    segments_NS = []

    for i, file in enumerate(file_list):
        # Load annotation file
        rec_path = file.filenames[0]
        pat_name = os.path.split(os.path.split(rec_path)[0])[1]
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

    segments = []

    for i, file in enumerate(file_list):
        # Load annotation file
        rec_path = file.filenames[0]
        pat_name = os.path.split(os.path.split(rec_path)[0])[1]
        ann_path = rec_path[0:-4] + '_a1.tsv'

        [events_times, _, _] = aux_functions.wrangle_tsv_sz1(ann_path, only_visible_bhe=False)

        if not events_times:
            n_segs =  int(np.floor((np.floor(file.tmax) - config.frame)/config.frame))
            seg_start = np.arange(0, n_segs)*config.frame
            seg_stop = seg_start + config.frame

            segments.extend(np.column_stack(([i]*n_segs, seg_start, seg_stop, np.zeros(n_segs))))
        else:
            for e, ev in enumerate(events_times):
                if ev[1] == 'None':
                    ev[1] = ev[0] + 10

                if e == 0:
                    n_segs = int(np.floor((ev[0])/config.frame)-1)
                    seg_start = np.arange(0, n_segs)*config.frame
                    seg_stop = seg_start + config.frame
                    segments.extend(np.column_stack(([i]*n_segs, seg_start, seg_stop, np.zeros(n_segs))))
                else:
                    n_segs = int(np.floor((ev[0] - events_times[e-1][1])/config.frame)-1)
                    seg_start = np.arange(0, n_segs)*config.frame + events_times[e-1][1]
                    seg_stop = seg_start + config.frame
                    segments.extend(np.column_stack(([i]*n_segs, seg_start, seg_stop, np.zeros(n_segs))))
                if e == len(events_times)-1:
                    n_segs = int(np.floor((np.floor(file.tmax) - ev[1])/config.stride)-1)
                    seg_start = np.arange(0, n_segs)*config.frame + ev[1]
                    seg_stop = seg_start + config.frame
                    segments.extend(np.column_stack(([i]*n_segs, seg_start, seg_stop, np.zeros(n_segs))))

    return segments


def train_net(config, gen_train, gen_val):

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

    model_weights_path = 'Models\Saved_models\Weights'      # path to saved weights

    fs_preds = 1/config.frame      # sampling frequency of the predictions output vector

    K.set_image_data_format('channels_last')

    model = ChronoNet.net()
    model.load_weights(os.path.join(model_weights_path, config.name + '.h5'))

    input_shape = model.layers[0].output_shape

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