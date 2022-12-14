import os
from tensorflow.keras import backend as K

import aux_functions
import ChronoNet


model_weights_path = 'Models\Saved_models\Weights'      # path to saved weights
model_name = 'ChronoNet_test'                           # name of the model

frame = 2               # segment window size in seconds
fs_preds = 1/frame      # sampling frequency of the predictions output vector


aux_functions.set_gpu()
K.set_image_data_format('channels_last')

model = ChronoNet.net()
model.load_weights(os.path.join(model_weights_path, model_name + '.h5'))

input_shape = model.layers[0].output_shape

y_probas = []
y_true = []
file_names = []

for i, t_idx in enumerate(test_idx):
    
    print(file_names[t_idx])

    if config.load_data_offline:
        data_test_rec = signals[t_idx]
        labels_test_rec = labels[t_idx]
    else:
        with h5py.File(config.dataPath, 'r') as f:
            data_test_rec = np.array(list(f["signals"][t_idx]))
            if data_test_rec.shape[0] > data_test_rec.shape[1]:
                data_test_rec = data_test_rec.T
            # Normalise

            labels_test_rec = np.array(list(f["labels"][t_idx]))


    if len(labels_test_rec) <= config.batch_size_test * 2 * config.fs:
        batch_size = 1
    else:
        batch_size = config.batch_size_test

    if len(config.dataPath) == 1:
        gen_test_rec = generator_array.SegmentedGenerator(config.dataPath[0], True, [data_test_rec], [labels_test_rec], [0],
                                                        input_shape=input_shape, batch_size=batch_size,
                                                        stride=config.stride, stride_s=config.stride,
                                                        factor=None, n_channels=len(config.montage[0]),
                                                        window_size=config.frame, fs=config.fs, n_classes=config.nr_classes,
                                                        shuffle=False, filter_rms=False,
                                                        filter_rms_percent=0)

        # gen_test_rec = generator_array.SegmentedGenerator([data_test[1][i]], [data_test[2][i]],
        #                                         input_shape=input_shape, batch_size=batch_size,
        #                                         stride=config.stride, stride_s=config.stride,
        #                                         factor=None, n_channels=len(config.montage[0]),
        #                                         window_size=config.frame, fs=config.fs, n_classes=config.nr_classes,
        #                                         shuffle=False, filter_rms=False, filter_rms_percent=0)

        y_aux = []
        for j in range(len(gen_test_rec)):
            _, y = gen_test_rec[j]
            y_aux.append(y)
        true_labels = np.vstack(y_aux)

        prediction = model.predict(gen_test_rec)

        pred = np.empty(len(prediction), dtype='float32')
        for j in range(len(pred)):
            pred[j] = prediction[j][1]

        true = np.empty(len(true_labels), dtype='uint8')
        for j in range(len(true)):
            true[j] = true_labels[j][1]

        y_probas.append(pred)
        y_true.append(true)
        file_names_test.append(file_names[t_idx])

# Saving predictions
dt_fl = h5py.vlen_dtype(np.dtype('float32'))
dt_int = h5py.vlen_dtype(np.dtype('uint8'))
dt_str = h5py.special_dtype(vlen=str)

if not os.path.exists(os.path.join(root, 'Predictions')):
    os.mkdir(os.path.join(root, 'Predictions'))
if not os.path.exists(os.path.join(root, 'Predictions', config.group)):
    os.mkdir(os.path.join(root, 'Predictions', config.group))

with h5py.File(os.path.join(root, 'Predictions', config.group, config.name + '.h5'), 'w') as f:
    dset_signals = f.create_dataset('predictions', (len(y_probas),), dtype=dt_fl)
    dset_labels = f.create_dataset('labels', (len(y_true),), dtype=dt_int)
    dset_file_names = f.create_dataset('filenames', (len(file_names_test),), dtype=dt_str)

    for i in range(len(file_names_test)):
        dset_signals[i] = y_probas[i]
        dset_labels[i] = y_true[i]
        dset_file_names[i] = file_names_test[i]