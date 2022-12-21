import numpy as np
from tensorflow import keras
import aux_functions
from tqdm import tqdm


class SegmentedGenerator(keras.utils.Sequence):
    ''' Class where the keras data generator is built.

    Args:
        files_list: list of raw instances (of the mne package) containing EEG recordings
        segments: list of keys (each key is a list [1x4] containing the recording index in the files_list,
                  the start and stop of the segment in seconds and the label of the segment)
        montages: a list of montages (list of lists with pairs of strings where each pair is an electrode name
                  used for each channel in the montage) corresponding to each recording in the files_list
        normalize: boolean, if True the channel data is normalized with the z-score method
        batch_size: batch size of the generator
        shuffle: boolean, if True, the segments are randomly mixed in every batch
    
    '''

    def __init__(self, files_list, segments, montages, normalize=True, batch_size=32, shuffle=True):
        
        'Initialization'
        self.files_list = files_list
        self.segments = segments
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.data_segs = np.empty(shape=[len(self.segments), 400, 2])
        self.labels = np.empty(shape=[len(self.segments), 2])
        segs_to_load = self.segments

        pbar = tqdm(total = len(segs_to_load)+1)
        count = 0
        while segs_to_load:

            curr_rec = int(segs_to_load[0][0])
            comm_recs = [i for i, x in enumerate(segs_to_load) if x[0] == curr_rec]

            rec = files_list[curr_rec]

            rec_signal, fs_out = aux_functions.apply_montage(rec, montages[curr_rec], normalize)
            
            for r in comm_recs:
                start_seg = int(segs_to_load[r][1]*fs_out)
                stop_seg = int(segs_to_load[r][2]*fs_out)

                self.data_segs[count, :, 0] = rec_signal[0][start_seg:stop_seg]
                self.data_segs[count, :, 1] = rec_signal[1][start_seg:stop_seg]

                if segs_to_load[r][3] == 1:
                    self.labels[count, :] = [0, 1]
                elif segs_to_load[r][3] == 0:
                    self.labels[count, :] = [1, 0]

                count += 1
                pbar.update(1)

            segs_to_load = [s for i, s in enumerate(segs_to_load) if i not in comm_recs]
        
        self.key_array = np.arange(len(self.labels))

        self.on_epoch_end()


    def __len__(self):
        return len(self.key_array) // self.batch_size

    def __getitem__(self, index):
        keys = np.arange(start=index * self.batch_size, stop=(index + 1) * self.batch_size)
        x, y = self.__data_generation__(keys)
        return x, y

    def on_epoch_end(self):
        if self.shuffle:
            self.key_array = np.random.permutation(self.key_array)

    def __data_generation__(self, keys):
        return self.data_segs[self.key_array[keys], :, :], self.labels[self.key_array[keys]]
