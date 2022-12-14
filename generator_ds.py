import os
import numpy as np
from tensorflow import keras
import random
import time
import aux_functions
from tqdm import tqdm


class SegmentedGenerator(keras.utils.Sequence):

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




# class SequentialGenerator(keras.utils.Sequence):
#     def __init__(self, files_list, montages, normalize=True, batch_size=32,):

#         self.data = data
#         self.labels = labels
#         self.batch_size = batch_size
#         self.shuffle = shuffle
#         self.n_channels = n_channels
#         self.window_size = window_size
#         self.fs = fs
#         self.frpwin = self.fs * self.window_size

#         key_array = []

#         for i, array in enumerate(self.data):
#             n_frames = array.shape[0]
#             j = 0
#             frame_idx = (self.window_size + j) * self.fs
#             while frame_idx < (n_frames - self.frpwin):
#                 key_array.append([i, frame_idx])
#                 j += 1
#                 frame_idx = (self.window_size + j) * self.fs
#         self.key_array = np.asarray(key_array, dtype=np.uint32)

#         self.on_epoch_end()

#     def __len__(self):
#         return len(self.key_array) // self.batch_size

#     def __getitem__(self, index):
#         keys = np.arange(start=index * self.batch_size, stop=(index + 1) * self.batch_size)
#         x, y = self.__data_generation__(keys)

#         return x, y

#     def on_epoch_end(self):
#         if self.shuffle:
#             self.key_array = np.random.permutation(self.key_array)

#     def __data_generation__(self, keys):
#         frames_per_sample = self.fs + 2 * self.frpwin
#         x = np.empty(shape=(self.batch_size, frames_per_sample, self.n_channels, 26), dtype=np.float32)
#         y = np.empty(shape=(self.batch_size, 2))

#         for i in range(self.batch_size):
#             key = self.key_array[keys[i]]
#             start_frame = key[1] - self.frpwin
#             stop_frame = start_frame + 2 * self.frpwin + self.fs
#             data_seg = self.data[key[0]][:, :, start_frame:stop_frame]
#             data_seg = np.reshape(data_seg, newshape=(data_seg.shape[2], 22, 26))
#             x[i, :, :, :] = data_seg
#             uni_label = np.amax(self.labels[key[0]][key[1]:key[1] + self.fs])
#             if uni_label == 0:
#                 y[i] = [1., 0.]
#             else:
#                 y[i] = [0., 1.]

#         return x, y