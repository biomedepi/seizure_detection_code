import os
import numpy as np
import random
import aux_functions

def generate_data_keys_subsample(file_list, frame, stride, stride_s, factor):

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
            n_segs =  int(np.floor((np.floor(file.tmax) - frame)/stride))
            seg_start = np.arange(0, n_segs)*stride
            seg_stop = seg_start + frame

            segments_NS.extend(np.column_stack(([i]*n_segs, seg_start, seg_stop, np.zeros(n_segs))))
        else:
            for e, ev in enumerate(events_times_S):
                n_segs = int(np.floor((ev[1] - ev[0])/stride_s)-1)
                seg_start = np.arange(0, n_segs)*stride_s + ev[0]
                seg_stop = seg_start + frame
                segments_S.extend(np.column_stack(([i]*n_segs, seg_start, seg_stop, np.ones(n_segs))))

            for e, ev in enumerate(events_times_NS):
                if ev[1] == 'None':
                    ev[1] = ev[0] + 10

                if e == 0:
                    n_segs = int(np.floor((ev[0])/stride)-1)
                    seg_start = np.arange(0, n_segs)*stride
                    seg_stop = seg_start + frame
                    segments_NS.extend(np.column_stack(([i]*n_segs, seg_start, seg_stop, np.zeros(n_segs))))
                else:
                    n_segs = int(np.floor((ev[0] - events_times_NS[e-1][1])/stride)-1)
                    seg_start = np.arange(0, n_segs)*stride + events_times_NS[e-1][1]
                    seg_stop = seg_start + frame
                    segments_NS.extend(np.column_stack(([i]*n_segs, seg_start, seg_stop, np.zeros(n_segs))))
                if e == len(events_times_NS)-1:
                    n_segs = int(np.floor((np.floor(file.tmax) - ev[1])/stride)-1)
                    seg_start = np.arange(0, n_segs)*stride + ev[1]
                    seg_stop = seg_start + frame
                    segments_NS.extend(np.column_stack(([i]*n_segs, seg_start, seg_stop, np.zeros(n_segs))))

    segments_S.extend(random.sample(segments_NS, factor*len(segments_S)))
    random.shuffle(segments_S)

    return segments_S