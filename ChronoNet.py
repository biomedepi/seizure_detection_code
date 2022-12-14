import tensorflow.keras as kr
import numpy as np

from tensorflow.keras import layers, Input
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.models import Model, Sequential, load_model
import tensorflow.keras.activations as activations

## Source: https://github.com/aguscerdo/EE239AS-Project

def net(inception=True, res=True, strided=True, maxpool=False, avgpool=False, batchnorm=True):

    input_shape = (400, 2)


    l2 = 0.01               # L2-regularization factor
    pad = 'same'
    padp = 'same'
    state_size = 32
    filters = 32
    strides = 2
    c_act = 'relu'
    r_act = 'sigmoid'
    rk_act = 'tanh'

    rec_drop = 0
    cnn_drop = 0.6

    r = kr.regularizers.l2(l2)
    stride_size = strides if strided else 1

    input = Input(input_shape)

    if inception:
        c0 = layers.Conv1D(filters, kernel_size=2, strides=stride_size, padding=pad,
                           activation=c_act)(input)
        c1 = layers.Conv1D(filters, kernel_size=4, strides=stride_size, padding=pad,
                           activation=c_act)(input)
        c2 = layers.Conv1D(filters, kernel_size=8, strides=stride_size, padding=pad,
                           activation=c_act)(input)

        c = layers.concatenate([c0, c1, c2])

        if maxpool:
            c = layers.MaxPooling1D(2, padding=padp)(c)
        elif avgpool:
            c = layers.AveragePooling1D(2, padding=padp)(c)
        if batchnorm:
            c = layers.BatchNormalization()(c)

        c = layers.SpatialDropout1D(cnn_drop)(c)


        c0 = layers.Conv1D(filters, kernel_size=2, strides=stride_size, padding=pad,
                           activation=c_act)(c)
        c1 = layers.Conv1D(filters, kernel_size=4, strides=stride_size, padding=pad,
                           activation=c_act)(c)
        c2 = layers.Conv1D(filters, kernel_size=8, strides=stride_size, padding=pad,
                           activation=c_act)(c)

        c = layers.concatenate([c0, c1, c2])

        if maxpool:
            c = layers.MaxPooling1D(2, padding=padp)(c)
        elif avgpool:
            c = layers.AveragePooling1D(2, padding=padp)(c)
        if batchnorm:
            c = layers.BatchNormalization()(c)

        c = layers.SpatialDropout1D(cnn_drop)(c)


        c0 = layers.Conv1D(filters, kernel_size=2, strides=stride_size, padding=pad,
                           activation=c_act)(c)
        c1 = layers.Conv1D(filters, kernel_size=4, strides=stride_size, padding=pad,
                           activation=c_act)(c)
        c2 = layers.Conv1D(filters, kernel_size=8, strides=stride_size, padding=pad,
                           activation=c_act)(c)

        c = layers.concatenate([c0, c1, c2])

        if maxpool:
            c = layers.MaxPooling1D(2, padding=padp)(c)
        elif avgpool:
            c = layers.AveragePooling1D(2, padding=padp)(c)
        if batchnorm:
            c = layers.BatchNormalization()(c)

        c = layers.SpatialDropout1D(cnn_drop)(c)


    else:  # No inception Modules
        c = layers.Conv1D(filters, kernel_size=4, strides=stride_size, padding=pad, activation=c_act)(
            input)
        if maxpool:
            c = layers.MaxPooling1D(2, padding=padp)(c)
        elif avgpool:
            c = layers.AveragePooling1D(2, padding=padp)(c)
        if batchnorm:
            c = layers.BatchNormalization()(c)
        c = layers.SpatialDropout1D(cnn_drop)(c)

        c = layers.Conv1D(filters, kernel_size=4, strides=stride_size, padding=pad, activation=c_act)(c)
        if maxpool:
            c = layers.MaxPooling1D(2, padding=padp)(c)
        elif avgpool:
            c = layers.AveragePooling1D(2, padding=padp)(c)
        if batchnorm:
            c = layers.BatchNormalization()(c)
        c = layers.SpatialDropout1D(cnn_drop)(c)

        c = layers.Conv1D(filters, kernel_size=4, strides=stride_size, padding=pad, activation=c_act)(c)
        if maxpool:
            c = layers.MaxPooling1D(2, padding=padp)(c)
        elif avgpool:
            c = layers.AveragePooling1D(2, padding=padp)(c)
        if batchnorm:
            c = layers.BatchNormalization()(c)
        c = layers.SpatialDropout1D(cnn_drop)(c)

    if res:  # Residual RNN
        g1 = layers.GRU(state_size, return_sequences=True, activation=rk_act,
                        recurrent_activation=r_act, dropout=rec_drop, recurrent_dropout=rec_drop,
                        recurrent_regularizer=r, kernel_regularizer=r)(c)
        g2 = layers.GRU(state_size, return_sequences=True, activation=rk_act,
                        recurrent_activation=r_act, dropout=rec_drop, recurrent_dropout=rec_drop,
                        recurrent_regularizer=r, kernel_regularizer=r)(g1)

        g_concat1 = layers.concatenate([g1, g2])

        g3 = layers.GRU(state_size, return_sequences=True, activation=rk_act, recurrent_activation=r_act,
                        dropout=rec_drop, recurrent_dropout=rec_drop,
                        recurrent_regularizer=r, kernel_regularizer=r)(g_concat1)

        g_concat2 = layers.concatenate([g1, g2, g3])

        g = layers.GRU(state_size, return_sequences=False, activation=rk_act, recurrent_activation=r_act,
                       dropout=rec_drop, recurrent_dropout=rec_drop,
                       recurrent_regularizer=r, kernel_regularizer=r)(g_concat2)

    else:  # No Residual RNN
        g = layers.GRU(state_size, return_sequences=True, activation=rk_act, recurrent_activation=r_act,
                       dropout=rec_drop, recurrent_dropout=rec_drop,
                       recurrent_regularizer=r, kernel_regularizer=r)(c)

        g = layers.GRU(state_size, return_sequences=True, activation=rk_act, recurrent_activation=r_act,
                       dropout=rec_drop, recurrent_dropout=rec_drop,
                       recurrent_regularizer=r, kernel_regularizer=r)(g)
        g = layers.GRU(state_size, return_sequences=True, activation=rk_act, recurrent_activation=r_act,
                       dropout=rec_drop, recurrent_dropout=rec_drop,
                       recurrent_regularizer=r, kernel_regularizer=r)(g)

        g = layers.GRU(state_size, return_sequences=False, activation=rk_act, recurrent_activation=r_act,
                       dropout=rec_drop, recurrent_dropout=rec_drop,
                       recurrent_regularizer=r, kernel_regularizer=r)(g)

    d = layers.Dense(2)(g)
    out = layers.Softmax()(d)

    model = Model(input, out)

    return model

