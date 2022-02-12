import os
import sys
import random
import warnings
import glob
import netCDF4 as nc

import numpy as np
import pandas as pd

from keras.preprocessing.image import load_img, img_to_array
from keras.models import Model, load_model, Sequential
from keras.layers import Input, LSTM, Dense, Permute, Reshape, LeakyReLU
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv1D, SeparableConv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import normalize
from keras.regularizers import l2
import keras.layers as kl
from keras.utils import Sequence

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras.optimizers import SGD, Adam
from keras.metrics import mean_squared_error
import tensorflow as tf
from keras.callbacks import CSVLogger

from numpy.random import seed, randint, rand, uniform


def r2_keras(y_true, y_pred):
    y_t = tf.multiply(y_true, tf.cast(tf.not_equal(y_true, 0), tf.float32))
    y_p = tf.multiply(y_pred, tf.cast(tf.not_equal(y_true, 0), tf.float32))
    SS_res =  K.sum(K.square(y_t - y_p)) 
    SS_tot = K.sum(K.square(y_t - K.mean(y_t))) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )


class data_generator( Sequence ) :
  def __init__(self, xnames, ynames, batch_size) :
    self.xnames = xnames
    self.ynames = ynames
    self.batch_size = batch_size

  def __len__(self) :
    return np.ceil( len(self.xnames) / float(self.batch_size)).astype(int)
  
  def __getitem__(self, idx) :

    batch_x = self.xnames[ idx*self.batch_size : idx*self.batch_size + self.batch_size ]
    batch_y = self.ynames[ idx*self.batch_size : idx*self.batch_size + self.batch_size ]


    d2m = np.stack([ np.load(s)['d2m'] for s in batch_x])   # batch_size, 10, 56, 360
    t2m = np.stack([ np.load(s)['t2m'] for s in batch_x])   # batch_size, 10, 56, 360
    skt = np.stack([ np.load(s)['skt'] for s in batch_x])   # batch_size, 10, 56, 360
    sp = np.stack([ np.load(s)['sp'] for s in batch_x])   # batch_size, 10, 56, 360
    sza = np.stack([ np.load(s)['sza'] for s in batch_x])   # batch_size, 10, 56, 360
    raa = np.stack([ np.load(s)['raa'] for s in batch_x])   # batch_size, 10, 56, 360
    o4 = np.stack([ np.load(s)['o4'] for s in batch_x])   # batch_size, 10, 56, 360
    tempx1 = np.stack([d2m, t2m, skt, sp, sza, raa, o4], axis=-1)
    tprof = np.stack([ np.load(s)['tprof'] for s in batch_x])   # batch_size, 10, 56, 360
    qprof = np.stack([ np.load(s)['qprof'] for s in batch_x])   # batch_size, 10, 56, 360
    uprof = np.stack([ np.load(s)['uprof'] for s in batch_x])   # batch_size, 10, 56, 360
    vprof = np.stack([ np.load(s)['vprof'] for s in batch_x])   # batch_size, 10, 56, 360
    tempx2 = np.stack([tprof, qprof, uprof, vprof], axis=-1)

    tempy = np.stack([ np.load(s)['aero'] for s in batch_y])[:, :, np.newaxis]   # batch, 

    return [tempx1, tempx2], tempy

def data_split(x, y, ratio1, ratio2, maskname=None):
    dsize1 = int(x.shape[0]*ratio1)
    dsize2 = int(x.shape[0]*ratio2)
    dmask = np.array(list(range(0, x.shape[0])))
    random.shuffle(dmask)

    dmask1 = dmask[:dsize1]
    x1 = x[dmask1]
    y1 = y[dmask1]

    dmask2 = dmask[dsize1:dsize2]
    x2 = x[dmask2]
    y2 = y[dmask2]

    dmask3 = dmask[dsize2:]
    x3 = x[dmask3]
    y3 = y[dmask3]

    if maskname:
        np.savez(maskname, train=dmask1, valid=dmask2, test=dmask3)

    return x1, y1, x2, y2, x3, y3

def get_x(fname):
    d2m = np.load(fname)['d2m']    # batch_size, 10, 56, 360
    t2m = np.load(fname)['t2m']    # batch_size, 10, 56, 360
    skt = np.load(fname)['skt']    # batch_size, 10, 56, 360
    sp = np.load(fname)['sp']    # batch_size, 10, 56, 360
    sza = np.load(fname)['sza']    # batch_size, 10, 56, 360
    raa = np.load(fname)['raa']    # batch_size, 10, 56, 360
    o4 = np.load(fname)['o4']    # batch_size, 10, 56, 360
    tempx1 = np.stack([d2m, t2m, skt, sp, sza, raa, o4], axis=-1)
    tempx1 = tempx1[np.newaxis, :, :]
    tprof = np.load(fname)['tprof']    # batch_size, 10, 56, 360
    qprof = np.load(fname)['qprof']    # batch_size, 10, 56, 360
    uprof = np.load(fname)['uprof']    # batch_size, 10, 56, 360
    vprof = np.load(fname)['vprof']    # batch_size, 10, 56, 360
    tempx2 = np.stack([tprof, qprof, uprof, vprof], axis=-1)
    tempx2 = tempx2[np.newaxis, :, :]
    return [tempx1, tempx2]

# seed(6658)



xfiles = np.array(sorted(glob.glob('X/X_*.npz')))
yfiles = np.array(sorted(glob.glob('Y/Y_*.npz')))

xlist1, ylist1, xlist2, ylist2, xlist3, ylist3 = data_split(xfiles, yfiles, 0.7225, 0.85, maskname='train_valid_test_mask.npz')

mask1 = np.load('train_valid_test_mask.npz')['train']
mask2 = np.load('train_valid_test_mask.npz')['valid']
mask3 = np.load('train_valid_test_mask.npz')['test']
xlist1 = xfiles[mask1]
ylist1 = yfiles[mask1]
xlist2 = xfiles[mask2]
ylist2 = yfiles[mask2]
xlist3 = xfiles[mask3]
ylist3 = yfiles[mask3]

print('Splitting: ', len(xlist1), len(ylist1), len(xlist2), len(ylist2), len(xlist3), len(ylist3) )
train_generator = data_generator(xlist1, ylist1, 5)   # 90%*900 = 810
valid_generator = data_generator(xlist2, ylist2, 5)   # 10%*900 = 90


prof_input = Input((21, 4), name='rean_input')
meas_input = Input((9, 7), name='meas_input')

pc1 = Conv1D(36, 2, strides=1, padding='same', activation="relu")(meas_input)
pc1 = Conv1D(36, 2, strides=1, padding='same', activation="relu")(pc1)
pd1 = Dense(36, activation="relu")(pc1) # 36
pc2 = Conv1D(128, 2, strides=1, padding='same', activation="relu")(pd1)
pc2 = Conv1D(128, 2, strides=1, padding='same', activation="relu")(pc2)
pd2 = Dense(128, activation="relu")(pc2) # 128
pc3 = Conv1D(256, 2, strides=1, padding='same', activation="relu")(pd2)
pc3 = Conv1D(256, 2, strides=1, padding='same', activation="relu")(pc3)
pd3 = Dense(256, activation="relu")(pc3) # 21

lstm = LSTM(21, name='LSTM1') (pd3)  # this gives a pseudo profile
lstm = kl.Reshape( (21, 1)) (lstm)

concat_layer = concatenate([prof_input, lstm])   # 21, 5

c1 = Conv1D(128, 2, strides=1, padding='same', activation="relu")(concat_layer)  # 21, 32
c1 = Conv1D(128, 2, strides=1, padding='same', activation="relu")(c1)  # 21, 32
p1 = Dense(128, activation="relu") (c1)   # 21, 32

c2 = Conv1D(128, 2, strides=1, padding='same', activation="relu")(p1)  # 21, 64 
c2 = Conv1D(128, 2, strides=1, padding='same', activation="relu")(c2)          # 21, 64
p2 = Dense(128, activation="relu") (c2)   # 21, 64

c3 = Conv1D(128, 2, strides=1, padding='same', activation="relu")(p2) # 21, 128
c3 = Conv1D(128, 2, strides=1, padding='same', activation="relu")(c3)         # 21, 128
p3 = Dense(128, activation="relu") (c3)   # 21, 128

u = concatenate([p1, p2, p3])   # 21, 5

output = Dense(1, activation="linear")(u) # 1

# define a model with a list of two inputs
model = Model(inputs=[meas_input, prof_input], outputs=output)


opt = Adam(lr=1e-5)
model.compile(optimizer=opt, loss='mse', metrics=[r2_keras, 'mse'])
model.summary()
model.load_weights('DLMAXDOAS_final.h5')

csv_logger = CSVLogger( 'DLMAXDOAS_log.csv', append=True, separator=';')
earlystopper = EarlyStopping(patience=20, verbose=1)
checkpointer = ModelCheckpoint('DLMAXDOAS_checkpt_{val_loss:.2f}.h5', verbose=1, save_best_only=True)
results = model.fit(train_generator,
                    validation_data=valid_generator, epochs=500,
                    callbacks=[earlystopper, checkpointer, csv_logger])
model.save('DLMAXDOAS_final.h5')


def outname(inname):
	return 'Ypred' + inname.split('/')[-1][1:-1] + 'y'

flist = np.array(sorted(glob.glob('X/X_*.npz')))

for file in flist:
	ypred = model.predict( get_x(file) )[0]
	np.save('Y_pred/' + outname(file), ypred)