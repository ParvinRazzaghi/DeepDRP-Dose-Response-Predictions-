
import sys, os
import numpy as np
import tensorflow as tf
import scipy.sparse as sp
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
!pip install tffm==1.0.1
#tf.optimizers.Adam()
from tffm.models import TFFMRegressor # if get error, in line 96 replace tf.train.AdamOptimizer with tf.optimizers.Adam

from comboFM.utils import concatenate_features, standardize



from __future__ import print_function
import tensorflow
from keras.layers import Input, Dropout, Dense
from keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras.regularizers import l2
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import os, sys, re, math, time, scipy
import numpy as np
import argparse
from scipy.io import loadmat
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import json
import pickle
import collections
from sklearn import metrics
from sklearn.metrics import r2_score
from collections import OrderedDict
#import tensorflow_addons
import numpy as np
import tensorflow as tf, keras
import tensorflow as tf
#import tensorflow_addons as tfa
from tensorflow.keras import regularizers
from keras import layers
from keras.optimizers import Adam
from keras import utils
from keras.losses import MeanSquaredError
from keras.metrics import MeanSquaredError
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping,LearningRateScheduler
from keras.layers import Input, Dense, Concatenate, Average, Dropout, Concatenate, Conv1D, GlobalAveragePooling1D, Embedding, GlobalMaxPooling1D, Maximum, Add
from keras.models import Model
from copy import deepcopy
import keras.backend as K
from keras.layers import Layer

from tensorflow.python.framework import ops
import kegra
from kegra.layers.graph import GraphConvolution
from kegra.utils import *

import time
from keras import utils
import tensorflow.keras.backend as K
import keras

experiment = "new_dose-response_matrix_entries"
# Outer test fold to run
outer_fold = 1

seed = 123 # Random seed
n_epochs = 50 # Number of epochs
learning_rate=0.001 # Learning rate of the optimizer
batch_size = 1024 # Batch size
init_std=0.01 # Initial standard deviation
input_type='sparse' # Input type: 'sparse' or 'dense'
order = 5 # Order of the factorization machine (comboFM)
reg = 10**4 # Regularization parameter
rank = 50 # Rank of the factorization

print('GPU available:')
print(tf.test.is_gpu_available())

my_path = 'comboFM_data/data/'

# Features in position 1: Drug A - Drug B
features_tensor_1 = ("drug1_concentration__one-hot_encoding.csv",
                         "drug2_concentration__one-hot_encoding.csv",
                         "drug1__one-hot_encoding.csv",
                         "drug2__one-hot_encoding.csv",
                         "cell_lines__one-hot_encoding.csv")
features_auxiliary_1 = ("drug1_drug2_concentration__values.csv",
                            "cell_lines__gene_expression.csv",
                            "drug1__estate_fingerprints.csv",
                            "drug2__estate_fingerprints.csv")

import pandas as pd
all_SMILES = pd.read_csv(my_path + 'drugs__SMILES.csv')
all_dataset = pd.read_csv(my_path + 'NCI-ALMANAC_subset_555300.csv')
#############################################


# Load your data into a Pandas dataframe
df = all_dataset

# Find the pair of drugs with the maximum number of rows (i.e., maximum number of doses)
max_rows_pair = df.groupby(['Drug1', 'Drug2']).size().idxmax()

# Filter the dataframe to only include the rows for this pair of drugs
max_rows_df = df[(df['Drug1'] == max_rows_pair[0]) & (df['Drug2'] == max_rows_pair[1])]
max_rows_df = df[(df['Drug1'] == max_rows_pair[0]) & (df['Drug2'] == max_rows_pair[1]) |
                 (df['Drug1'] == max_rows_pair[1]) & (df['Drug2'] == max_rows_pair[0])]
#max_rows_df = df[(df['Drug1'] == 'Chlorambucil') & (df['Drug2'] == 'Exemestane')|
#                 (df['Drug2'] == 'Chlorambucil') & (df['Drug1'] == 'Exemestane')]
print(max_rows_pair)
drug1_dose_range = np.linspace(max_rows_df['Conc1'].min(), max_rows_df['Conc1'].max(), 100)
drug2_dose_range = np.linspace(max_rows_df['Conc2'].min(), max_rows_df['Conc2'].max(), 100)
drug1_dose_grid, drug2_dose_grid = np.meshgrid(drug1_dose_range, drug2_dose_range)

# Interpolate the synergy values onto the meshgrid
from scipy.interpolate import griddata
synergy_grid = griddata((max_rows_df['Conc1'], max_rows_df['Conc2']), max_rows_df['PercentageGrowth'], (drug1_dose_grid, drug2_dose_grid), method='nearest')
"""
# Plot the 3D figure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(drug1_dose_grid, drug2_dose_grid, synergy_grid, cmap='viridis')
ax.set_xlabel('Drug 1 Dose')
ax.set_ylabel('Drug 2 Dose')
ax.set_zlabel('Synergy Value')
plt.title('Synergy Value vs. Drug 1 and Drug 2 Doses')
plt.show()
# Plot the first 2D figure: Drug 1 Dose vs. Synergy Value
plt.figure(figsize=(8, 6))
plt.plot(drug1_dose_grid, synergy_grid[:, 50])#, cmap='viridis')
plt.xlabel('Drug 1 Dose')
plt.ylabel('Synergy Value')
plt.title('Synergy Value vs. Drug 1 Dose')
plt.show()

# Plot the second 2D figure: Drug 2 Dose vs. Synergy Value
plt.figure(figsize=(8, 6))
plt.contourf(drug2_dose_grid[50, :], synergy_grid[50, :], cmap='viridis')
plt.xlabel('Drug 2 Dose')
plt.ylabel('Synergy Value')
plt.title('Synergy Value vs. Drug 2 Dose')
plt.show()
# Plot the 3D figure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(max_rows_df['Conc1'], max_rows_df['Conc2'], max_rows_df['PercentageGrowth'])
ax.set_xlabel('Drug 1 Dose')
ax.set_ylabel('Drug 2 Dose')
ax.set_zlabel('Synergy Value')
plt.title('Synergy Value vs. Drug 1 and Drug 2 Doses')
plt.show()

# Plot the first 2D figure: Drug 1 Dose vs. Synergy Value
plt.figure(figsize=(8, 6))
plt.scatter(max_rows_df['Conc1'], max_rows_df['PercentageGrowth'])
plt.xlabel('Drug 1 Dose')
plt.ylabel('Synergy Value')
plt.title('Synergy Value vs. Drug 1 Dose')
plt.show()

# Plot the second 2D figure: Drug 2 Dose vs. Synergy Value
plt.figure(figsize=(8, 6))
plt.scatter(max_rows_df['Conc2'], max_rows_df['PercentageGrowth'])
plt.xlabel('Drug 2 Dose')
plt.ylabel('Synergy Value')
plt.title('Synergy Value vs. Drug 2 Dose')
plt.show()
#####################"""

# set the index to be the names column
all_SMILES.set_index('Drug', inplace=True)


print(len(all_dataset))
idx_drugA = np.zeros((len(all_dataset)))
idx_drugB = np.zeros((len(all_dataset)))
for i in range(0, len(all_dataset)):
    try:
       idx_drugA[i]= all_SMILES.index.get_loc(all_SMILES[all_SMILES.index == all_dataset.loc[i, "Drug1"]].index[0])
    except:
       idx_drugA[i] = 71
    try:
       idx_drugB[i]= all_SMILES.index.get_loc(all_SMILES[all_SMILES.index == all_dataset.loc[i, "Drug2"]].index[0])
    except:
       idx_drugB[i] = 71
#print(all_SMILES.iloc[idx]["SMILE"])

X_tensor_1 = concatenate_features(my_path, features_tensor_1)
X_auxiliary_1 = concatenate_features(my_path, features_auxiliary_1)
X_1 = np.concatenate((X_tensor_1, X_auxiliary_1), axis = 1)

# Features in position 2: Drug B - Drug A
features_tensor_2 = ("drug2_concentration__one-hot_encoding.csv",
                         "drug1_concentration__one-hot_encoding.csv",
                         "drug2__one-hot_encoding.csv",
                         "drug1__one-hot_encoding.csv",
                         "cell_lines__one-hot_encoding.csv")
features_auxiliary_2 =("drug2_drug1_concentration__values.csv",
                            "cell_lines__gene_expression.csv",
                            "drug2__estate_fingerprints.csv",
                            "drug1__estate_fingerprints.csv")
X_tensor_2 = concatenate_features(my_path, features_tensor_2)
X_auxiliary_2 = concatenate_features(my_path, features_auxiliary_2)
X_2 = np.concatenate((X_tensor_2, X_auxiliary_2), axis = 1)


# Concatenate the features from both positions vertically
X = np.concatenate((X_1, X_2), axis=0)
print('Dataset shape: {}'.format(X.shape))
print('Non-zeros rate: {:.05f}'.format(np.mean(X != 0)))
print('Number of one-hot encoding features: {}'.format(X_tensor_1.shape[1]))
print('Number of auxiliary features: {}'.format(X_auxiliary_1.shape[1]))
i_aux = X_tensor_1.shape[1]
del X_tensor_1, X_auxiliary_1, X_tensor_2, X_auxiliary_2, X_1, X_2

# Read responses
y  = np.loadtxt(my_path+"/responses.csv", delimiter = ",", skiprows = 1)
y = np.concatenate((y, y), axis=0)
X_tr = X[0:400000,:]
y_tr = y[0:400000]
X_te = X[400000:,:]
y_te = y[400000:]
idx_drugA1 = np.concatenate(((idx_drugA, idx_drugB)), axis=0)#(idx_drugA, idx_drugB)
idx_drugB1 = np.concatenate(((idx_drugB, idx_drugA)), axis=0)#(idx_drugA, idx_drugB)
idx_drugA1_tr, idx_drugA1_te, idx_drugB1_tr, idx_drugB1_te = idx_drugA1[0:400000], idx_drugA1[400000:], idx_drugB1[0:400000], idx_drugB1[400000:]
# Read cross-validation folds and divide the data
"""te_idx = np.loadtxt('cross-validation_folds/%s/test_idx_outer_fold-%d.txt'%(experiment, outer_fold)).astype(int)
tr_idx = np.loadtxt('cross-validation_folds/%s/train_idx_outer_fold-%d.txt'%(experiment, outer_fold)).astype(int)

X_tr, X_te, y_tr, y_te = X[tr_idx,:], X[te_idx,:], y[tr_idx], y[te_idx]



idx_drugA1_tr, idx_drugA1_te, idx_drugB1_tr, idx_drugB1_te = idx_drugA1[tr_idx,:], idx_drugA1[te_idx,:], idx_drugB1[tr_idx], idx_drugB1[te_idx]

print('Training set shape: {}'.format(X_tr.shape))
print('Test set shape: {}'.format(X_te.shape))

# Standardize, i_aux is denotes the index from which the auxiliary descriptors to be standardized start (one-hot encodings should not be standardized)
X_tr, X_te = standardize(X_tr, X_te, i_aux)"""

smiles_dict = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}

def label_smiles(line, MAX_SMI_LEN, smi_ch_ind):
	X = np.zeros(MAX_SMI_LEN)
	for i, ch in enumerate(line[:MAX_SMI_LEN]): #	x, smi_ch_ind, y
		X[i] = smi_ch_ind[ch]

	return X #.tolist()

#!pip install -q keras-nightly
#!pip install tensorflow==2.12.0 --user


def sinusoidal_embedding(x):
    embedding_min_frequency = 1.0
    embedding_max_frequency = 1000.0
    embedding_dims = 32
    frequencies = tf.math.exp(
        tf.linspace(
            tf.math.log(embedding_min_frequency),
            tf.math.log(embedding_max_frequency),
            embedding_dims // 2,
        )
    )
    angular_speeds = tf.cast(2.0 * math.pi * frequencies, "float32")
    embeddings = tf.concat(
        [tf.math.sin(angular_speeds * x), tf.math.cos(angular_speeds * x)], axis=-1
    )
    return embeddings

smiles_max_len = 100
smiles_dict_len = 64

Drug1_input = Input(shape=(smiles_max_len,), dtype='int32',name='drug1_input')
Drug2_input = Input(shape=(smiles_max_len,), dtype='int32',name='drug2_input')


encode_smiles_layer1 = Embedding(input_dim=smiles_dict_len+1, output_dim = 128, input_length=smiles_max_len,name='smiles_embedding')
encode_smiles_layer2  = Conv1D(filters=32, kernel_size=4,  activation='relu', padding='valid',  strides=1, name='conv1_smiles')
encode_smiles_layer3  = Conv1D(filters=32*2, kernel_size=4,  activation='relu', padding='valid',  strides=1, name='conv2_smiles')
encode_smiles_layer4  = Conv1D(filters=32*3, kernel_size=4,  activation='relu', padding='valid',  strides=1, name='conv3_smiles')
encode_smiles_layer5  = GlobalMaxPooling1D()


drug1_layer1 = encode_smiles_layer1(Drug1_input)
drug1_layer2 = encode_smiles_layer2(drug1_layer1)
drug1_layer3 = encode_smiles_layer3(drug1_layer2)
drug1_layer4 = encode_smiles_layer4(drug1_layer3)
drug1_layer5 = encode_smiles_layer5(drug1_layer4)

drug2_layer1 = encode_smiles_layer1(Drug2_input)
drug2_layer2 = encode_smiles_layer2(drug2_layer1)
drug2_layer3 = encode_smiles_layer3(drug2_layer2)
drug2_layer4 = encode_smiles_layer4(drug2_layer3)
drug2_layer5 = encode_smiles_layer5(drug2_layer4)

cells_dict_len = 78
Cell_input = Input(shape=(cells_dict_len,),  name = 'cell_input')
encode_cell_layer1 = Dense(32, activation = 'relu')(Cell_input)
encode_cell_layer2 = Dense(32*2, activation = 'relu')(encode_cell_layer1)
encode_cell_layer3 = Dense(32*3, activation = 'relu')(encode_cell_layer2)

concentration_drug1_input = keras.Input(shape=(1,))
concentration_drug2_input = keras.Input(shape=(1,))

concentration_drug1 = layers.Lambda(sinusoidal_embedding, output_shape=(32,))(concentration_drug1_input)
concentration_drug2 = layers.Lambda(sinusoidal_embedding, output_shape=(32,))(concentration_drug2_input)

embedding = layers.Concatenate()([drug1_layer5, drug2_layer5, encode_cell_layer3, concentration_drug1, concentration_drug2])

model_embedding = Model(inputs=[Drug1_input, Drug2_input, Cell_input, concentration_drug1_input, concentration_drug2_input], outputs=[embedding])
model_embedding.summary()
"""FC1 = Dense(1024, activation='relu', name='dense1_')(embedding)
FC2 = Dropout(0.1)(FC1)
FC2 = Dense(1024, activation='relu', name='dense2_')(FC2)
FC2 = Dropout(0.1)(FC2)"""

embedding_input = Input(shape=(352,), dtype='int32',name='embedding_input')
FC2 = Dense(512, activation='relu', name='dense3_')(embedding_input)
predictions = Dense(1, name='dense4_')(FC2)

model_task = Model(inputs = [embedding_input], outputs = [predictions])

output = model_task(model_embedding([Drug1_input, Drug2_input, Cell_input, concentration_drug1_input, concentration_drug2_input]))
model_pred = Model(inputs=[Drug1_input, Drug2_input, Cell_input, concentration_drug1_input, concentration_drug2_input], outputs=[output])

adam=Adam(learning_rate=0.0001)

def generate_data(batch_size):
  i_c=0
  while True:
        input1 = []
        input2 = []

        output1 = []
        batch_counter=0

        if i_c>=len(X_tr)-1:
          i_c=0

        drug1 = []
        drug2 = []
        for j in range(i_c,(i_c+batch_size)):
            drug1.append(label_smiles(all_SMILES.iloc[int(idx_drugA1_tr[j])][0],
                             smiles_max_len, smiles_dict))
        for j in range(i_c,(i_c+batch_size)):
            drug2.append(label_smiles(all_SMILES.iloc[int(idx_drugB1_tr[j])][0],
                             smiles_max_len, smiles_dict))

        cell_line = X_tr[i_c:(i_c+batch_size), 254:(254+78)]
        concentration_drug1 = X_tr[i_c:(i_c+batch_size), 252]
        concentration_drug2 = X_tr[i_c:(i_c+batch_size), 253]
        #in_prots = np.array(val_prots_t[i_c:(i_c+batch_size)])
        lbl = y_tr[i_c:(i_c+batch_size),]
        i_c = i_c + batch_size
        yield (np.array(drug1), np.array(drug2), np.array(cell_line), np.array(concentration_drug1), np.array(concentration_drug2)), (np.array(lbl))

def generate_data_val(batch_size):
  i_c=0
  while True:
        input1 = []
        input2 = []

        output1 = []
        batch_counter=0

        if i_c>=len(X_te)-1:
          i_c=0

        drug1 = []
        drug2 = []
        for j in range(i_c,(i_c+batch_size)):
            drug1.append(label_smiles(all_SMILES.iloc[int(idx_drugA1_te[j])][0],
                             smiles_max_len, smiles_dict))
        for j in range(i_c,(i_c+batch_size)):
            drug2.append(label_smiles(all_SMILES.iloc[int(idx_drugB1_te[j])][0],
                             smiles_max_len, smiles_dict))

        cell_line = X_te[i_c:(i_c+batch_size), 254:(254+78)]
        concentration_drug1 = X_te[i_c:(i_c+batch_size), 252]
        concentration_drug2 = X_te[i_c:(i_c+batch_size), 253]
        #in_prots = np.array(val_prots_t[i_c:(i_c+batch_size)])
        lbl = y_te[i_c:(i_c+batch_size), ]
        yield (np.array(drug1), np.array(drug2), np.array(cell_line), np.array(concentration_drug1), np.array(concentration_drug2)), (np.array(lbl))

import tensorflow as tf
#import tensorflow.compat.v1 as tf_v1
import tensorflow_probability as tfp
def tf_pearson(y_true, y_pred):
    return tfp.stats.correlation(y_pred, y_true)

model_pred.compile(optimizer=adam, loss=['mse'] , metrics=['mse'])
model_pred.fit(generate_data(batch_size=512), epochs=1, steps_per_epoch=int(len(X_tr)/512),
               shuffle=False,  validation_data = generate_data_val(batch_size=512),validation_steps= int(len(X_te)/512),
               )



import pandas as pd

df = all_dataset.loc[:,'CellLine']
unique_cells =  df.drop_duplicates( ignore_index=False)

df = all_dataset.loc[:,['Drug1', 'Drug2','CellLine']]
unique_combinations =  df.drop_duplicates( ignore_index=False)

ans = all_dataset.loc[all_dataset.loc[:, 'Drug1':'CellLine'].eq(unique_combinations.loc[0,:]).all(axis=1)]

from sklearn.neighbors import NearestNeighbors
knn = NearestNeighbors(n_neighbors=3)




def generate_data(batch_size, counter):
  i_c = counter * batch_size
  while True:
        input1 = []
        input2 = []

        output1 = []
        batch_counter=0

        if i_c>=len(unique_combinations):
          i_c=0

        drug1 = []
        drug2 = []
        cell_line = []
        concentration_drug1 = []
        concentration_drug2 = []
        lbl = []
        # unique_combinations_ = unique_combinations.drop(i_c)
        ans = all_dataset.loc[all_dataset.loc[:, 'Drug1':'CellLine'].eq(unique_combinations.loc[i_c,:]).all(axis=1)]
        for k in range(0, len(ans)):
          try:
              drug1.append(label_smiles(all_SMILES.iloc[int(idx_drugA1_tr[ans.index[k]])][0],
                             smiles_max_len, smiles_dict))
              drug2.append(label_smiles(all_SMILES.iloc[int(idx_drugB1_tr[ans.index[k]])][0],
                             smiles_max_len, smiles_dict))

              cell_line.append(X_tr[ans.index[k], 254:(254+78)])
              concentration_drug1.append(X_tr[ans.index[k], 252])
              concentration_drug2.append(X_tr[ans.index[k], 253])
              #in_prots = np.array(val_prots_t[i_c:(i_c+batch_size)])
              lbl.append(y_tr[ans.index[k],])
              
          except:
              print("") # Do nothing

        if True:
          features = model_embedding.predict([np.array(drug1), np.array(drug2), np.array(cell_line)
                                      , np.array(concentration_drug1), np.array(concentration_drug2)])
          lbl_pred = model_pred.predict([np.array(drug1), np.array(drug2), np.array(cell_line)
                                      , np.array(concentration_drug1), np.array(concentration_drug2)])
          lbl_pred = np.array(lbl_pred)
          lbl_pred = np.reshape(lbl_pred, (-1,1))
          #for h in range(0, len(concentration_drug1))
          knn.fit(features)
          out_idx = knn.kneighbors(features, return_distance=False)
          print(features.shape)
          Adj = np.zeros((out_idx.shape[0], out_idx.shape[0]))
          for jj in range(out_idx.shape[0]):
            for kk in range(0,out_idx.shape[1]):
              Adj[jj,out_idx[jj,kk]] = 1

          Adj = Adj+np.eye(Adj.shape[0])
          d = np.diag(np.power(np.array(Adj.sum(1)), -0.5).flatten(), 0)
          Adj = Adj.dot(d).transpose().dot(d)
          i_c = i_c + 1
          yield (np.array(lbl_pred), np.array(features), np.array(Adj)), np.array(lbl)

def generate_data_val(batch_size, counter):
  i_c = counter * batch_size
  while True:
        input1 = []
        input2 = []

        output1 = []
        batch_counter=0

        if i_c>=len(unique_combinations):
          i_c=0

        drug1 = []
        drug2 = []
        cell_line = []
        concentration_drug1 = []
        concentration_drug2 = []
        lbl = []
        # unique_combinations_ = unique_combinations.drop(i_c)
        ans = all_dataset.loc[all_dataset.loc[:, 'Drug1':'CellLine'].eq(unique_combinations.loc[i_c,:]).all(axis=1)]
        for k in range(0, len(ans)):
          try:
              drug1.append(label_smiles(all_SMILES.iloc[int(idx_drugA1_te[ans.index[k]])][0],
                             smiles_max_len, smiles_dict))
              drug2.append(label_smiles(all_SMILES.iloc[int(idx_drugB1_te[ans.index[k]])][0],
                             smiles_max_len, smiles_dict))

              cell_line.append(X_te[ans.index[k], 254:(254+78)])
              concentration_drug1.append(X_te[ans.index[k], 252])
              concentration_drug2.append(X_te[ans.index[k], 253])
              #in_prots = np.array(val_prots_t[i_c:(i_c+batch_size)])
              lbl.append(y_te[ans.index[k],])
          except:
              print("") # Do nothing

        features = model_embedding.predict([np.array(drug1), np.array(drug2), np.array(cell_line)
                                    , np.array(concentration_drug1), np.array(concentration_drug2)])
        lbl_pred = model_pred.predict([np.array(drug1), np.array(drug2), np.array(cell_line)
                                    , np.array(concentration_drug1), np.array(concentration_drug2)])
        lbl_pred = np.array(lbl_pred)
        lbl_pred = np.reshape(lbl_pred, (-1,1))
        #for h in range(0, len(concentration_drug1))
        knn.fit(features)
        out_idx = knn.kneighbors(features, return_distance=False)
        print(features.shape)
        Adj = np.zeros((out_idx.shape[0], out_idx.shape[0]))
        for jj in range(out_idx.shape[0]):
          for kk in range(0,out_idx.shape[1]):
            Adj[jj,out_idx[jj,kk]] = 1

        Adj = Adj+np.eye(Adj.shape[0])
        d = np.diag(np.power(np.array(Adj.sum(1)), -0.5).flatten(), 0)
        Adj = Adj.dot(d).transpose().dot(d)
        i_c = i_c + 1
        yield (np.array(lbl_pred), np.array(features), np.array(Adj)), np.array(lbl)

# Normalize X
#X = np.diag(1./np.array(X.sum(1)).flatten()).dot(X)

X_in = Input(shape=(352,))
G = [Input(batch_shape=(None, None))]
lbl_pred = Input(shape=(1,))
# Define model architecture
# NOTE: We pass arguments for graph convolutional layers as a list of tensors.
# This is somewhat hacky, more elegant options would require rewriting the Layer base class.
H = Dropout(0.5)(X_in)
H = GraphConvolution(16,  activation='relu')([H]+G)
H = Dropout(0.5)(H)
Y = GraphConvolution(1)([H]+G)
lbl_pred = layers.Flatten()(lbl_pred)
Y =  Concatenate()([Y, lbl_pred])
Y = Dense(1)(Y)
print(model.summary())
# Compile model
model = Model(inputs=[lbl_pred, X_in, G], outputs=Y)
model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))

nb_epochs = 3 # set the number of epochs 
for ii in range (0, nb_epochs):   
  for i_c in range(0, len(X_tr)):  
      model.fit(generate_data(1, i_c), steps_per_epoch = 1, epochs=1)


y_pred = []
for i_c in range(0, len(X_te)): 
    y_pred.append(model.predict(generate_data_val(1, i_c), steps= len(X_te )))

correlation_coefficient = np.corrcoef(y_te, y_pred)[0, 1]

