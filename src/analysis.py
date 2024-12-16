# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 14:16:19 2022

@author: Mehdi Orouji
"""
import os
import random as rn
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.utils import shuffle
from tensorflow.keras.utils import to_categorical
from numpy.random import seed
seed(2)
import keras
from keras.losses import binary_crossentropy, categorical_crossentropy,MeanSquaredError
from keras import backend as K
from sklearn.preprocessing import StandardScaler
import tensorflow
tensorflow.random.set_seed(42)
import IPython as IP
IP.get_ipython().magic('reset -f')
from tensorflow.keras import optimizers
from keras.layers import Input, Dense, Dropout, Lambda
from keras.models import Model
from keras.regularizers import l2
#from keras.utils.vis_utils import plot_model
from keras.datasets import mnist, fashion_mnist
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.compat.v1 as tf

###############################################################################

def load_fMRI(subject, dim, net):

    '''
    This function loads the fMRI decoded data found by our models by taking 3 arguments and returns 5 outputs

    Parameters
    ----------
    1. subject: which subject?
    2. dim: what dimension in the bottlneck?
    3. net: the type of network that was used to generate the result?

    Returns
    -------
    1. accuracy: the accuracy of the attached classifier to the bottleneck
    2. X_test: the suffled input data
    3. decoded: decoded version of input
    4. BN: encoded bottleneck features
    5. map: the mapping since the input data is shuffled before feeding into the network
    '''
    source = '/content/gdrive/MyDrive/Colab Notebooks/VE_manuscript/Archive/dim_reduc_project/AE_'+net+'/fMRI/s'
    accuracy = pd.read_csv(source + str(subject)+ '/accuVsdim.csv', header=None ).values
    X_test = pd.read_csv(source +  str(subject)+ '/X_test_'+str(dim)+'.csv', header=None ).values
    decoded = pd.read_csv(source + str(subject)+ '/decoded_VT_'+str(dim)+'.csv', header=None ).values
    BN = pd.read_csv(source + str(subject)+ '/BN_dim_'+str(dim)+'.csv', header=None ).values
    map = pd.read_csv(source +  str(subject)+ '/map'+str(dim)+'.csv', header=None ).values
    return(accuracy, X_test, decoded, BN, map)

###############################################################################

def truncate (X,y, remove):

  '''
  this funciton shuffle X(input) and y(labels)
  then tunrcate X and correspondint labels y
  and returns the shuffled and truncated version of X, y

  Parameters
  ----------
  1. X: input features
  2. y: labels
  3. remove: the ration being removed

  Returns
  -------
  1. X_truncated: truncated version of X
  2. y_truncated: truncated version of y
  3. map_truncated: the shuffling map
  '''

  length=len(y)
  keep = int((1-remove)*length+1)
  maped = range(length)
  #X_shffled, y_shuffled, maped = shuffle(X, y, maped, random_state=42)
  #X_truncated = X_shffled[:keep,:]
  #y_truncated = y_shuffled[:keep]
  map_truncated = maped[:keep]

  X_truncated = X[:keep,:]
  y_truncated = y[:keep]
  return X_truncated, y_truncated, map_truncated

###############################################################################

def which_mnist(select_data, remove):
  '''
  this function loads, preprocess, and truncate mnist and fashion mnist datasets

  Parameters
  ----------
  1. select_data: mnist or fashion_mnist
  2. remove: the percentage for truncating the dataset

  Returns
  -------
  1. X_train: truncated version of features in training set
  2. y_categ_train: truncated version of labels in training set
  3. X_test: test set
  4. y_categ_test: categorical repsentation of labels in test set
  '''

  keep = (1-remove)
  if select_data == 'mnist':
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    [X_train, y_train, map1] = turncate (X_train,y_train, keep)
    X_train = np.reshape(X_train/255,(X_train.shape[0],X_train.shape[1]**2))
    X_test = np.reshape(X_test/255,(X_test.shape[0],X_test.shape[1]**2))
  elif select_data == 'fashion_mnist':
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    [X_train, y_train, map1] = turncate (X_train,y_train, keep)
    X_train = np.reshape(X_train/255,(X_train.shape[0],X_train.shape[1]**2))
    X_test = np.reshape(X_test/255,(X_test.shape[0],X_test.shape[1]**2))

  classes = np.unique(y_test)
  class_number = len(classes)
  y_categ_train = to_categorical(y_train, num_classes= class_number)
  y_categ_test = to_categorical(y_test, num_classes=class_number)
  return(X_train, y_categ_train, X_test, y_categ_test)

###############################################################################

def dataframed(corr, test_label):

  '''
  This function takes in a numpy matrix i.e. corr and returns a pandas dataframe i.e. corr_df
  which rows and columns are labeled by the categories (i.e. test_label)

  Parameters
  ----------
  1. corr: pair-wise correlation matrix
  2. test_label: labels of trials


  Returns
  -------
  1. corr_df: converting the correlation matrix "corr" from numpy array
     to pandas dataframe
  '''

  num_trials = len(test_label)
  diag_index = range(num_trials) # to find the index of the diagonals of the test-set matrix
  corr_zero = np.copy(corr)
  #corr_zero[diag_index, diag_index]=0 # to remove the auto-correlaiton of the trials
  corr_df = pd.DataFrame(corr_zero) # turning it to DataFrame to make my life easier by ALOT
  corr_df.columns=test_label # to name the columns of the correlation
  corr_df.index = test_label # to name the rows of the correlation

  return(corr_df)

###############################################################################

def class_diagonal(corr, labels):

  '''
  this function gets a correlation matrix, make the main diogonal of matrix i.e.
  the autocorrelaiton zero and then calculated the average correlation within
  and between each class

  Parameters
  ----------
  1. corr: the correlation matrix
  2. num_trails: the total number of samples
  3. labels: the label of samples in the correlation matix

  Returns
  -------
  class_matrix_corr: which is a categ X categ matrix containing the average
  of the correlaiton of all samples based on within and between category
  '''

  corr_df = dataframed(corr, labels) # calling the Dataframed function
  categ = np.unique(labels) # to find what uniqe classes we have in the test-set
  class_matrix_corr = np.zeros((len(categ),len(categ))) # this is the correlation of the mean values of the classes

  for i, row in enumerate(categ):
    for j, column in enumerate(categ):
      if i==j:
        n_examples = np.sqrt(corr_df.loc[row][column].size) # this is for the case of autocorrelation that we have already put zeros for them so it is fair to not consider them while getting the mean
        class_matrix_corr [i,j] = corr_df.loc[row][column].sum().sum()/(n_examples**2 - n_examples)
      else:
        class_matrix_corr [i,j]= corr_df.loc[row][column].mean().mean()
  return(class_matrix_corr)

###############################################################################

def specificity (corr, y_test):

    '''
    This function calculates the class specificity for a given correlation matrix

    Parameters
    ----------
    1. corr: pairwise correlation matrix of all trials
    2. y_test: labels of all trials

    Returns
    -------
    1. specificity: class specificity

    '''
    diag_index = range(len(y_test)) # to find the index of the diagonals of the test-set matrix
    corr_copy = np.copy(corr)
    corr_copy[diag_index, diag_index]=0 # to remove the auto-correlaiton of the trials

    class_corr = class_diagonal(corr, y_test) # calling another function
    within_class_corr = np.diagonal(class_corr).mean()
    #to calculate the off_diag_corr for reconstuct"
    off_diag_indx = np.where(~np.eye(class_corr.shape[0],dtype=bool))
    between_class_corr = class_corr[off_diag_indx].mean()

    specificity = within_class_corr - between_class_corr
    return specificity
###############################################################################

def specificity_exp (corr, y_test, cetg_arange):

    '''
    This function calculates the class specificity for a given correlation matrix

    Parameters
    ----------
    1. corr: pairwise correlation matrix of all trials
    2. y_test: labels of all trials

    Returns
    -------
    1. specificity: class specificity

    '''
    '''
    This function calculates the class specificity for a given correlation matrix

    Parameters
    ----------
    1. corr: pairwise correlation matrix of all trials
    2. y_test: labels of all trials

    Returns
    -------
    1. specificity: class specificity

    '''
    num_trials = len(y_test)
    diag_index = range(num_trials) # to find the index of the diagonals of the test-set matrix
    corr_zero = np.copy(corr)
    np.fill_diagonal(corr_zero, 0) # to remove the auto-correlaiton of the trials
    corr_df = pd.DataFrame(corr_zero, index = y_test, columns=y_test) # turning it to DataFrame to make my life easier by ALOT

    categ = np.unique(y_test) # to find what uniqe classes we have in the test-set
    class_matrix_corr = np.zeros((len(categ),len(categ))) # this is the correlation of the mean values of the classes
    for i, row in enumerate(cetg_arange):
      for j, column in enumerate(cetg_arange):
        if row==column:
          n_examples = np.sqrt(corr_df.loc[row, column].size) # this is for the case of autocorrelation that we have already put zeros for them so it is fair to not consider them while getting the mean
          class_matrix_corr [i,j] = corr_df.loc[row, column].sum().sum()/(n_examples**2 - n_examples)
        else:
          class_matrix_corr [i,j]= corr_df.loc[row, column].mean().mean()

    within_class_corr = np.diagonal(class_matrix_corr).mean()
    #to calculate the off_diag_corr for reconstuct"
    off_diag_indx = np.where(~np.eye(class_matrix_corr.shape[0],dtype=bool))
    between_class_corr = class_matrix_corr[off_diag_indx].mean()

    specificity = within_class_corr - between_class_corr
    return specificity
###############################################################################

def AE(encoding_dim, X_train, patience=25, best=False):

    '''
    This funcction is written to implement a deep AE with 3 hidden layers

    Parameters
    ----------
    1. encoding_dim: the size of the BN features
    2. X_train: the input featrues

    Returns
    -------
    1. decoder: returens the decoder model
    2. encoder: returns the encoder model
    3. bn_model: returns the weights to calculate BN layer only
    4. names: returns the name of the layers
    5. retuns all the parametes of the network
    '''

    drop = 0.02
    activation_BT = 'linear'
    learning_rate = 1e-4
    ES = EarlyStopping(monitor='val_loss', mode='min', patience=patience, start_from_epoch=50, restore_best_weights=best)
    # this is our input placeholder for VT (creating a tensor)
    decoding_vox = Input(shape=(X_train.shape[1],), name="VT-input")
    encoded = Dense(512, activation='tanh', name="encoded_VT")(decoding_vox)
    encoded= Dropout(drop)(encoded)

    Bneck = Dense(encoding_dim, activation=activation_BT, name="Bneck")(encoded)

    decoded = Dense(512, activation='tanh', name="decoded-VT")(Bneck)
    decoded= Dropout(drop)(decoded)
    decoded = Dense(X_train.shape[1], activation='linear', name="decoded-output")(decoded)
    decoded= Dropout(drop)(decoded)

    encoder = Model(decoding_vox, Bneck)
    decoder = Model(Bneck, decoded) # model architecture
    AE = Model(decoding_vox, decoded)
    #plot_model(decoder, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    #bn_model = Model(inputs= decoder.input, outputs = decoder.get_layer('Bneck').output)

    # fitting the model and defining loss functiojn for each imput
    opt = optimizers.Adam(learning_rate)
    AE.compile(opt, loss='mean_squared_error')
    print('training AE for drop= '+ str(drop) + ', activation_BT= ' + activation_BT + ', learning_rate= '+ str(learning_rate))
    #return decoder, encoder, bn_model, Bneck
    return AE, decoder, encoder, ES

###############################################################################

def TRACE(encoding_dim, X_train, num_classes, reg=True, reg_size= 0.0001, patience=25, best=False ):

    '''
    This funcction is written to implement the TRACE network with 3 hidden layers
    and a classifier attached to the BN

    Parameters
    ----------
    1. encoding_dim: the size of the BN features
    2. X_train: the input featrues
    3. num_classes: how many unique class exist

    Returns
    -------
    1. decoder: returens the decoder model
    2. encoder: returns the encoder model
    3. bn_model: returns the weights to calculate BN layer only
    4. names: returns the name of the layers
    5. retuns all the parametes of the network
    '''

    drop = 0.02
    activation_BT = 'linear'
    learning_rate = 1e-4
    # this is our input placeholder for VT (creating a tensor)
    decoding_vox = Input(shape=(X_train.shape[1],), name="VT-input")
    encoded = Dense(512, activation='tanh', name="encoded_VT")(decoding_vox)
    encoded= Dropout(drop)(encoded)

    ES = EarlyStopping(monitor='val_loss', mode='min', patience=patience, start_from_epoch=50, restore_best_weights=best)
    Bneck = Dense(encoding_dim, activation=activation_BT, name="Bneck")(encoded)
    if reg == True:
      classifier1 = Dense(num_classes, activation='softmax', name='classifier', kernel_regularizer=l2(reg_size))(Bneck)
    else:
      classifier1 = Dense(num_classes, activation='softmax', name='classifier')(Bneck)

    decoded = Dense(512, activation='tanh', name="decoded-VT")(Bneck)
    decoded= Dropout(drop)(decoded)
    decoded = Dense(X_train.shape[1], activation='linear', name="decoded-output")(decoded)
    decoded= Dropout(drop)(decoded)

    encoder = Model(decoding_vox, Bneck)
    decoder = Model(decoding_vox, [decoded, classifier1]) # model architecture
    #plot_model(decoder, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    #bn_model = Model(inputs= decoder.input, outputs = decoder.get_layer('Bneck').output)

    opt = optimizers.Adam(learning_rate)
    decoder.compile(opt, loss=['mean_squared_error', 'categorical_crossentropy'], loss_weights=[1, 0.01])
    print('training TRACE for drop= '+ str(drop) + ', activation_BT= ' + activation_BT + ', learning_rate= '+str(learning_rate))

    #return decoder, encoder, bn_model, Bneck
    return decoder, encoder, ES


 ###############################################################################

def R2R(encoding_dim, XVT_train, XPFC_train, num_classes, reg=True, reg_size= 0.0001, patience=25, best=False ):

    '''
    This funcction is written to implement the TRACE network with 3 hidden layers
    and a classifier attached to the BN to map one RIO to another ROI(e.g. VTC to PFC )

    Parameters
    ----------
    1. encoding_dim: the size of the BN features
    2. X_train: the input featrues
    3. num_classes: how many unique class exist

    Returns
    -------
    1. decoder: returens the decoder model
    2. encoder: returns the encoder model
    3. bn_model: returns the weights to calculate BN layer only
    4. names: returns the name of the layers
    5. retuns all the parametes of the network
    '''

    drop = 0.02
    activation_BT = 'linear'
    learning_rate = 1e-4
    ES = EarlyStopping(monitor='val_loss', mode='min', patience=patience, start_from_epoch=50, restore_best_weights=best)
    # this is our input placeholder for VT (creating a tensor)
    decoding_vox = Input(shape=(XVT_train.shape[1],), name="VT-input")
    encoded = Dense(512, activation='tanh', name="encoded_VT")(decoding_vox)
    encoded= Dropout(drop)(encoded)

    Bneck = Dense(encoding_dim, activation=activation_BT, name="Bneck")(encoded)
    if reg == True:
      classifier1 = Dense(num_classes, activation='softmax', name='classifier', kernel_regularizer=l2(reg_size))(Bneck)
    else:
      classifier1 = Dense(num_classes, activation='softmax', name='classifier')(Bneck)

    decoded = Dense(512, activation='tanh', name="decoded-VT")(Bneck)
    decoded= Dropout(drop)(decoded)
    decoded = Dense(XPFC_train.shape[1], activation='linear', name="decoded-output")(decoded)
    decoded= Dropout(drop)(decoded)

    encoder = Model(decoding_vox, Bneck)
    decoder = Model(decoding_vox, [decoded, classifier1]) # model architecture

    encoder = Model(decoding_vox, Bneck)
    Rgn2Rgn = Model(decoding_vox, decoded)

    opt = optimizers.Adam(learning_rate)
    decoder.compile(opt, loss=['mean_squared_error', 'categorical_crossentropy'], loss_weights=[1, 0.01])
    print('training TRACE for drop= '+ str(drop) + ', activation_BT= ' + activation_BT + ', learning_rate= '+str(learning_rate))

    return decoder, encoder, ES

###############################################################################
# Define sampling with reparameterization trick
def sample_z(args):
  mu, sigma = args
  batch     = K.shape(mu)[0]
  dim       = K.int_shape(mu)[1]
  reset_seed(SEED = 12345)
  eps       = K.random_normal(shape=(batch, dim))
  return mu + K.exp(sigma / 2) * eps



def sampling(args):
    z_mean, z_log_sigma = args
    reset_seed(SEED = 12345)
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], K.shape(z_mean)[1]),
                              mean=0., stddev=0.1)
    return z_mean + K.exp(z_log_sigma) * epsilon
###############################################################################
# denoising autoencoder
def DAE(encoding_dim, X_train, drop=0.02, noise_sd = 0.1, patience=25, best=False):
    drop = 0.02
    activation_BT = 'linear'
    learning_rate = 1e-4
    ES = EarlyStopping(monitor='val_loss', mode='min', patience=patience, start_from_epoch=50, restore_best_weights=best)
    # this is our input placeholder for VT (creating a tensor)
    decoding_vox = Input(shape=(X_train.shape[1],), name="VT-input")
    decoding_vox_noise = keras.layers.GaussianNoise(noise_sd)(decoding_vox)
    encoded = Dense(512, activation='tanh', name="encoded_VT")(decoding_vox_noise)
    encoded= Dropout(drop)(encoded)

    Bneck = Dense(encoding_dim, activation=activation_BT, name="Bneck")(encoded)

    decoded = Dense(512, activation='tanh', name="decoded-VT")(Bneck)
    decoded= Dropout(drop)(decoded)
    decoded = Dense(X_train.shape[1], activation='linear', name="decoded-output")(decoded)
    decoded= Dropout(drop)(decoded)

    encoder = Model(decoding_vox, Bneck)
    decoder = Model(Bneck, decoded) # model architecture
    AE = Model(decoding_vox, decoded)
    #plot_model(decoder, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    #bn_model = Model(inputs= decoder.input, outputs = decoder.get_layer('Bneck').output)

    # fitting the model and defining loss functiojn for each imput
    opt = optimizers.Adam(learning_rate)
    AE.compile(opt, loss='mean_squared_error')
    print('training AE for drop= '+ str(drop) + ', activation_BT= ' + activation_BT + ', learning_rate= '+ str(learning_rate))
    #return decoder, encoder, bn_model, Bneck
    return AE, decoder, encoder, ES

###############################################################################
def loss_func(reconstruction_loss, kl_loss):
  # The loss function has to be written seperately to be able
  # to use early stopping
  loss = K.mean(reconstruction_loss + kl_loss)
  return loss


def VAE(encoding_dim, X_train, drop=0.02, patience=25, best=False):
  activation_BT = 'linear'
  learning_rate = 1e-4
  ES = EarlyStopping(monitor='val_loss_func', mode='min', patience=patience, start_from_epoch=50, restore_best_weights=best)
  # this is our input placeholder for VT (creating a tensor)
  inputs = Input(shape=(X_train.shape[1],))
  encoded = Dense(512, activation='tanh', name='encode1')(inputs)
  encoded= Dropout(drop)(encoded)
  z_mean = Dense(encoding_dim)(encoded)
  z_log_sigma = Dense(encoding_dim)(encoded)
  z = Lambda(sampling)([z_mean, z_log_sigma])
  encoder = Model(inputs, [z_mean, z_log_sigma, z], name='encoder')

  # Create decoder
  latent_inputs = Input(shape=(encoding_dim,), name='z_sampling')
  decoded = Dense(512, activation='tanh', name='decode1')(latent_inputs)
  decoded= Dropout(drop)(decoded)
  outputs = Dense(X_train.shape[1], activation='sigmoid')(decoded)
  decoder = Model(latent_inputs, outputs, name='decoder')

  # instantiate VAE model
  Bneck = encoder(inputs)[2]
  outputs = decoder(Bneck)
  vae = Model(inputs, outputs, name='vae_mlp')

  reconstruction_loss = binary_crossentropy(inputs, outputs)
  reconstruction_loss *= X_train.shape[1]
  kl_loss = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
  kl_loss = K.sum(kl_loss, axis=-1)
  kl_loss *= -0.5
  vae_loss = loss_func(reconstruction_loss, kl_loss)
  vae.add_loss(vae_loss)
  opt = optimizers.Adam(learning_rate)
  vae.compile(opt, metrics=[loss_func])
  return vae, decoder, encoder, ES

###############################################################################
def VAE0(encoding_dim, X_train, drop=0.02):
  activation_BT = 'linear'
  learning_rate = 1e-4
  # this is our input placeholder for VT (creating a tensor)
  inputs = Input(shape=(X_train.shape[1],))
  encoded = Dense(512, activation='tanh', name='encode1')(inputs)
  encoded= Dropout(drop)(encoded)
  z_mean = Dense(encoding_dim)(encoded)
  z_log_sigma = Dense(encoding_dim)(encoded)
  z = Lambda(sampling)([z_mean, z_log_sigma])
  encoder = keras.Model(inputs, [z_mean, z_log_sigma, z], name='encoder')

  # Create decoder
  latent_inputs = Input(shape=(encoding_dim,), name='z_sampling')
  decoded = Dense(512, activation='tanh', name='decode1')(latent_inputs)
  decoded= Dropout(drop)(decoded)
  outputs = Dense(X_train.shape[1], activation='linear')(decoded)
  decoder = Model(latent_inputs, outputs, name='decoder')

  # instantiate VAE model
  Bneck = encoder(inputs)[2]
  outputs = decoder(Bneck)
  vae = Model(inputs, outputs, name='vae_mlp')

  reconstruction_loss = MSE(inputs, outputs)
  #reconstruction_loss *= X_train.shape[1]
  kl_loss = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
  kl_loss = K.sum(kl_loss, axis=-1)
  kl_loss *= -0.5
  vae_loss = K.mean(reconstruction_loss + kl_loss)
  vae.add_loss(vae_loss)
  vae.compile(optimizer='Adam')
  return vae, decoder, encoder, Bneck
###############################################################################

def VAE_fmri(encoding_dim, X_train, drop=0.02, patience=25, best=False):
  activation_BT = 'linear'
  learning_rate = 1e-4
  ES = EarlyStopping(monitor='val_loss_func', mode='min', patience=patience, start_from_epoch=50, restore_best_weights=best)
  # this is our input placeholder for VT (creating a tensor)
  inputs = Input(shape=(X_train.shape[1],))
  encoded = Dense(512, activation='tanh', name='encode1')(inputs)
  encoded= Dropout(drop)(encoded)
  z_mean = Dense(encoding_dim)(encoded)
  z_log_sigma = Dense(encoding_dim)(encoded)
  z = Lambda(sampling)([z_mean, z_log_sigma])
  encoder = Model(inputs, [z_mean, z_log_sigma, z], name='encoder')

  # Create decoder
  latent_inputs = Input(shape=(encoding_dim,), name='z_sampling')
  decoded = Dense(512, activation='tanh', name='decode1')(latent_inputs)
  decoded= Dropout(drop)(decoded)
  outputs = Dense(X_train.shape[1], activation='linear')(decoded)
  decoder = Model(latent_inputs, outputs, name='decoder')

  # instantiate VAE model
  Bneck = encoder(inputs)[2]
  outputs = decoder(Bneck)
  vae = Model(inputs, outputs, name='vae_mlp')

  reconstruction_loss = MSE(inputs, outputs)
  reconstruction_loss *= X_train.shape[1]
  kl_loss = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
  kl_loss = K.sum(kl_loss, axis=-1)
  kl_loss *= -0.5
  vae_loss = loss_func(reconstruction_loss, kl_loss)
  vae.add_loss(vae_loss)
  opt = optimizers.Adam(learning_rate)
  vae.compile(opt, metrics=[loss_func])
  return vae, decoder, encoder, ES
###############################################################################

def log_classifier(X_scl, num_classes, patience=25, l_rate=5e-5, reg_size=0.007):

    '''
    Parameters
    ----------
    X_scl : numpy
        Input feautes.
    classes : string
        the name of the classes.

    Returns
    -------
    classifier : keras object
        the classifer model.

    '''
    ES = EarlyStopping(monitor='val_accuracy', mode='max', patience=patience, start_from_epoch=50)
    feat = Input(shape=(X_scl.shape[1],), name="input dim")
    out_class = Dense(num_classes, activation='softmax', name ="output", kernel_regularizer=l2(reg_size))(feat)
    classifier = Model(feat, out_class, name='classifier')
    opt = optimizers.Adam(l_rate)
    classifier.compile(opt, loss = 'categorical_crossentropy',  metrics =['accuracy'])
    #print('training logistic classifier for learning_rate= '+ str(learning_rate) + ', catergory lenght= '+ str(num_classes))
    return classifier, ES

###############################################################################

def fidelity (corr):

  '''
  This function returns reconstruciton fidelity

  Parameters
  ----------
    1. corr: pair-wise correlation matrix of all trials between the input and reconstructed input

  Returns
  -------
    2. fidelity: reonstruction fidelity
  '''
  fidelity = np.diagonal(corr).mean()
  return fidelity

###############################################################################

def withinVSbetween(categ, corr_df):
    '''
    This funciton returns two matrices that all the trials of a specific catefory as within class
    and another matrix as beteween class

    Parameters
    ----------
    1. categ: a category
    2. corr_df: a pandas dataframe of the pair-wise correlation matrix of all trials.

    Returns
    -------
    1. within: within class trials
    2. between: between class trials
    '''

    within = corr_df.loc[categ][categ]
    within = np.triu(within) # just get the upper half of the matrix since it is symetrical
    within = within[within!=0] # removing all the zero elements from  matrix witch makes a it a vector

    between = corr_df.loc[corr_df.index==categ, corr_df.columns!=categ]
    between = np.triu(between) # just get the upper half of the matrix since it is symetrical
    between = between[between!=0] # removing all the rezo elements from  matrix with makes a it a vector
    return within, between

 ###############################################################################

def cohnsd (a, b):

  '''
  this function calculate the cohensd of two variables
  Mostly to find the cohensd of within and between class correlation

  Parameters
  ----------
  1. a: input vector 1
  2. b: input vector 2

  Returns
  -------
  consd: the cohensd
  '''

  a_mean = a.mean()
  a_sd = a.std()
  b_mean = b.mean()
  b_sd = b.std()
  consd = (a_mean - b_mean)/(np.sqrt((a_sd**2+b_sd**2)/2))
  return consd

###############################################################################
def reset_seed(SEED = 42):
  #tf.reset_default_graph()
  #keras.backend.clear_session()
  os.environ['PYTHONHASHSEED']='0'
  np.random.seed(SEED)
  tf.set_random_seed(SEED)
  rn.seed(SEED)
