import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
device_nb = 4
os.environ["CUDA_VISIBLE_DEVICES"]=str(device_nb)

import pandas as pd
import datetime
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import tarfile 
import numpy as np
import _pickle as cPickle
import os
import wfdb
from datetime import datetime
from datetime import timedelta
import numpy
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.constraints import maxnorm


df_o = pd.read_csv("df_MASTER_DATA.csv")

import random
random.seed(9)
random.sample(range(1, 61089), 10000)

#sub-select 20% for testing
test_idx = random.sample(range(df_o.shape[0]), int(df_o.shape[0] * 0.2))
df_test = df_o.loc[test_idx]
train_idx = list(set([ x for x in range(df_o.shape[0])]) - set(test_idx))
df_train = df_o.loc[train_idx]





#mean impute
#delete HADM_ID, SUBJECT_ID, ETHNICITY, MARITAL_STATUS, INSURANCE, RELIGION, INTIME, OUTTIME
#dummify
def preprocessing(df_labeled):


    #pre-processing, missingness > 0.5 

    df_labeled_filtered = df_labeled

    for column in df_labeled_filtered.columns:
        missing_r = df_labeled[column].isnull().sum()/df_labeled.shape[0]
        if missing_r > 0.5:

            #keep arterial BP
            if not (column.startswith('Arterial_BP') or column.startswith('Mean_Arterial')):


                df_labeled_filtered = df_labeled_filtered.drop(columns=[column])


    #mean impute
    df_labeled_filtered = df_labeled_filtered.fillna(df_labeled_filtered.mean())

    #delete HADM_ID, SUBJECT_ID, ETHNICITY, MARITAL_STATUS, INSURANCE, RELIGION, INTIME, OUTTIME
    colmuns_todrop = ['Unnamed: 0','FIRST_CAREUNIT','ETHNICITY','MARITAL_STATUS','HADM_ID', 'ICUSTAY_ID',
                      'INSURANCE','RELIGION','INTIME', 'OUTTIME', 'SUBJECT_ID','LANGUAGE','Time_To_readmission', 
                      'IsReadmitted_24hrs', 'IsReadmitted_48hrs',  'IsReadmitted_7days', 
                      'IsReadmitted_30days', 'IsReadmitted_Bounceback']
    #don't delete ==> 'IsReadmitted_72hrs',
    
    for column_d in colmuns_todrop:
        
        try:
            df_labeled_filtered = df_labeled_filtered.drop(columns=column_d)
        except:
            pass

    #dummify
    df_labeled_filtered = pd.get_dummies(df_labeled_filtered)
    
    
    return df_labeled_filtered




# split into input (X) and output (Y) variables
df_filtered = preprocessing(df_train)
Y = df_filtered['IsReadmitted_72hrs']
X = df_filtered.drop(columns=['IsReadmitted_72hrs'])



############## Use scikit-learn to grid search the number of neurons #############
def create_model(neurons=1):
    # create model
    model = Sequential()
    model.add(Dense(neurons, input_dim=2291, kernel_initializer='uniform', activation='relu', kernel_constraint=maxnorm(4)))
    model.add(Dropout(0.2))
    model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# create model
model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=4096, verbose=0)
# define the grid search parameters
neurons = [50, 100, 200, 300, 400, 500, 1000]
param_grid = dict(neurons=neurons)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)
grid_result = grid.fit(X.values, Y.values)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
for key in grid_result.best_params_:
    best_Num_neurons = grid_result.best_params_[key]

        
        
############# Use scikit-learn to grid search the batch size and epochs   #############
def create_model(optimizer='adam'):
    # create model
    model = Sequential()
    model.add(Dense(best_Num_neurons, input_dim=2291, activation='relu', kernel_constraint=maxnorm(4)))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# create model
model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=4096, verbose=0)
# define the grid search parameters
optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
param_grid = dict(optimizer=optimizer)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)
grid_result = grid.fit(X.values, Y.values)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
for key in grid_result.best_params_:
    best_optimizer = grid_result.best_params_[key]

        

############## Use scikit-learn to grid search the weight initialization  #############
def create_model(init_mode='uniform'):
    # create model
    model = Sequential()
    model.add(Dense(best_Num_neurons, input_dim=2291, kernel_initializer=init_mode, activation='relu'))
    model.add(Dense(1, kernel_initializer=init_mode, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=best_optimizer, metrics=['accuracy'])
    return model
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# create model
model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=4096, verbose=0)
# define the grid search parameters
init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
param_grid = dict(init_mode=init_mode)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)
grid_result = grid.fit(X.values, Y.values)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
        
        
        
        
        
from numba import cuda
cuda.select_device(0)
cuda.close()