
# coding: utf-8

# In[8]:


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
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras import backend as K


# In[2]:


#df_CE = pd.read_csv('all_chart_events.csv')
#df_CE.columns = ['icustay_id', 'itemid', 'valuenum', 'charttime']


# In[3]:


#df_itemID = pd.read_csv('unique_chartevent_items.csv')
#df_itemID.columns = ['itemid', 'measurement']


# In[9]:


df_trainID = pd.read_csv('df_train_subjects.csv')
df_testID = pd.read_csv('df_test_subjects.csv')
df_valID = pd.read_csv('df_val_subjects.csv')


# In[10]:


df_trainID


# # bounce back has 6% of the data

# In[5]:


import os

path, dirs, files = next(os.walk("./each_icustay_csv"))
file_count = len(files)
file_count


# In[6]:


import os

path, dirs, files = next(os.walk("./feature_folder"))
file_count = len(files)
file_count


# # feature space has non-sparse ratio mean: 23%, max 51%, min 0%

# # get features prepared for LSTM (training set)

# In[22]:


#nan is from mean, std, that there's no mean, std (just one value in a window)
#0 is from max, min, count


x_train = []
y_train = []


for index, row in df_trainID.iterrows():
    
    icustay_id = row['ICUSTAY_ID']
    
    
    #see if a icu stay has a feature file
    try:
        outfile = './feature_folder/feature_space_%s.file.npy'%(str(icustay_id))
        all_feature_vectors = np.load(outfile)

        #change NaN to 0
        all_feature_vectors = np.nan_to_num(all_feature_vectors)
        
        
        #count non-zero
        #non_0 = np.count_nonzero(np.nan_to_num(all_feature_vectors))
        #non_0_ratio = non_0/(all_feature_vectors.shape[0]*all_feature_vectors.shape[1])

        x_train.append(all_feature_vectors)

        y_train.append(row['IsReadmitted_Bounceback'])
        
    except FileNotFoundError:
        continue
        
x_train = np.asarray(x_train)
y_train = np.asarray(y_train)


# In[24]:


x_train.shape


# # get features prepared for LSTM (validation set)

# In[8]:


#nan is from mean, std, that there's no mean, std (just one value in a window)
#0 is from max, min, count


x_valid = []
y_valid = []


for index, row in df_valID.iterrows():
    
    icustay_id = row['ICUSTAY_ID']
    
    
    #see if a icu stay has a feature file
    try:
        outfile = './feature_folder/feature_space_%s.file.npy'%(str(icustay_id))
        all_feature_vectors = np.load(outfile)

        #change NaN to 0
        all_feature_vectors = np.nan_to_num(all_feature_vectors)
        
        
        #count non-zero
        #non_0 = np.count_nonzero(np.nan_to_num(all_feature_vectors))
        #non_0_ratio = non_0/(all_feature_vectors.shape[0]*all_feature_vectors.shape[1])

        x_valid.append(all_feature_vectors)

        y_valid.append(row['IsReadmitted_Bounceback'])
        
    except FileNotFoundError:
        continue
        
x_valid = np.asarray(x_valid)
y_valid = np.asarray(y_valid)


# # get features prepared for LSTM (testing set)

# In[9]:


#nan is from mean, std, that there's no mean, std (just one value in a window)
#0 is from max, min, count


x_test = []
y_test = []


for index, row in df_testID.iterrows():
    
    icustay_id = row['ICUSTAY_ID']
    
    
    #see if a icu stay has a feature file
    try:
        outfile = './feature_folder/feature_space_%s.file.npy'%(str(icustay_id))
        all_feature_vectors = np.load(outfile)

        #change NaN to 0
        all_feature_vectors = np.nan_to_num(all_feature_vectors)
        
        
        #count non-zero
        #non_0 = np.count_nonzero(np.nan_to_num(all_feature_vectors))
        #non_0_ratio = non_0/(all_feature_vectors.shape[0]*all_feature_vectors.shape[1])

        x_test.append(all_feature_vectors)

        y_test.append(row['IsReadmitted_Bounceback'])
        
    except FileNotFoundError:
        continue
        
x_test = np.asarray(x_test)
y_test = np.asarray(y_test)


# # test: not output along time dimension, but along neurons dimesnion

# In[17]:


import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
device_nb = '7'
os.environ["CUDA_VISIBLE_DEVICES"]=str(device_nb)



from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM, Dense
from numpy import array
# define model
inputs1 = Input(shape=(3, 1))
#lstm1 = LSTM(2, return_state=True)(inputs1)
lstm1 = LSTM(10, name='lstm')(inputs1)
output = Dense(1)(lstm1)
model = Model(inputs=inputs1, outputs=output)
# define input data
data = array([[0.8, 0.9, 0.7],[0.1, 0.2, 0.3]]).reshape((2,3,1))
# make and show prediction
prediction=model.predict(data)
print(prediction)





# In[18]:


df = pd.DataFrame(data.reshape(2,-1))
df.values


# In[6]:


model.summary()


# In[20]:


layer_name = 'lstm'
intermediate_layer_model_output = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
intermediate_output = intermediate_layer_model_output.predict(df.values.reshape(2,3,1))


# In[21]:


intermediate_output


# In[25]:


pd.DataFrame(intermediate_output)


# In[2]:


from numba import cuda
cuda.select_device(0)
cuda.close()


# In[26]:


cohortvector_train = pd.read_csv('cohortvector_train.csv')


# In[27]:


cohortvector_train


# # LSTM

# In[10]:


import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
device_nb = 2
os.environ["CUDA_VISIBLE_DEVICES"]=str(device_nb)



model = Sequential()
#model.add(LSTM(32, return_sequences=True, input_shape=(48, 30)))
model.add(LSTM(2048, input_shape=(48, 30)))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')
history = model.fit(x_train, y_train, epochs=5, batch_size=2014, verbose=2, validation_data=(x_valid, y_valid))


y_pred_proba = model.predict_proba(x_test)
y_pred = model.predict(x_test)


# In[23]:


from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, y_pred_proba)


# In[12]:


import matplotlib.pyplot as plt
# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# In[22]:


y_pred_proba


# In[21]:


y_pred


# In[6]:


from numba import cuda
cuda.select_device(0)
cuda.close()


# In[15]:


itemid1 = 211
itemid2 = 224697

icustay_id = 294638


intime = pd.to_datetime(df_trainID[df_trainID['ICUSTAY_ID'] == icustay_id]['INTIME'].values[0])

outtime = pd.to_datetime(df_trainID[df_trainID['ICUSTAY_ID'] == icustay_id]['OUTTIME'].values[0])



with open("./each_icustay_csv/" + str(icustay_id) + '.csv') as f0:
    Adf_ICUstay_CE = pd.read_csv(f0)


onetime = pd.to_datetime(Adf_ICUstay_CE[(Adf_ICUstay_CE['itemid'] == itemid1) | (Adf_ICUstay_CE['itemid'] == itemid2)]['charttime']).values[0]
df_onechannel = Adf_ICUstay_CE[(Adf_ICUstay_CE['itemid'] == itemid1) | (Adf_ICUstay_CE['itemid'] == itemid2)]

#Adf_ICUstay_CE[(Adf_ICUstay_CE['itemid'] == itemid1) | (Adf_ICUstay_CE['itemid'] == itemid2)]




# In[16]:


pd.to_datetime(df_onechannel['charttime'])
df_onechannel['charttime'] = pd.to_datetime(df_onechannel['charttime'])
df_onechannel['charttime'] .values[0]


# In[17]:


#given a icu stay's df, and what itemid, to get features in 48h

def OnefeaturesIN48h(Adf_ICUstay_CE, itemid1, itemid2, intime, outtime):
    
    feature_vector = []
    
    df_onechannel = Adf_ICUstay_CE[(Adf_ICUstay_CE['itemid'] == itemid1) | (Adf_ICUstay_CE['itemid'] == itemid2)]
    
    #change 'charttime' str to datetime object
    df_onechannel['charttime'] = pd.to_datetime(df_onechannel['charttime'])
    
    #TO DO:  first impute all of them 
    
    end_time = intime
    
    #first 24 hours
    for pasthours in range(0, 24):
        
        
        start_time = end_time
        end_time = start_time + np.timedelta64(1, 'h')
        
        df_InATimewindow = df_onechannel[ (start_time < df_onechannel['charttime']) & (df_onechannel['charttime'] < end_time) ]
        
        
        #add mean feature
        try:
            mean_feature = np.mean(df_InATimewindow['valuenum'].values)
        except ValueError:
            mean_feature = 0
        
        
        #add std feature
        try:
            std_feature = np.std(df_InATimewindow['valuenum'].values)
        except ValueError:
            std_feature = 0
        
        #add max feature
        try:
            max_feature = np.amax(df_InATimewindow['valuenum'].values)
        except ValueError:
            max_feature = 0
        
        #add min feature
        try:
            min_feature = np.amin(df_InATimewindow['valuenum'].values)
        except ValueError:
            min_feature = 0
        
        #add count feature
        try:
            count_feature = df_InATimewindow['valuenum'].values.shape[0]
        except ValueError:
            count_feature = 0
        
        features_1h_window = [mean_feature, std_feature, max_feature, min_feature, count_feature]
        
        feature_vector.append(features_1h_window)
        
     
    
    end_time = outtime - np.timedelta64(24, 'h')
    
    #last 24 hours
    for pasthours in range(0, 24):
        
        start_time = end_time
        end_time = start_time + np.timedelta64(1, 'h')
        
        df_InATimewindow = df_onechannel[ (start_time < df_onechannel['charttime']) & (df_onechannel['charttime'] < end_time) ]
        
        
        #add mean feature
        try:
            mean_feature = np.mean(df_InATimewindow['valuenum'].values)
        except ValueError:
            mean_feature = 0
        
        
        #add std feature
        try:
            std_feature = np.std(df_InATimewindow['valuenum'].values)
        except ValueError:
            std_feature = 0
        
        #add max feature
        try:
            max_feature = np.amax(df_InATimewindow['valuenum'].values)
        except ValueError:
            max_feature = 0
        
        #add min feature
        try:
            min_feature = np.amin(df_InATimewindow['valuenum'].values)
        except ValueError:
            min_feature = 0
        
        #add count feature
        try:
            count_feature = df_InATimewindow['valuenum'].values.shape[0]
        except ValueError:
            count_feature = 0
        
        
        features_1h_window = [mean_feature, std_feature, max_feature, min_feature, count_feature]
        
        feature_vector.append(features_1h_window)
        
        
        
    return feature_vector


# In[18]:


def AICUstay_goodchannels(df_CE, df_itemID, icustay_id):
    goodchannels = []
    for item_count in df_CE[df_CE['icustay_id'] == icustay_id]['itemid'].value_counts().iteritems():
        itemid = item_count[0]
        item_name = df_itemID[df_itemID['itemid']==itemid]['measurement'].values[0]
        goodratio =  missing_ratio(df_CE, icustay_id, itemid)
        if goodratio >= 0.5:
            goodchannels.append(item_name)
            #print(item_name, goodratio)
    return goodchannels


# In[19]:


def missing_ratio(df_CE, icustay_id, itemid):
    return (df_CE[(df_CE['icustay_id'] == icustay_id) & (df_CE['itemid'] == itemid)]['valuenum'].shape[0]-df_CE[(df_CE['icustay_id'] == icustay_id) & (df_CE['itemid'] == itemid)]['valuenum'].isnull().sum()) / df_CE[(df_CE['icustay_id'] == icustay_id) & (df_CE['itemid'] == itemid)]['valuenum'].shape[0]

