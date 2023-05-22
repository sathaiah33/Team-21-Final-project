# import libraries
import pandas as pd
import numpy as np
import sys
import sklearn
import io
import random
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing


import urllib.request
import requests
import threading
import json
import random
     

from tensorflow.keras.layers import Dense, Flatten, Dropout,BatchNormalization ,Activation
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from IPython.display import Image
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from keras.layers import SimpleRNN
import sklearn.metrics as metrics
     

## add the columns' name and read the KDDTrain and KDDTest datasets
col_names = ["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label"]



print(col_names)



#Training set
df = pd.read_csv("C:/Users/satha/OneDrive/Desktop/intrusion_ds/NSL_KDD_Train_1.csv",header=None, names = col_names)
# Testing set
df_test = pd.read_csv("C:/Users/satha/OneDrive/Desktop/intrusion_ds/NSL_KDD_Test_1.csv", header=None, names = col_names)
print('Training: ',df.shape)
print('Testing: ',df_test.shape)


     

df.head()

df_test.head()
print("Target for training set:")
print(df['label'].value_counts())

from matplotlib.pyplot import figure

figure(figsize=(8, 6), dpi=80)

df['label'].value_counts().plot(kind='bar')



# Step 1: Data Preprocessing

print('Training set unique categories in column 2, 3 and 4:')
for col_name in df.columns:
    if df[col_name].dtypes == 'object' :
        unique_cat = len(df[col_name].unique())
        print("{col_name} : {unique_cat}".format(col_name=col_name, unique_cat=unique_cat))


print('Distribution of categories in service:')
print(df['service'].value_counts().sort_values(ascending=False).head())

print('Distribution of categories in flag:')
print(df['flag'].value_counts().sort_values(ascending=False).head())


print('Distribution of categories in label:')
print(df['label'].value_counts().sort_values(ascending=False).head())


## Similarly for the test dataset
# Test set
print('Test set unique categories in column 2, 3 and 4:')
for col_name in df_test.columns:
    if df_test[col_name].dtypes == 'object' :
        unique_cat = len(df_test[col_name].unique())
        print("{col_name} : {unique_cat}".format(col_name=col_name, unique_cat=unique_cat))


#Label Encoder : Converting the categorical values to numerica values

# protocol type
unique_protocol=sorted(df.protocol_type.unique())
unique_protocol2=[x for x in unique_protocol]
print(unique_protocol2)

# service
unique_service=sorted(df.service.unique())
unique_service2=[x for x in unique_service]
print(unique_service2)


# flag
unique_flag=sorted(df.flag.unique())
unique_flag2=[x for x in unique_flag]
print(unique_flag2)


# put together
dumcols=unique_protocol2 + unique_service2 + unique_flag2



print(len(unique_protocol2))
print(len(unique_service2))
print(len(unique_flag2))



#do it for test set
#service
unique_service_test=sorted(df_test.service.unique())
unique_service2_test=[x for x in unique_service_test]

#protocol
unique_protocol_test=sorted(df_test.protocol_type.unique())
unique_protocol2_test=[x for x in unique_protocol_test]

#flag
unique_flag_test=sorted(df_test.flag.unique())
unique_flag2_test=[x for x in unique_flag_test]

testdumcols=unique_protocol2_test + unique_service2_test + unique_flag2_test

print(len(unique_protocol2_test))
print(len(unique_service2_test))
print(len(unique_flag2_test))

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
categorical_columns=['protocol_type', 'service', 'flag']

df_categorical = df[categorical_columns]
# for Test
testdf_categorical = df_test[categorical_columns]

df_categorical_enc=df_categorical.apply(LabelEncoder().fit_transform)

print(df_categorical.head())
print(testdf_categorical.head())

## after label encoding
print(df_categorical_enc.head())

# test set
testdf_categorical_enc=testdf_categorical.apply(LabelEncoder().fit_transform)
print(testdf_categorical_enc.head())

print('Training: ',df_categorical_enc.shape)
print('Testing:',testdf_categorical_enc.shape)
# one-hot encoding


enc = OneHotEncoder(categories='auto')
df_categorical_values_encenc = enc.fit_transform(df_categorical_enc)
df_onehot_data = pd.DataFrame(df_categorical_values_encenc.toarray(),columns=dumcols)
print(df_onehot_data.head())

print("STAGE_2")

# test set
enc = OneHotEncoder(categories='auto')
testdf_categorical_values_encenc = enc.fit_transform(testdf_categorical_enc)
testdf_onehot_data = pd.DataFrame(testdf_categorical_values_encenc.toarray(),columns=testdumcols)
testdf_onehot_data.head()


print('Training onehot df:',df_onehot_data.shape)
print('Testing onehot df:',testdf_onehot_data.shape)

# Since the testdf had 6 columns less in the service, we add the missing values 
trainservice=df['service'].tolist()
testservice= df_test['service'].tolist()
difference=list(set(trainservice) - set(testservice))
difference=[x for x in difference]

for col in difference:
    testdf_onehot_data[col] = 0

print(df_onehot_data.shape)    
print(testdf_onehot_data.shape)



# df = df.dropna('columns')# drop columns with NaN

# df = df[[col for col in df if df[col].nunique() > 1]]# keep columns where there are more than 1 unique values
# test df correlation between values

corr = df_categorical_enc.corr()
plt.figure(figsize=(8,6))
sb.heatmap(corr)
plt.show()

corr = testdf_categorical_enc.corr()
plt.figure(figsize=(8,6))
sb.heatmap(corr)
plt.show()

traindf=df.join(df_onehot_data)
traindf.drop('flag', axis=1, inplace=True)
traindf.drop('protocol_type', axis=1, inplace=True)
traindf.drop('service', axis=1, inplace=True)
print(traindf.shape)
traindf.head()


# test data
testdf=df_test.join(testdf_onehot_data)
testdf.drop('flag', axis=1, inplace=True)
testdf.drop('protocol_type', axis=1, inplace=True)
testdf.drop('service', axis=1, inplace=True)
print(testdf.shape)
testdf.head()
print("STEP 1 completed ")

# Step 2: Normalization

log2_duration = []
for i in traindf['duration']:
  if(i==0):
    log2_duration.append(0)
  else:
    log2_duration.append(np.log2(i))

print(log2_duration)

log2_src_bytes = []
for i in traindf['src_bytes']:
  if(i==0):
    log2_src_bytes.append(0)
  else:
    log2_src_bytes.append(np.log2(i))

print(log2_src_bytes)

# similarly for the testdf
log2_duration_test = []
for i in testdf['duration']:
  if(i==0):
    log2_duration_test.append(0)
  else:
    log2_duration_test.append(np.log2(i))

print(log2_duration_test)



log2_dst_bytes_test = []
for i in testdf['dst_bytes']:
  if(i==0):
    log2_dst_bytes_test.append(0)
  else:
    log2_dst_bytes_test.append(np.log2(i))

print(log2_dst_bytes_test)

log2__src_bytes_test = []
for i in testdf['src_bytes']:
  if(i==0):
    log2__src_bytes_test.append(0)
  else:
    log2__src_bytes_test.append(np.log2(i))

print(log2__src_bytes_test)

# step1: apply the logarithmic scaling method for scaling to obtain the ranges of `duration[0,4.77]', `src_bytes[0,9.11]' and `dst_bytes[0,9.11]

testdf['log2_duration'] = log2_duration_test
testdf['log2_src_bytes'] = log2__src_bytes_test
testdf['log2_dst_bytes'] = log2_dst_bytes_test

testdf=testdf.drop(['duration','src_bytes','dst_bytes'], axis=1)
print(testdf.shape)
print(testdf)

traindf.head()
testdf.head()

#Min-max normalization



labeldf=traindf['label']
labeldf_test=testdf['label']

# change the label column
newlabeldf=labeldf.replace({ 'normal' : 0, 'neptune' : 1 ,'back': 1, 'land': 1, 'pod': 1, 'smurf': 1, 'teardrop': 1,'mailbomb': 1, 'apache2': 1, 'processtable': 1, 'udpstorm': 1, 'worm': 1,
                           'ipsweep' : 2,'nmap' : 2,'portsweep' : 2,'satan' : 2,'mscan' : 2,'saint' : 2
                           ,'ftp_write': 3,'guess_passwd': 3,'imap': 3,'multihop': 3,'phf': 3,'spy': 3,'warezclient': 3,'warezmaster': 3,'sendmail': 3,'named': 3,'snmpgetattack': 3,'snmpguess': 3,'xlock': 3,'xsnoop': 3,'httptunnel': 3,
                           'buffer_overflow': 4,'loadmodule': 4,'perl': 4,'rootkit': 4,'ps': 4,'sqlattack': 4,'xterm': 4})
newlabeldf_test=labeldf_test.replace({ 'normal' : 0, 'neptune' : 1 ,'back': 1, 'land': 1, 'pod': 1, 'smurf': 1, 'teardrop': 1,'mailbomb': 1, 'apache2': 1, 'processtable': 1, 'udpstorm': 1, 'worm': 1,
                           'ipsweep' : 2,'nmap' : 2,'portsweep' : 2,'satan' : 2,'mscan' : 2,'saint' : 2
                           ,'ftp_write': 3,'guess_passwd': 3,'imap': 3,'multihop': 3,'phf': 3,'spy': 3,'warezclient': 3,'warezmaster': 3,'sendmail': 3,'named': 3,'snmpgetattack': 3,'snmpguess': 3,'xlock': 3,'xsnoop': 3,'httptunnel': 3,
                            'buffer_overflow': 4,'loadmodule': 4,'perl': 4,'rootkit': 4,'ps': 4,'sqlattack': 4,'xterm': 4})

print('New Label' , newlabeldf)

print(newlabeldf_test)
#put the new label column back
traindf['label'] = newlabeldf
testdf['label'] = newlabeldf_test


traindf.head()


testdf.head()


newlabeldf.head()


print('Distribution of categories in service:')
print(newlabeldf.value_counts().sort_values(ascending=False).head())


# Further steps:

# making all normal to 0 and the rest anomalous as 1

#traindf
target = []
for i in traindf['label']:
  if(i==0):
    target.append(0)
  else:
    target.append(1)

print(target)
traindf['target'] = target
traindf = traindf.drop(['label'],axis=1)
traindf.head()


sb.countplot(x="target", data=traindf)
plt.title('Class Distributions in KDDTrain+ \n 0: Normal || 1: Attack', fontsize=14)
plt.show()


#testdf
target_test = []
for i in testdf['label']:
  if(i==0):
    target_test.append(0)
  else:
    target_test.append(1)

print(target_test)
testdf['target'] = target_test
testdf = testdf.drop(['label'],axis=1)
testdf.head()

sb.countplot(x="target", data=testdf)
plt.title('Class Distributions in KDDTest+ \n 0: Normal || 1: Attack', fontsize=14)
plt.show()

# Splitting into x and y

X_train = traindf.iloc[:,0:122]
Y_train = traindf['target']
print(X_train.shape)
print(Y_train.shape)



print(X_train)



print(Y_train)

# scaling
scale = MinMaxScaler()
scale = preprocessing.StandardScaler().fit(X_train)
X_train_scaled=scale.transform(X_train)
print(X_train_scaled)


#Testing Df


X_test = testdf.iloc[:,0:122]
Y_test = testdf['target']
print(X_test.shape)
print(Y_test.shape)


print(X_test)


print(Y_test)

# scaling
scale = MinMaxScaler()
scale = preprocessing.StandardScaler().fit(X_test)
X_test_scaled=scale.transform(X_test)
print(X_test_scaled)


# Ensemble learning classification models: Random forest, Multi Layer Perceptron, SVM, Naive Bayes and Decision Tree


from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

mlp = MLPClassifier(solver='adam', learning_rate_init = 0.005, learning_rate = 'adaptive', activation="relu", max_iter=2000, random_state=42)
dec = DecisionTreeClassifier(criterion="entropy", max_depth=3)
ran = RandomForestClassifier(n_estimators=50)
svm = SVC(random_state=1)
naive = GaussianNB()

models = {"J48" : dec,
          "NB" : naive,
          "RF" : ran,
          "MLP" : mlp,
          "SVM" : svm
          }
scores= { }



###Test the models
##for key, value in models.items():    
##    model = value
##    model.fit(X_train_scaled,Y_train)
##    scores[key] = model.score(X_test_scaled, Y_test)
##    
##
### results
##scores_frame = pd.DataFrame(scores, index=["Accuracy Score"]).T
##scores_frame.sort_values(by=["Accuracy Score"], axis=0 , inplace=True)
##scores_frame.head()
##
##
##
##plt.figure(figsize=(8,5))
##sb.barplot(x=scores_frame.index,y=scores_frame["Accuracy Score"])
##plt.ylim(0, 1)
##plt.ylabel("Accuracy%")
##plt.xlabel("Binary Classification")
##
##
##Text(0.5, 0, 'Binary Classification')
##

# Building the RNN model
def thingspeak_post(value):
    threading.Timer(15,thingspeak_post).start()    
    URl='https://api.thingspeak.com/update?api_key='
    KEY='DQPEJ3065A9EPRH5'
    HEADER='&field1={}&field2={}&field3={}'.format(str(value),str(value),str(value))
    NEW_URL = URl+KEY+HEADER
    print(NEW_URL)
    data=urllib.request.urlopen(NEW_URL)
    print(data)

from tensorflow import keras
import numpy as np
import datetime
import time

sample = X_train_scaled.shape[0]
features = X_train_scaled.shape[1]
#Train: convert 2D to 3D for input RNN
x_train = np.reshape(X_train_scaled,(sample,features,1)) #shape  = (909, 18, 1)
#Test: convert 2D to 3D for input RNN

x_test = testdf.values
x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1]))


#print(x_train.shape)


#print(Y_train.shape)



import tensorflow as tf
METRICS = [
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall')
]



import tensorflow.keras.backend as K
def recall(y_true, y_pred):
    y_true = K.ones_like(y_true) 
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    all_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    
    recall = true_positives / (all_positives + K.epsilon())
    return recall

def precision(y_true, y_pred):
    y_true = K.ones_like(y_true) 
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_score(y_true, y_pred):
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2*((prec*rec)/(prec+rec+K.epsilon()))

#RNN model 1: hidden node = 60, learning rate = 0.01


from keras import optimizers 
model = Sequential()
model.add(SimpleRNN(60,input_shape=(features,x_train.shape[2]), activation='sigmoid'))
model.add(Dense(1))
opt = keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy', 'mae', recall,precision,f1_score])
model.summary()

#training model 1

history = model.fit(x_train, Y_train, validation_split=0.33, epochs=2, batch_size= 32)  


# RNN model 2: hidden node = 80, learning rate = 0.01

from keras import optimizers 
model = Sequential()
model.add(SimpleRNN(80,input_shape=(features,x_train.shape[2]), activation='sigmoid'))
model.add(Dense(1))
opt = keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy', 'mae', recall,precision,f1_score])

#training model 2


history2 = model.fit(x_train, Y_train, validation_split=0.33, epochs=5, batch_size= 32) 

model.summary()
mcp = ModelCheckpoint('RNN_binary2.h5')




# RNN model 3: hidden node = 120, learning rate = 0.01

from keras import optimizers 
model = Sequential()
model.add(SimpleRNN(120,input_shape=(features,x_train.shape[2]), activation='sigmoid'))
model.add(Dense(1))
opt = keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy', 'mae', recall,precision,f1_score])
model.summary()

# training model 3

history3 = model.fit(x_train, Y_train, validation_split=0.33, epochs=2, batch_size= 32) 



fig, (ax1) = plt.subplots(figsize= (8,5))
print('loss value :',history.history['loss'])
print('loss value :',history2.history['loss'])
print('loss value :',history3.history['loss'])
plt.plot(history.history['loss'])
plt.plot(history2.history['loss'])
plt.plot(history3.history['loss'])
ax1.set_title('History of Loss - Training')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.legend(['model 1', 'model 2','model 3'])



fig, (ax1) = plt.subplots(figsize= (8,5))
plt.plot(history.history['recall'])
plt.plot(history2.history['recall'])
plt.plot(history3.history['recall'])
ax1.set_title('History of Recall - Training')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Recall')
ax1.legend(['model 1', 'model 2','model 3'])



fig, (ax1) = plt.subplots(figsize= (8,5))
plt.plot(history.history['mae'])
plt.plot(history2.history['mae'])
plt.plot(history3.history['mae'])
ax1.set_title('History of MAE - Training')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('MAE')
ax1.legend(['model 1', 'model 2','model 3'])



# RNN model 4: hidden node = 60, learning rate = 0.1

from keras import optimizers 
model = Sequential()
model.add(SimpleRNN(60,input_shape=(features,x_train.shape[2]), activation='sigmoid'))
model.add(Dense(1))
opt = keras.optimizers.Adam(learning_rate=0.1)
model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy', 'mae', recall,precision,f1_score])
model.summary()


mcp = ModelCheckpoint('RNN_binary4.h5')


history = model.fit(x_train, Y_train, validation_split=0.33, epochs=2, batch_size= 32) 

# RNN model 5: hidden node = 80, learning rate = 0.1



from keras import optimizers 
model = Sequential()
model.add(SimpleRNN(80,input_shape=(features,x_train.shape[2]), activation='sigmoid'))
model.add(Dense(1))
opt = keras.optimizers.Adam(learning_rate=0.1)
model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy', 'mae', recall,precision,f1_score])
model.summary()



mcp = ModelCheckpoint('RNN_binary5.h5')


history2 = model.fit(x_train, Y_train, validation_split=0.33, epochs=2, batch_size= 32) 


# RNN model 6: hidden node = 120, learning rate = 0.1

from keras import optimizers 
model = Sequential()
model.add(SimpleRNN(120,input_shape=(features,x_train.shape[2]), activation='sigmoid'))
model.add(Dense(1))
opt = keras.optimizers.Adam(learning_rate=0.1)
model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy', 'mae', recall,precision,f1_score])
model.summary()

mcp = ModelCheckpoint('RNN_binary5.h5')



history3 = model.fit(x_train, Y_train, validation_split=0.33, epochs=2, batch_size= 32) 



fig, (ax1) = plt.subplots(figsize= (8,5))
print('accuracy value :',history.history['accuracy'])
print('accuracy value :',history2.history['accuracy'])
print('accuracy value :',history3.history['accuracy'])
plt.plot(history.history['accuracy'])
plt.plot(history2.history['accuracy'])
plt.plot(history3.history['accuracy'])
ax1.set_title('History of Accuracy')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Accuracy')
ax1.legend(['model 1', 'model 2','model 3'])


fig, (ax1) = plt.subplots(figsize= (8,5))
plt.plot(history.history['loss'])
plt.plot(history2.history['loss'])
plt.plot(history3.history['loss'])
ax1.set_title('History of Loss')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.legend(['model 1', 'model 2','model 3'])



fig, (ax1) = plt.subplots(figsize= (8,5))
plt.plot(history.history['recall'])
plt.plot(history2.history['recall'])
plt.plot(history3.history['recall'])
ax1.set_title('History of Recall')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Recall')
ax1.legend(['model 1', 'model 2','model 3'])



fig, (ax1) = plt.subplots(figsize= (8,5))
plt.plot(history.history['mae'])
plt.plot(history2.history['mae'])
plt.plot(history3.history['mae'])
#print("history of MAE",history.history['mae'])
#print(type(history.history['mae']))
val=history.history['mae']
value=val[0]
thingspeak_post(value)
ax1.set_title('History of MAE')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('MAE')
ax1.legend(['model 1', 'model 2','model 3'])


#Using tanh and sigmoid as activation functions in LSTM



Model = keras.Sequential([

        keras.layers.LSTM(80,input_shape=(features,x_train.shape[2]),activation='tanh',recurrent_activation='sigmoid'),
        keras.layers.Dense(1,activation="tanh")
    ])

Model.compile(optimizer='adam',loss='mse', metrics=['accuracy'])

#Training the model
hist = Model.fit(x_train, Y_train, epochs=25, batch_size= 128) 
Model.summary()

#Using tanh and sigmoid as activation functions
model = Sequential()
model.add(SimpleRNN(80,input_shape=(features,x_train.shape[2]), activation='tanh'))
model.add(Dense(60, input_dim=122, activation='relu'))
model.add(Dense(1))
opt = keras.optimizers.Adam(learning_rate=0.1)
model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy', 'mae', recall,precision,f1_score])
model.summary()

mcp = ModelCheckpoint('LSTM_model.h5')


history_lstm = model.fit(x_train, Y_train, validation_split=0.33, epochs=2, batch_size= 64 ,callbacks=[mcp]) 


#Using tanh and sigmoid as activation functions
from keras.layers import Dropout
from keras.layers import LSTM

model = Sequential()
model.add(LSTM(80,input_shape=(features,x_train.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(80))
model.add(Dropout(0.2))
model.add(Dense(1, activation='softmax'))
opt = keras.optimizers.Adam(learning_rate=0.1)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy', 'mae', recall,precision,f1_score])
model.summary()


mcp2 = ModelCheckpoint('new_model.h5')


history_lstm = model.fit(x_train, Y_train, validation_split=0.33, epochs=2, batch_size= 64 ,callbacks=[mcp])
print('LSTM Testing value',history_lstm.history['accuracy'])

x_test = testdf.drop(columns='target').values
print(x_test.shape)

x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))



y_test = testdf.target
print(y_test)


# Final evaluation of the model
#scores = Model.evaluate(x_test, Y_test, verbose=0)
#print("/n")
#print("Accuracy: %.2f%%" % (scores[1]*100))


import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
models = ['Naive Bayes', 'Random Forest', 'SVM', 'MLP', 'Decision Tree','RNN 1','RNN 2','RNN 3','LSTM']
accu = [0.527,0.722,0.78,0.805,0.8306,0.9593,0.9710,0.9475,0.4021]
ax.bar(models,accu)
plt.show()






