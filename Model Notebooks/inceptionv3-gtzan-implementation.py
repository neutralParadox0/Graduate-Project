#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load


import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense, BatchNormalization, Flatten, Dropout
from tensorflow.keras.applications import inception_v3
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import librosa.display
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras.metrics import AUC
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import os
from keras.callbacks import ReduceLROnPlateau

# if os.path.exists('./prep/'):
#     shutil.rmtree('./prep/')

label_csv = './input/gtzan-dataset-music-genre-classification/Data/features_30_sec.csv'
audio_fpath = "./input/gtzan-dataset-music-genre-classification/Data/genres_original/"
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


def create_dir(dirname):
    if os.path.exists(dirname):
        pass
    else:
        os.makedirs(dirname)


# In[ ]:


df = pd.read_csv(label_csv)

print(df.head())
df.shape


# In[ ]:


trainframe = []
for index, row in tqdm(df.iterrows()):  
    file_name = os.path.join(os.path.abspath(audio_fpath),str(row['label'] + '/'), str(row['filename']))
    dirname='./prep/' + row['label']
    if not os.path.exists(dirname):
        create_dir(dirname)
    try:
        data, sr = librosa.load(file_name)
    except:
        print(str(row['filename'])+ ' is not a valid file')
    else: 
        
        file_name = os.path.join(os.path.abspath(audio_fpath),str(row['label'] + '/'), str(row['filename']))
        mfcc = []
        length = row['length']/10
        start = 0
        end = length
        for i in range(10):
            out_file = dirname+ '/' + row['filename'].split('.wav')[0] + '.' + str(i) + '.png'
            file_name=(out_file.split('/')[-1])
            label=(row['label'])
            trainframe.append([file_name, label])
            if os.path.exists(out_file):
                t = end
                start = end
                end = t + length
                continue
            frag = data[int(start):int(end)]
            mfcc = librosa.feature.mfcc(y = frag, sr = sr, n_mfcc=20)
            mfcc = np.subtract(mfcc, np.mean(mfcc))
            mfcc_gram = librosa.amplitude_to_db(mfcc, ref=np.min)
            librosa.display.specshow(mfcc_gram, sr=sr)
            plt.colorbar(format='%+2.0f dB')
            plt.savefig(out_file)
            plt.clf()
            t = end
            start = end
            end = t + length


# In[ ]:


np.array(trainframe).shape


# In[ ]:


feature_df = pd.DataFrame(trainframe, columns=['filename','class'])

feature_df.head()


# In[ ]:


x = np.array(feature_df['filename'].tolist())

x.shape


# In[ ]:


target = np.array(feature_df['class'].tolist())


# In[ ]:


y_new = pd.get_dummies(target)
print(y_new.shape)
y_new.head()


# In[ ]:


data = feature_df
data['filename'] = feature_df['filename'].apply(lambda x:x.replace('.wav', '.png'))
data['temp'] = feature_df['class'] +'/' + feature_df['filename'] 
data['path'] = data['temp']
# .apply(lambda x:'./prep/' + x)
data = data[['filename','path', 'class']]
# data['class'] = data['target']
data.head()


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(
         data, y_new, test_size=0.2, random_state=15)

print(str(X_train.shape)+ ', ' + str(X_test.shape))
datagen = ImageDataGenerator()
it = datagen.flow_from_dataframe(
    data,
    x_col="path",
    y_col="class",batch_size=1,
    directory='./prep/')

inp = Input(shape = next(it)[0][0].shape)
print(inp)
model = inception_v3.InceptionV3(
    include_top=False,
    weights="imagenet",
    classes=10,
    input_tensor = inp
)
# y_train
X_test


# In[ ]:


for layer in model.layers:
    layer.trainable=False


# In[ ]:


new_model = Sequential()
new_model.add(model)
new_model.add(Flatten())
new_model.add(BatchNormalization())
new_model.add(Dense(128, activation='relu'))
new_model.add(Dropout(0.5))
new_model.add(BatchNormalization())
new_model.add(Dense(64, activation='relu'))
new_model.add(Dropout(0.5))
new_model.add(Dense(10, activation='softmax'))

new_model.compile(loss = 'categorical_crossentropy',optimizer='adam',metrics=['accuracy', AUC()])
print(new_model.summary())


# In[ ]:


es = EarlyStopping(
    monitor='val_accuracy', 
    restore_best_weights=True, 
    patience=20, 
    min_delta = 0.001
)
lrr = ReduceLROnPlateau(monitor='val_accuracy', 
                        patience=3, 
                        verbose=1, 
                        factor=0.5,
                        min_lr=0.00001)
history = new_model.fit(datagen.flow_from_dataframe(
    X_train,
    x_col="path",
    y_col="class",
    batch_size=512, 
    directory = './prep'
),
              validation_data=datagen.flow_from_dataframe(
            X_test,
            x_col="path",
            y_col="class",
                  
                  batch_size=100, 
                  directory = './prep'),
              epochs=500,
              callbacks=[es])
new_model.save('InceptionV3_GTZAN_trained_model')

# In[ ]:


fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(12,6))
ax2.plot(history.history['loss'], color='orange', label='Loss')
ax2.plot(history.history['val_loss'], color='blue', label='val_loss')
ax2.legend(loc='upper right')
ax1.plot(history.history['accuracy'], label='Accuracy', color='orange')
ax1.plot(history.history['val_accuracy'], label='val_accuracy', color='blue')
ax1.legend(loc="upper right")
ax1.set_title("Model-Accuracy w.r.t Epochs", loc='center')
plt.xlabel("Epochs")
plt.ylabel("Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Accuracy")
ax3.plot(history.history['auc'], color='orange', label='AUC')
ax3.plot(history.history['val_auc'], color='blue', label='val_auc')
ax3.set_xlabel("Epoch")
ax3.set_ylabel("AUC")
ax3.legend(loc='upper right')
plt.savefig('model_acc_Inception_GTZAN.png')


# In[ ]:


X_test.head()


# In[ ]:


test_dat = ImageDataGenerator()
test_gen = test_dat.flow_from_dataframe(data, 
                                        x_col="path", 
                                        y_col="class", 
                                        batch_size=100, 
                                        class_mode='categorical',
                                        shuffle = False,
                                        directory = './prep')

labels = pd.get_dummies(np.array(test_gen.labels))
y_pred = new_model.predict(test_gen,steps =9990)
for q in labels:
    print(q)


# In[ ]:


y_pred_classes = np.argmax(np.array(y_pred), axis=1)

y_pred_classes


# In[ ]:


y_true = np.argmax(np.array(labels), axis=1)

y_true


# In[ ]:


print(np.mean(y_pred_classes == y_true))
m = AUC()
m.update_state(labels, y_pred)
print(m.result())


# In[ ]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_pred_classes, y_true,)
import itertools
plt.clf()
fig = plt.figure(figsize = (12,12)) 
ax = fig.add_subplot(111)
ax.set_aspect(1)
plt.imshow(cm, cmap = plt.cm.Blues, interpolation='nearest')

plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(len(range(10)))
plt.xticks(tick_marks, range(10), rotation=45)
plt.yticks(tick_marks, range(10))
thresh = cm.max()/2
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i , cm[i,j], horizontalalignment="center", color="white" if cm[i,j]>thresh else "black")

plt.tight_layout()
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.savefig('con_matrix_Inception_GTZAN.png')

