import os

# for loading and visualizing audio files
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
# to play audio
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense, BatchNormalization, Flatten, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.metrics import AUC
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm

# if os.path.exists('./prep/'):
#     shutil.rmtree('./prep/')
label_csv = 'input/environmental-sound-classification-50/esc50.csv'
audio_fpath = "input/environmental-sound-classification-50/audio/audio/44100/"


class AudioAugmentation:
    def add_noise(self, data):
        noise = np.random.randn(len(data))
        data_noise = data + 0.005 * noise
        return data_noise

    def shift(self, data):
        return np.roll(data, 44100)

    def stretch(self, data, rate=1):
        input_length = 220500
        data = librosa.effects.time_stretch(data, rate)
        if len(data) > input_length:
            data = data[:input_length]
        else:
            data = np.pad(data, (0, max(0, input_length - len(data))), "constant")
        return data





def create_dir(dirname):
    if os.path.exists(dirname):
        pass
    else:
        os.makedirs(dirname)


audio_fpath = "input/environmental-sound-classification-50/audio/audio/44100/"
audio_clips = os.listdir(audio_fpath)
print("No. of .wav files in audio folder = ", len(audio_clips))
df = pd.read_csv(label_csv, usecols=['filename', 'target', 'category'])

df.head()

aa = AudioAugmentation()
trainframe = []
for index, row in tqdm(df.iterrows()):
    file_name = os.path.join(os.path.abspath(audio_fpath), str(row['filename']))
    dirname = './prep/'
    if not os.path.exists(dirname):
        create_dir(dirname)
    try:
        y, sr = librosa.load(file_name)
    except:
        print(str(row['filename']) + ' is not a valid file')
    else:

        file_name = os.path.join(os.path.abspath(audio_fpath), str(row['category'] + '/'), str(row['filename']))
        mfcc = []
        for i in range(8):
            out_file = dirname + row['filename'].split('.wav')[0] + '.' + str(i) + '.png'
            file_name = (out_file.split('/')[-1])
            label = (row['category'])
            trainframe.append([file_name, label])
            if i == 1 or i > 3:
                feature = aa.add_noise(y)
            if i % 3 == 2 or i == 7:
                feature = aa.shift(y)
            if i % 3 == 0 or i == 7:
                feature = aa.stretch(y, rate=1.25)
            if os.path.exists(out_file):
                continue
            mfcc = librosa.feature.mfcc(y=feature, sr=sr, n_mfcc=20)
            mfcc = np.subtract(mfcc, np.mean(mfcc))
            mfcc_gram = librosa.amplitude_to_db(mfcc, ref=np.min)
            librosa.display.specshow(mfcc_gram, sr=sr)
            plt.colorbar(format='%+2.0f dB')
            plt.savefig(out_file)
            plt.clf()
feature_df = pd.DataFrame(trainframe, columns=['filename','class'])

feature_df.head()
x = np.array(feature_df['filename'].tolist())

x.shape

target = np.array(feature_df['class'].tolist())

y_new = pd.get_dummies(target)
print(y_new.shape)
y_new.head()

data = feature_df
data['filename'] = feature_df['filename'].apply(lambda x:x.replace('.wav', '.png'))
# data['temp'] = feature_df['class'] +'/' + feature_df['filename']
# data['path'] = data['temp']
data = data[['filename', 'class']]
data.head()

X_train, X_test, y_train, y_test = train_test_split(
    data,
    y_new,
    test_size=0.2,
    random_state=15
)
datagen = ImageDataGenerator()
it = datagen.flow_from_dataframe(
    data,
    x_col="filename",
    y_col='class',
    batch_size=1,
    directory='./prep/')

print(it.class_indices.keys())
inp = Input(shape = next(it)[0][0].shape)
print(inp)
model = ResNet50(
    include_top=False,
    weights="imagenet",
    classes=50,
    input_tensor = inp
)

print(type(x), type(sr))
print(x.shape, sr)

for layer in model.layers:
    layer.trainable=False
new_model = Sequential()
new_model.add(model)
new_model.add(Flatten())
new_model.add(BatchNormalization())
new_model.add(Dense(128, activation='relu'))
new_model.add(Dropout(0.5))
new_model.add(BatchNormalization())
new_model.add(Dense(64, activation='relu'))
new_model.add(Dropout(0.5))
new_model.add(Dense(50, activation='softmax'))
adam = tf.keras.optimizers.Adam(learning_rate=0.002)
new_model.compile(
    loss = 'categorical_crossentropy',
    optimizer=adam,
    metrics=['accuracy', AUC()]
)
es = EarlyStopping(
    monitor='val_accuracy',
    restore_best_weights=True,
    patience=15,
    min_delta = 0.001
)

history = new_model.fit(
    datagen.flow_from_dataframe(
        X_train,
        x_col="filename",
        y_col="class",
        batch_size=256,
        directory = './prep'
    ),
    validation_data=datagen.flow_from_dataframe(
        X_test,
        x_col="filename",
        y_col="class",
        batch_size=256,
        directory = './prep'),
    epochs=500,
    callbacks=[es]
)
new_model.save('ResNet50_ESC50_Trained_model2')
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
plt.savefig('model_acc_ResNet50_ESC50.png')

test_dat = ImageDataGenerator()
test_gen = test_dat.flow_from_dataframe(data,
                                        x_col="filename",
                                        y_col="class",
                                        batch_size=100,
                                        class_mode='categorical',
                                        shuffle = False,
                                        directory = './prep')

labels = pd.get_dummies(np.array(test_gen.labels))
y_pred = new_model.predict(test_gen,steps =16000)
print(labels)

y_pred_classes = np.argmax(y_pred, axis=1)

y_true = np.argmax(np.array(labels), axis=1)
for i in labels:
    print(i)
print(np.mean(y_pred_classes == y_true))
m = AUC()
m.update_state(labels, y_pred)
print(m.result())

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
tick_marks = np.arange(len(range(50)))
plt.xticks(tick_marks, range(50), rotation=45)
plt.yticks(tick_marks, range(50))
thresh = cm.max()/2
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i , cm[i,j], horizontalalignment="center", color="white" if cm[i,j]>thresh else "black")

plt.tight_layout()
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.savefig('con_matrix_ResNet50_ESC50.png')