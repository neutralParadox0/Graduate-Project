import matplotlib.pyplot as plt
import os
import shutil
from sklearn.metrics import confusion_matrix
# for loading and visualizing audio files
import librosa
import librosa.display
import numpy as np
# to play audio
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense, LSTM, Flatten, Bidirectional
from sklearn.model_selection import train_test_split
from tensorflow.keras.metrics import AUC
from tensorflow.keras.models import Sequential
from tqdm import tqdm
import itertools

if os.path.exists('./prep/'):
    shutil.rmtree('./prep/')
label_csv = 'input/environmental-sound-classification-50/esc50.csv'
audio_fpath = "input/environmental-sound-classification-50/audio/audio/44100/"
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
class AudioAugmentation:

    def read_audio_file(self, file_path):
        input_length = 220500
        data = librosa.load(file_path)[0]
        if len(data) > input_length:
            data = data[:input_length]
        else:
            data = np.pad(data, (0, max(0, input_length - len(data))), "constant")
        return data

    def add_noise(self, data):
        noise = np.random.randn(len(data))
        data_noise = data + 0.005 * noise
        return data_noise

    def shift(self, data):
        return np.roll(data, 22050)

    def stretch(self, data, rate=1):
        input_length = 220500
        data = librosa.effects.time_stretch(data, rate)
        if len(data) > input_length:
            data = data[:input_length]
        else:
            data = np.pad(data, (0, max(0, input_length - len(data))), "constant")
        return data

    def write_audio_file(self, file, data, sample_rate=44100):
        librosa.output.write_wav(file, data, sample_rate)



if __name__ == '__main__':
    audio_clips = os.listdir(audio_fpath)
    print("No. of .wav files in audio folder = ", len(audio_clips))
    df = pd.read_csv(label_csv)
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print(df.head())
    df.shape
    x, sr = librosa.load(audio_fpath + audio_clips[0], sr=44100)

    print(type(x), type(sr))
    print(x.shape, sr)

    aa = AudioAugmentation()
    extracted_data = []
    for index, row in tqdm(df.iterrows()):
        file_name = os.path.join(os.path.abspath(audio_fpath), str(row["filename"]))
        class_labels = row['category']
        y, sr = librosa.load(file_name, sr=44100)
        for i in range(8):
            if i == 1 or i > 3:
                data = aa.add_noise(y)
            if i % 3 == 2 or i == 7:
                data = aa.shift(y)
            if i % 3 == 0 or i == 7:
                data = aa.stretch(y, rate=1.25)
            feature = librosa.feature.mfcc(y=data, sr=sr)
            scaled_feature = np.mean(feature.T, axis=0)
            extracted_data.append([scaled_feature, class_labels])
    print(np.array(extracted_data).shape)
    feature_df = pd.DataFrame(extracted_data, columns=['feature', 'class'])

    print(feature_df.head())
    x = np.array(feature_df['feature'].tolist())
    target = np.array(feature_df['class'].tolist())
    y_new = pd.get_dummies(target)
    print(y_new.shape)
    y_new.head()
    X_train, X_test, y_train, y_test = train_test_split(x,
                                                        y_new,
                                                        test_size=0.2,
                                                        random_state=15)

    print(str(X_train.shape) + ', ' + str(X_test.shape))
    inp = (1, 20)
    np.array(X_train).reshape(12800, 1, 20)
    model = Sequential()
    model.add(Input(shape=(1, 20)))
    model.add(Bidirectional(LSTM(1024, return_sequences=True, recurrent_dropout=0.1)))
    model.add(Flatten())
    model.add(Dense(50, activation='softmax'))
    adam = tf.keras.optimizers.Adam(learning_rate=0.002)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy', AUC()])
    print(model.summary())
    es = EarlyStopping(
        monitor='val_accuracy',
        restore_best_weights=True,
        patience=30, min_delta=0.001
    )

    history = model.fit(np.array(X_train).reshape(12800, 1, 20),
                        np.array(y_train),
                        epochs=500,
                        callbacks=[es],
                        batch_size=100,
                        shuffle = True,
                        validation_data=(np.array(X_test).reshape(3200, 1, 20),
                                         np.array(y_test))
                        )
    model.save('LSTM_ESC50_trained_model2')
    y_pred = model.predict(np.array(x).reshape(16000, 1, 20))
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(np.array(y_new), axis=1)
    print(np.mean(y_pred_classes == y_true))
    m = AUC()
    m.update_state(y_new, y_pred)
    print(m.result())

    cm = confusion_matrix(y_pred_classes, y_true, )


    plt.clf()
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    plt.imshow(cm, cmap=plt.cm.Blues, interpolation='nearest')

    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(range(50)))
    plt.xticks(tick_marks, range(50), rotation=45)
    plt.yticks(tick_marks, range(50))
    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig('con_matrix_LSTM_ESC50.png')
