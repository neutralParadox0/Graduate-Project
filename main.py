# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import tkinter as tk
from tkinter import filedialog
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import tensorflow as tf
from tensorflow import keras
import os
import shutil
from PIL import Image
from skimage import transform
import sys
from tensorflow.keras.preprocessing.image import ImageDataGenerator

gtzan_class_labels = {0:'blues',
                      1:'classical',
                      2:'country',
                      3:'disco',
                      4:'hiphop',
                      5:'jazz',
                      6:'metal',
                      7:'pop',
                      8:'reggae',
                      9:'rock'}
esc50_class_labels = {0:'airplane'
                    ,1:'breathing'
                    ,2:'brushing_teeth'
                    ,3:'can_opening'
                    ,4:'car_horn'
                    ,5:'cat'
                    ,6:'chainsaw'
                    ,7:'chirping_birds'
                    ,8:'church_bells'
                    ,9:'clapping'
                    ,10:'clock_alarm'
                    ,11:'clock_tick'
                    ,12:'coughing'
                    ,13:'cow'
                    ,14:'crackling_fire'
                    ,15:'crickets'
                    ,16:'crow'
                    ,17:'crying_baby'
                    ,18:'dog'
                    ,19:'door_wood_creaks'
                    ,20:'door_wood_knock'
                    ,21:'drinking_sipping'
                    ,22:'engine'
                    ,23:'fireworks'
                    ,24:'footsteps'
                    ,25:'frog'
                    ,26:'glass_breaking'
                    ,27:'hand_saw'
                    ,28:'helicopter'
                    ,29:'hen'
                    ,30:'insects'
                    ,31:'keyboard_typing'
                    ,32:'laughing'
                    ,33:'mouse_click'
                    ,34:'pig'
                    ,35:'pouring_water'
                    ,36:'rain'
                    ,37:'rooster'
                    ,38:'sea_waves'
                    ,39:'sheep'
                    ,40:'siren'
                    ,41:'sneezing'
                    ,42:'snoring'
                    ,43:'thunderstorm'
                    ,44:'toilet_flush'
                    ,45:'train'
                    ,46:'vacuum_cleaner'
                    ,47:'washing_machine'
                    ,48:'water_drops'
                    ,49:'wind'}


def convert(inFile):
    if clicked.get() == 'Environmental':
        x, sr = librosa.load(inFile, sr=44100)
    elif clicked.get() == 'Music Genre':
        x, sr = librosa.load(inFile, sr=22050)
    length = sr*5-1
    sam = x[length]
    mfcc = librosa.feature.mfcc(y=np.array(x[:length]), sr=sr, n_mfcc=20)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    mfcc_gram = librosa.amplitude_to_db(np.subtract(mfcc,np.mean(mfcc)), ref=np.min)
    sample = np.array(mfcc_mean).reshape(1, 1, 20)
    librosa.display.specshow(mfcc_gram, sr=sr)
    if os.path.exists('./tmp/'):
        shutil.rmtree('./tmp/')
    os.makedirs('./tmp/')
    os.makedirs('./tmp/class_sample/')
    out_file = './tmp/class_sample/' + inFile.split('/')[-1].replace('.wav', '.png')
    plt.colorbar(format='%+2.0f dB')
    plt.savefig(out_file)
    plt.clf()
    smpImg = Image.open(out_file)
    class_gen = ImageDataGenerator()
    smpImg = class_gen.flow_from_directory(directory='./tmp/')
    # smpImg = np.array(smpImg).astype('float32')
    # smpImg = transform.resize(smpImg, (256,256,3))
    # smpImg = np.expand_dims(smpImg, axis=0)
    return sample, smpImg


def classify():
    filePath = fileEntry.get()
    print(filePath)
    samp, sampImg = convert(inFile=filePath)
    pred_results = []
    for i in range(3):
        if clicked.get() == 'Environmental':
            if i == 0:
                pred = models[i].predict(samp)
            else:
                pred = models[i].predict(sampImg, steps=1)
        elif clicked.get() == 'Music Genre':
            if i == 0:
                pred = models[i+3].predict(samp)
            else:
                pred = models[i+3].predict(sampImg,steps=1)
        pred_results.append(pred)
    # top5 = np.argpartition(pred, -5)[-5:]
    for j in range(3):
        pred_class = np.argmax(pred_results[j], axis=1)
        if clicked.get() == 'Environmental':
            results[j].config(text=esc50_class_labels[pred_class[0]])
        elif clicked.get() == 'Music Genre':
            results[j].config(text=gtzan_class_labels[pred_class[0]])


def brwsFiles():
    filename = filedialog.askopenfilename(title='Choose file to classify',
                                          filetypes=[('WAV files', '*.wav*')])
    print(filename)
    fileEntry.delete(0, tk.END)
    fileEntry.insert(0, filename)


def test():
    dir = 'input/gtzan-dataset-music-genre-classification/Data/genres_original/blues/'
#     tst = []
#     results0 = {}
#     results1 = {}
    results = {}
    for filename in os.listdir(dir):
        samp, sampImg = convert(dir+filename)
        pred = models[1].predict(samp)
        pred_class = np.argmax(pred)
        if pred_class in results:
            results[pred_class] = results[pred_class] + 1
        else:
            results[pred_class] = 1
    return results
#         # for i in range(2):
#         # if i == 0:
#             if clicked.get() == 'Environmental':
#                 pred = models[0].predict(samp)
#             else:
#                 pred = models[1].predict(samp)
#             top5 = np.argpartition(pred, -5)[-5:]
#             pred_class = np.argmax(pred, axis=1)
#             results0[filename] = pred_class
#         # else:
#             pred = models[i].predict(np.array(sampImg))  # .reshape(1, 256, 256, 3))
#             top5 = np.argpartition(pred, -5)[-5:]
#             pred_class = np.argmax(pred, axis=1)
#             results1[filename] = pred_class
#     tst.append(results0)
#     tst.append(results1)
#     return tst


options = ['Environmental', 'Music Genre']
models = []
models.append(keras.models.load_model('LSTM_ESC50_trained_model'))
models.append(keras.models.load_model('ResNet50_ESC50_Trained_model'))
models.append(keras.models.load_model('InceptionV3_ESC50_trained_model'))
models.append(keras.models.load_model('LSTM_GTZAN_Training_model'))
models.append(keras.models.load_model('ResNet50_GTZAN_Training_model'))
models.append(keras.models.load_model('InceptionV3_GTZAN_trained_model'))
# test_results = test()
m = tk.Tk()
m.title('Audio Classifier')
browseBtn = tk.Button(m, text='Browse', command=brwsFiles)
clicked = tk.StringVar()
clicked.set('Environmental')
dataChoose = tk.OptionMenu(m, clicked, *options)
dataChoose.grid(row=0)
tk.Label(m, text='.wav File: ').grid(row=1)
fileEntry = tk.Entry(m)
fileEntry.grid(row=1, column=1, )
browseBtn.grid(row=1, column=2)
classifyBtn = tk.Button(m, text='Classify', command=classify)
# tk.Label(m).grid(row=0)
classifyBtn.grid(row=2, columnspan=2)
results = []
results.append(tk.Label(m, text='LSTM'))
results.append(tk.Label(m, text='ResNet50'))
results.append(tk.Label(m, text='Inception'))
for i in range(3):
    results[i].grid(row=3, column=i)
if __name__ == '__main__':
    m.mainloop()
    sys.exit()
