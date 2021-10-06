# -*- coding: utf-8 -*-
"""
Created on Sat May 15 16:09:51 2021

@author: xiatong

1. resample and normalize
2. extract opensmile features
3. save to csv

"""
import os

import librosa
import numpy as np
import soundfile as sf

SR = 16000  # sample rate
SR_VGG = 16000  # VGG pretrained model sample rate
FRAME_LEN = int(SR / 10)  # 100 ms
HOP = int(FRAME_LEN / 2)  # 50%overlap, 5ms


##########################################
path = "../data/data_0426_en_task2/"


def extract_opensmile(sample):
    y, sr = librosa.load(path + sample, sr=SR, mono=True, offset=0.0, duration=None)
    yt, index = librosa.effects.trim(y, frame_length=FRAME_LEN, hop_length=HOP)
    yt = yt / np.max(np.abs(yt))  # normolized the sound

    savepath = "OpenSmile_features/"
    name = sample.replace("\\", "___")
    sf.write(savepath + name + "_normalized.wav", yt, SR, subtype="PCM_24")

    # change to path of opensmil
    cmd = (
        'SMILExtract -C opensmile-3.0-win-x64/config/is09-13/IS13_ComParE.conf -I "'
        + savepath
        + name
        + '_normalized.wav"'
        + " -O "
        + '"'
        + savepath
        + name
        + '.ex6373_n.txt"'
    )
    os.system(cmd)
    cmd = (
        'SMILExtract -C opensmile-3.0-win-x64/config/is09-13/IS09_emotion.conf -I "'
        + savepath
        + name
        + '_normalized.wav"'
        + " -O "
        + '"'
        + savepath
        + name
        + '.ex384_n.txt"'
    )
    os.system(cmd)
    print(cmd)


def extract_opensmile2(sample):
    """opensmile ouput existing"""
    savepath = "OpenSmile_features/"
    # sample = savepath + sample.replace("\\", "___") +'.ex6373_n.txt'
    # with open(sample, 'r') as f:
    #     temp = f.readlines()
    #     features = temp[6380].split(',')[1:-1]
    #     features = [x for x in features]
    # f.close()
    sample = savepath + sample.replace("\\", "___") + ".ex384_n.txt"
    with open(sample, "r") as f:
        temp = f.readlines()
        features = temp[391].split(",")[1:-1]
        features = [x for x in features]
    f.close()

    return ";".join(features)


# with open('../data/data_0426_en_all.csv')as f:
#     for i, line in enumerate(f):
#         if i > 0:
#             _,cough_path,breath_path,voice_path,_,_,_,_ = line.strip().split(',')
#             extract_opensmile(cough_path)
#             extract_opensmile(breath_path)
#             extract_opensmile(voice_path)


# output = open('features_6373.csv','w')

output = open("features_384.csv", "w")


output.write("Index,cough_feature,breath_feature,voice_feature,label,uid,categs,fold" + "\n")
with open("../data/data_0426_en_task2.csv") as f:
    for i, line in enumerate(f):
        if i > 0:
            index, cough_path, breath_path, voice_path, label, uid, categs, fold = line.strip().split(",")
            cough = extract_opensmile2(cough_path)
            breath = extract_opensmile2(breath_path)
            voice = extract_opensmile2(voice_path)
            output.write(",".join([index, cough, breath, voice, label, uid, categs, fold]))
            output.write("\n")
