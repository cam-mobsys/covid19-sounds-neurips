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
path = "../data/0426_EN_used_task1"


def extract_opensmile(sample):
    y, sr = librosa.load(path + sample, sr=SR, mono=True, offset=0.0, duration=None)
    yt, index = librosa.effects.trim(y, frame_length=FRAME_LEN, hop_length=HOP)
    yt = yt / np.max(np.abs(yt))  # normolized the sound

    savepath = "OpenSmile_features/"
    name = sample.replace("\\", "___")
    sf.write(savepath + name + "_normalized.wav", yt, SR, subtype="PCM_24")

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


##########################################
def extract_opensmile2(sample):
    """opensmile ouput existing"""
    savepath = "I:/propocess/NIPS_prepared_data/opensmile/"
    # sample = savepath + sample + '.ex6373_n.txt'
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


# output = open('features_6373.csv','w')
output = open("features_384.csv", "w")

cate = {"1": "cough", "0": "None"}

output.write("Index,cough_feature,breath_feature,voice_feature,label,uid,categs,fold" + "\n")
with open("../data/data_0426_en_task1.csv") as f:
    for i, line in enumerate(f):
        if i > 0:
            temp = line.strip().split(";")
            uid = temp[0]
            print(i, uid)
            folder = temp[7]
            if uid == "MJQ296DCcN" and folder == "2020-11-26-17_00_54_657915":
                continue
            voice = temp[12]
            cough = temp[13]
            breath = temp[14]
            split = temp[15]
            label = temp[16]
            if "202" in uid:
                UID = "form-app-users"
            else:
                UID = uid
            fname_b = "___".join([UID, folder, breath])
            fname_c = "___".join([UID, folder, cough])
            fname_v = "___".join([UID, folder, voice])

            cough = extract_opensmile2(fname_c)
            breath = extract_opensmile2(fname_b)
            voice = extract_opensmile2(fname_v)
            output.write(",".join([str(i), cough, breath, voice, label, uid, cate[label], split]))
            output.write("\n")
