# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 11:25:58 2021

@author: xiatong
"""


import joblib
import librosa
import numpy as np

SR = 16000  # sample rate
SR_VGG = 16000  # VGG pretrained model sample rate
FRAME_LEN = int(SR / 10)  # 100 ms
HOP = int(FRAME_LEN / 2)  # 50%overlap, 5ms
MFCC_DIM = 13


def get_feature(file):
    y, sr = librosa.load(file, sr=SR, mono=True, offset=0.0, duration=None)
    yt, index = librosa.effects.trim(y, frame_length=FRAME_LEN, hop_length=HOP)
    yt_n = yt / np.max(np.abs(yt))  # normolized the sound
    return yt_n


user_all = {
    "train_cough_id": [],
    "vad_cough_id": [],
    "test_cough_id": [],
    "train_health_id": [],
    "vad_health_id": [],
    "test_health_id": [],
}


path = "0426_EN_used_task1"
with open("data_0426_en_task1.csv") as f:
    for index, line in enumerate(f):
        if index > 0:
            temp = line.strip().split(";")
            uid = temp[0]
            UID = temp[0]

            if "202" in uid:
                uid = "form-app-users"
            folder = temp[7]
            if uid == "MJQ296DCcN" and folder == "2020-11-26-17_00_54_657915":
                continue

            voice = temp[12]
            cough = temp[13]
            breath = temp[14]
            split = int(temp[15])
            label = int(temp[16])

            if split == 0 and label == 1:
                fold = "train_cough_id"
            if split == 0 and label == 0:
                fold = "train_health_id"
            if split == 1 and label == 1:
                fold = "vad_cough_id"
            if split == 1 and label == 0:
                fold = "vad_health_id"
            if split == 2 and label == 1:
                fold = "test_cough_id"
            if split == 2 and label == 0:
                fold = "test_health_id"
            fname_b = "/".join([path, uid, folder, breath])
            fname_c = "/".join([path, uid, folder, cough])
            fname_v = "/".join([path, uid, folder, voice])
            # if ';drycoough;' in line:
            # continue
            if UID not in user_all[fold]:
                user_all[fold].append(UID)

for f in user_all:
    print(f, len(user_all[f]))
f = open("audio_0426En_cough_users.pk", "wb")
joblib.dump(user_all, f)
f.close()
