# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 14:49:15 2021

@author: Jing Han

Sound type identification from YAMNet inference
"""


import numpy as np

yamnet_outfile = "data_0426_yamnet_all.list"
outfile = yamnet_outfile.replace(".list", "_final.list")

yamnetData = np.genfromtxt(yamnet_outfile, delimiter=";", dtype=np.str)
prediction_list = []
prediction_prob = []
prediction_first = []
for i in range(0, yamnetData.shape[0]):
    # 	print(yamnetData[i,0])
    fileName = yamnetData[i, 0]
    if ("cough" in yamnetData[i, 0]) and (
        "Cough" in yamnetData[i, 1]
        or "Cough" in yamnetData[i, 2]
        or "Cough" in yamnetData[i, 3]
        or "Cough" in yamnetData[i, 4]
        or "Cough" in yamnetData[i, 5]
    ):
        prediction_list.append(["c"])
    elif ("breath" in yamnetData[i, 0]) and (
        "Breathing" in yamnetData[i, 1]
        or "Breathing" in yamnetData[i, 2]
        or "Breathing" in yamnetData[i, 3]
        or "Breathing" in yamnetData[i, 4]
        or "Breathing" in yamnetData[i, 5]
    ):
        prediction_list.append(["b"])
    elif (
        "Speech" in yamnetData[i, 1]
        and float(yamnetData[i, 1].split(":")[1][:-1]) >= 0.4
        and ("read" in yamnetData[i, 0] or "voice" in yamnetData[i, 0])
    ):
        prediction_list.append(["v"])
    else:
        prediction_list.append(["n"])
    prediction_prob.append([(yamnetData[i, 1].split(":")[1][:-1])])
    prediction_first.append([(yamnetData[i, 1].split(":")[0][:])])
# print (prediction_list, prediction_prob, prediction_first)

data = np.hstack([yamnetData, prediction_list, prediction_prob, prediction_first])
print(data.shape)
np.savetxt(outfile, data, fmt="%s", delimiter=";", header="filename;p1;p2;p3;p4;p5;label;firstProb;firstCatogory")
