# -*- coding: utf-8 -*-
"""
Created on Sun Aug 22 22:45:18 2021

@author: xiato
"""


from collections import Counter

import pandas as pd

df = pd.read_csv("data_0426_en_task1.csv", sep=";")


df_train_pos = df[(df["split"] == 2) & (df["label"] == 0)]
df_train_pos = df_train_pos[["Uid", "Sex", "Age"]].drop_duplicates(keep="first")

sex = Counter(df_train_pos["Sex"])
age = Counter(df_train_pos["Age"])

print("Male", sex["Male"], sex["Male"] / len(df_train_pos))
print("Female", sex["Female"], sex["Female"] / len(df_train_pos))

print(
    "16-29", age["0-19"] + age["16-19"] + age["20-29"], (age["0-19"] + age["16-19"] + age["20-29"]) / len(df_train_pos)
)
print("30-39", age["30-39"], (age["30-39"]) / len(df_train_pos))
print("40-49", age["40-49"], (age["40-49"]) / len(df_train_pos))
print("50-59", age["50-59"], (age["50-59"]) / len(df_train_pos))
print("60-69", age["60-69"], (age["60-69"]) / len(df_train_pos))
print("70-", age["70-79"] + age["80-89"] + age["90-"], (age["70-79"] + age["80-89"] + age["90-"]) / len(df_train_pos))
