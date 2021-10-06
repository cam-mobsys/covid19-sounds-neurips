# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 11:55:32 2021

@author: xiatong

load csv and test the performance
10-fold cross validation
"""


import warnings

import numpy as np
from sklearn import decomposition, metrics, preprocessing
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

import pandas as pd  # noqa: E402

inputFile = "features_6373.csv"
df_features = pd.read_csv(inputFile)
df_cough = df_features["cough_feature"].map(lambda x: [float(v) for v in x.split(";")])
cough = np.array([x for x in df_cough])
df_breath = df_features["breath_feature"].map(lambda x: [float(v) for v in x.split(";")])
breath = np.array([x for x in df_breath])
df_voice = df_features["voice_feature"].map(lambda x: [float(v) for v in x.split(";")])
voice = np.array([x for x in df_voice])
x_data = np.concatenate([cough, breath, voice], axis=1)
# x_data = breath
# x_data = cough
# x_data = voice


y_label = np.array(df_features["label"])
y_set = np.array(df_features["fold"])

x_data_train = x_data[y_set == "train"]
y_label_train = y_label[y_set == "train"]
x_data_vad = x_data[y_set == "validatio"]
y_label_vad = y_label[y_set == "validatio"]
x_data_test = x_data[y_set == "test"]
y_label_test = y_label[y_set == "test"]

# scale data
scaler = preprocessing.StandardScaler().fit(x_data_train)
x_train_n = scaler.transform(x_data_train)
x_test_n = scaler.transform(x_data_test)
x_vad_n = scaler.transform(x_data_vad)

# use PCA to reduce the feature dimension
pca = decomposition.PCA(0.99)
pca.fit(x_train_n)
x_train_n_pca = pca.fit_transform(x_train_n)
x_test_n_pca = pca.transform(x_test_n)
x_vad_n_pca = pca.transform(x_vad_n)


clf = SVC(C=0.001, kernel="linear", gamma="auto", probability=True)
clf = clf.fit(x_train_n_pca, y_label_train)

predicted = clf.predict(x_vad_n_pca)
probs = clf.predict_proba(x_vad_n_pca)
auc = metrics.roc_auc_score(y_label_vad, probs[:, 1])
precision, recall, _ = metrics.precision_recall_curve(y_label_vad, probs[:, 1])
se = metrics.recall_score(y_label_vad, predicted, labels=[1], average=None)[0]
sp = metrics.recall_score(y_label_vad, predicted, labels=[0], average=None)[0]
print("auc", auc, "SE", se, "SP", sp)

predicted = clf.predict(x_test_n_pca)
probs = clf.predict_proba(x_test_n_pca)
auc = metrics.roc_auc_score(y_label_test, probs[:, 1])
precision, recall, _ = metrics.precision_recall_curve(y_label_test, probs[:, 1])
se = metrics.recall_score(y_label_test, predicted, labels=[1], average=None)[0]
sp = metrics.recall_score(y_label_test, predicted, labels=[0], average=None)[0]
print("auc", auc, "SE", se, "SP", sp)


def get_metrics(probs, label):
    predicted = []
    for i in range(len(probs)):
        if probs[i] > 0.5:
            predicted.append(1)
        else:
            predicted.append(0)

    auc = metrics.roc_auc_score(label, probs)
    TN, FP, FN, TP = metrics.confusion_matrix(label, predicted).ravel()
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP * 1.0 / (TP + FN)
    # Specificity or true negative rate
    TNR = TN * 1.0 / (TN + FP)

    return auc, TPR, TNR


def get_CI(data, AUC, Sen, Spe):
    AUCs = []
    TPRs = []
    TNRs = []
    for s in range(1000):
        np.random.seed(s)  # Para2
        sample = np.random.choice(range(len(data)), len(data), replace=True)
        samples = [data[i] for i in sample]
        sample_pro = [x[0] for x in samples]
        sample_label = [x[1] for x in samples]
        try:
            get_metrics(sample_pro, sample_label)
        except ValueError:
            np.random.seed(1001)  # Para2
            sample = np.random.choice(range(len(data)), len(data), replace=True)
            samples = [data[i] for i in sample]
            sample_pro = [x[0] for x in samples]
            sample_label = [x[1] for x in samples]
        else:
            auc, TPR, TNR = get_metrics(sample_pro, sample_label)
        AUCs.append(auc)
        TPRs.append(TPR)
        TNRs.append(TNR)

    q_0 = pd.DataFrame(np.array(AUCs)).quantile(0.025)[0]  # 2.5% percentile
    q_1 = pd.DataFrame(np.array(AUCs)).quantile(0.975)[0]  # 97.5% percentile

    q_2 = pd.DataFrame(np.array(TPRs)).quantile(0.025)[0]  # 2.5% percentile
    q_3 = pd.DataFrame(np.array(TPRs)).quantile(0.975)[0]  # 97.5% percentile

    q_4 = pd.DataFrame(np.array(TNRs)).quantile(0.025)[0]  # 2.5% percentile
    q_5 = pd.DataFrame(np.array(TNRs)).quantile(0.975)[0]  # 97.5% percentile

    print(
        str(AUC.round(2))
        + "("
        + str(q_0.round(2))
        + "-"
        + str(q_1.round(2))
        + ")"
        + "&"
        + str(Sen.round(2))
        + "("
        + str(q_2.round(2))
        + "-"
        + str(q_3.round(2))
        + ")"
        "&" + str(Spe.round(2)) + "(" + str(q_4.round(2)) + "-" + str(q_5.round(2)) + ")"
    )


data = [[probs[i, 1], y_label_test[i]] for i in range(len(y_label_test))]
AUC, Sen, Spe = get_metrics(probs[:, 1], y_label_test)
get_CI(data, AUC, Sen, Spe)
