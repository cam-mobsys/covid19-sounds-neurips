# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 11:55:32 2021

@author: xiatong

load csv and test the performance

"""


import numpy as np
import pandas as pd
from sklearn import decomposition, metrics, preprocessing
from sklearn.svm import SVC


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


inputFile = "features_384.csv"
df_features = pd.read_csv(inputFile)
df_cough = df_features["cough_feature"].map(lambda x: [float(v) for v in x.split(";")])
cough = np.array([x for x in df_cough])
df_breath = df_features["breath_feature"].map(lambda x: [float(v) for v in x.split(";")])
breath = np.array([x for x in df_breath])
df_voice = df_features["voice_feature"].map(lambda x: [float(v) for v in x.split(";")])
voice = np.array([x for x in df_voice])
# x_data = np.concatenate([cough,breath,voice],axis=1)
x_data = voice

y_label = np.array(df_features["label"])
y_set = np.array(df_features["fold"])

x_data_train = x_data[y_set == 0]
y_label_train = y_label[y_set == 0]
x_data_vad = x_data[y_set == 1]
y_label_vad = y_label[y_set == 1]
x_data_test = x_data[y_set == 2]
y_label_test = y_label[y_set == 2]

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

for c in [0.0001]:
    print(c)
    clf = SVC(C=c, kernel="linear", gamma="auto", probability=True)
    # clf = XGBClassifier(learning_rate =0.1, n_estimators=1000,
    #         max_depth=8,min_child_weight=1, gamma=0, subsample=0.8,
    #         colsample_bytree=0.8, objective= 'binary:logistic',
    #         nthread=8, scale_pos_weight=1, seed=0)

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

    data = [[probs[i, 1], y_label_test[i]] for i in range(len(y_label_test))]
    AUC, Sen, Spe = get_metrics(probs[:, 1], y_label_test)
    get_CI(data, AUC, Sen, Spe)


# clf = SVC(C=0.001, kernel='linear',gamma='auto', probability=True)
# auc 0.7086871921345558 SE 0.6648501362397821 SP 0.6520681265206812
# auc 0.7108886021859103 SE 0.6505944517833554 SP 0.6679636835278858
# 0.71(0.69-0.73)&0.64(0.61-0.66)&0.68(0.66-0.7)

# #384
# 0.01
# auc 0.7240216259936223 SE 0.6444141689373297 SP 0.6727493917274939
# auc 0.7083999403749184 SE 0.6373844121532365 SP 0.6627756160830091
# 0.71(0.69-0.73)&0.63(0.6-0.65)&0.67(0.65-0.7)
# 0.05
# auc 0.7214692018536566 SE 0.6376021798365122 SP 0.6727493917274939
# auc 0.70406812679582 SE 0.6360634081902246 SP 0.6640726329442282
# 0.7(0.69-0.72)&0.62(0.6-0.65)&0.67(0.65-0.7)
# 0.001
# auc 0.7297562932171814 SE 0.6362397820163488 SP 0.6909975669099757
# auc 0.7253151305498016 SE 0.6433289299867899 SP 0.6867704280155642
# 0.73(0.71-0.74)&0.64(0.62-0.67)&0.69(0.66-0.71)
# 0.0001
# auc 0.7306463268296239 SE 0.6103542234332425 SP 0.7068126520681265
# auc 0.7279068083961708 SE 0.631439894319683 SP 0.704928664072633
# 0.73(0.71-0.74)&0.65(0.62-0.67)&0.69(0.67-0.71)
