# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 17:18:28 2020

@author: XT
"""
from __future__ import print_function

import random

import joblib
import numpy as np
import pandas as pd
from sklearn import metrics

SR = 16000  # sample rate
import os  # noqa: E402
import sys  # noqa: E402

import librosa  # noqa: E402
import model_params as params  # noqa: E402

sys.path.append("../vggish")

from vggish_input import waveform_to_examples  # noqa: E402

SR_VGG = params.SR_VGG


def get_resort(files):
    """Re-sort the files under data path.

    :param files: file list
    :type files: list
    :return: alphabetic orders
    :rtype: list
    """
    name_dict = {}
    for sample in files:
        name = sample.lower()
        name_dict[name] = sample
    re_file = [name_dict[s] for s in sorted(name_dict.keys())]
    np.random.seed(222)
    np.random.shuffle(re_file)
    return re_file


def get_resort_test(files):
    """Re-sort the files under data path.

    :param files: file list
    :type files: list
    :return: alphabetic orders
    :rtype: list
    """
    name_dict = {}
    for sample in files:
        name = sample.lower()
        name_dict[name] = sample
    re_file = [name_dict[s] for s in sorted(name_dict.keys())]

    return re_file


def get_aug(y, type):
    """Augment data for training, validation and testing.
    :param data_path: path
    :type data_path: str
    :param is_aug: using augmentation
    :type is_aug: bool
    :return: batch
    :rtype: list
    """
    if type == "noise":
        y_aug = y + 0.005 * np.random.normal(0, 1, len(y))
    if type == "pitchspeed":
        step = np.random.uniform(-6, 6)
        y_aug = librosa.effects.pitch_shift(y, SR, step)
    yt_n = y_aug / np.max(np.abs(y_aug))  # re-normolized the sound
    return yt_n


def spec_augment(spec: np.ndarray, num_mask=2, freq_masking_max_percentage=0.1, time_masking_max_percentage=0.2):

    spec = spec.copy()
    for i in range(num_mask):
        all_frames_num, all_freqs_num = spec.shape
        freq_percentage = random.uniform(0.0, freq_masking_max_percentage)

        num_freqs_to_mask = int(freq_percentage * all_freqs_num)
        f0 = np.random.uniform(low=0.0, high=all_freqs_num - num_freqs_to_mask)
        f0 = int(f0)
        spec[:, f0 : f0 + num_freqs_to_mask] = 0

        time_percentage = random.uniform(0.0, time_masking_max_percentage)

        num_frames_to_mask = int(time_percentage * all_frames_num)
        t0 = np.random.uniform(low=0.0, high=all_frames_num - num_frames_to_mask)
        t0 = int(t0)
        spec[t0 : t0 + num_frames_to_mask, :] = 0

    return spec


def load_data(data_path, is_aug):
    """Load data for training, validation and testing.
    :param data_path: path
    :type data_path: str
    :param is_aug: using augmentation
    :type is_aug: bool
    :return: batch
    :rtype: list
    """
    print("start to load data:", data_path)
    data = joblib.load(open(data_path + "_cough.pk", "rb"))  # load positive samples
    data2 = joblib.load(open(data_path + "_health.pk", "rb"))  # load negative samples
    data.update(data2)
    user_all = joblib.load(open(data_path + "_cough_users.pk", "rb"))

    train_task = []
    covidcnt = 0
    noncvcnt = 0
    for uid in get_resort(user_all["train_cough_id"]):
        for temp in data[uid]:
            train_task.append(
                {"breath": temp["breath"], "cough": temp["cough"], "voice": temp["voice"], "label": [0, 1]}
            )
            covidcnt += 1
    for uid in get_resort(user_all["train_health_id"]):
        for temp in data[uid]:
            train_task.append(
                {"breath": temp["breath"], "cough": temp["cough"], "voice": temp["voice"], "label": [1, 0]}
            )
            noncvcnt += 1
    print("cough:", covidcnt, "health:", noncvcnt)
    total = len(train_task)

    # upsampling by repeating some covid to balance the class
    np.random.seed(1)
    add_covid = np.random.choice(range(covidcnt), (noncvcnt - covidcnt) * 2, replace=False)
    add_sample = [train_task[i] for i in add_covid]
    train_task = train_task + add_sample
    total = len(train_task)
    print("add cough:", noncvcnt - covidcnt, "total:", total)

    vad_task = []
    covidcnt = 0
    noncvcnt = 0
    for uid in get_resort(user_all["vad_cough_id"]):
        for temp in data[uid]:
            vad_task.append(
                {"breath": temp["breath"], "cough": temp["cough"], "voice": temp["voice"], "label": [0, 1]}
            )
            covidcnt += 1
    for uid in get_resort(user_all["vad_health_id"]):
        for temp in data[uid]:
            vad_task.append(
                {"breath": temp["breath"], "cough": temp["cough"], "voice": temp["voice"], "label": [1, 0]}
            )
            noncvcnt += 1
    print("cough:", covidcnt, "health:", noncvcnt)

    test_task = []
    covidcnt = 0
    noncvcnt = 0
    for uid in get_resort(user_all["test_cough_id"]):
        for temp in data[uid]:
            test_task.append(
                {"breath": temp["breath"], "cough": temp["cough"], "voice": temp["voice"], "label": [0, 1]}
            )
            covidcnt += 1
    for uid in get_resort(user_all["test_health_id"]):
        for temp in data[uid]:
            test_task.append(
                {"breath": temp["breath"], "cough": temp["cough"], "voice": temp["voice"], "label": [1, 0]}
            )
            noncvcnt += 1
    print("cough:", covidcnt, "health:", noncvcnt)
    test_task = test_task + test_task[:5]

    # suffle samples
    np.random.seed(222)
    np.random.shuffle(train_task)
    np.random.seed(222)
    np.random.shuffle(vad_task)
    np.random.seed(222)
    np.random.shuffle(test_task)

    return train_task, vad_task, test_task


def load_vad_data(data_path):
    """Load vad data only."""
    print("start to load data:", data_path)
    data = joblib.load(open(data_path + "_cough.pk", "rb"))
    data2 = joblib.load(open(data_path + "_health.pk", "rb"))
    data.update(data2)
    user_all = joblib.load(open(data_path + "_cough_users.pk", "rb"))

    vad_task = []
    covidcnt = 0
    noncvcnt = 0

    # i = 0
    for uid in get_resort(user_all["train_cough_id"]):
        for temp in data[uid]:
            vad_task.append(
                {"breath": temp["breath"], "cough": temp["cough"], "voice": temp["voice"], "label": [0, 1]}
            )
            covidcnt += 1
    for uid in get_resort(user_all["train_health_id"]):
        for temp in data[uid]:
            vad_task.append(
                {"breath": temp["breath"], "cough": temp["cough"], "voice": temp["voice"], "label": [1, 0]}
            )
            noncvcnt += 1
    for uid in get_resort(user_all["vad_cough_id"]):
        for temp in data[uid]:
            vad_task.append(
                {"breath": temp["breath"], "cough": temp["cough"], "voice": temp["voice"], "label": [0, 1]}
            )
            covidcnt += 1
    for uid in get_resort(user_all["vad_health_id"]):
        for temp in data[uid]:
            vad_task.append(
                {"breath": temp["breath"], "cough": temp["cough"], "voice": temp["voice"], "label": [1, 0]}
            )
            noncvcnt += 1
    print("cough:", covidcnt, "health:", noncvcnt)
    np.random.seed(222)
    np.random.shuffle(vad_task)
    return vad_task


def load_test_data(data_path):
    """Load test data only."""
    print("start to load data:", data_path)
    data = joblib.load(open(data_path + "_cough.pk", "rb"))
    data2 = joblib.load(open(data_path + "_health.pk", "rb"))
    data.update(data2)
    user_all = joblib.load(open(data_path + "_cough_users.pk", "rb"))

    test_task = []
    covidcnt = 0
    noncvcnt = 0

    i = 0
    for uid in get_resort_test(user_all["test_cough_id"]):
        for temp in data[uid]:
            i += 1
            test_task.append(
                {"breath": temp["breath"], "cough": temp["cough"], "voice": temp["voice"], "label": [0, 1]}
            )
            covidcnt += 1
    for uid in get_resort_test(user_all["test_health_id"]):
        for temp in data[uid]:
            i += 1
            test_task.append(
                {"breath": temp["breath"], "cough": temp["cough"], "voice": temp["voice"], "label": [1, 0]}
            )
            noncvcnt += 1
    return test_task


def load_test_dict_data(data_path):
    """load dict data with labal."""
    print("start to load data:", data_path)
    data = joblib.load(open(data_path, "rb"))
    return data


def get_input(sample):
    """transfer audio input into spectrogram."""
    vgg_b = waveform_to_examples(sample["breath"], SR_VGG)
    vgg_c = waveform_to_examples(sample["cough"], SR_VGG)
    vgg_v = waveform_to_examples(sample["voice"], SR_VGG)

    index = vgg_b.shape[0]
    index2 = vgg_c.shape[0] + index
    vgg_input = np.concatenate((vgg_b, vgg_c, vgg_v), axis=0)

    labels = sample["label"]
    symptoms = [[1] * 13]  # sample['sym']

    return vgg_input, [[index]], [[index2]], labels, symptoms


def get_metrics(probs, labels):
    """calculate metrics.
    :param probs: list
    :type probs: float
    :param labels: list
    :type labels: int
    :return: metrics
    """
    probs = np.array(probs)
    probs = np.squeeze(probs)

    predicted = []
    for i in range(len(probs)):
        if probs[i][0] > 0.5:
            predicted.append(0)
        else:
            predicted.append(1)

    label = np.array(labels)
    label = np.squeeze(label)

    predicted = np.array(predicted)
    predicted = np.squeeze(predicted)

    # pre = metrics.precision_score(label, predicted)
    # acc = metrics.accuracy_score(label, predicted)
    auc = metrics.roc_auc_score(label, probs[:, 1])
    precision, recall, _ = metrics.precision_recall_curve(label, probs[:, 1])
    # rec = metrics.recall_score(label, predicted)

    TN, FP, FN, TP = metrics.confusion_matrix(label, predicted).ravel()
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN)
    # Specificity or true negative rate
    TNR = TN / (TN + FP)

    # PPV = TP/(TP + FP)
    # NPV = TN/(TN + FN)

    fpr, tpr, thresholds = metrics.roc_curve(label, probs[:, 1])
    index = np.where(tpr > 0.9)[0][0] - 1
    print(
        "AUC:"
        + "{:.2f}".format(auc)
        + " Sensitivity:"
        + "{:.2f}".format(TPR)
        + " Specificity:"
        + "{:.2f}".format(TNR)
        + " spe@90%sen:"
        + "{:.2f}".format(1 - fpr[index])
    )

    return auc, TPR, TNR, 1 - fpr[index]


def get_metrics_t(probs, label):
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
            get_metrics_t(sample_pro, sample_label)
        except ValueError:
            np.random.seed(1001)  # Para2
            sample = np.random.choice(range(len(data)), len(data), replace=True)
            samples = [data[i] for i in sample]
            sample_pro = [x[0] for x in samples]
            sample_label = [x[1] for x in samples]
        else:
            auc, TPR, TNR = get_metrics_t(sample_pro, sample_label)
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


def is_exists(path):
    """Check directory exists."""
    if not os.path.exists(path):
        print("Not exists: {}".format(path))
        return False
    return True


def maybe_create_directory(dirname):
    """Check directory exists or create it."""
    if not os.path.exists(dirname):
        os.makedirs(dirname)


# model - already trained keras model with dropout
def mc_dropout(predictions, T):
    # predictions shape: (I, T, C) T - monte carlo samples, I input size, C number of classes

    # shape: (I, C)
    mean = np.mean(predictions, axis=1)
    mean = np.squeeze(mean)
    print("mean:", mean.shape)

    # shape: (I)
    variance = -1 * np.sum(np.log(mean) * mean, axis=1)
    return (mean, variance)
