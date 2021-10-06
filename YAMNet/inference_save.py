# Copyright 2019 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Inference demo for YAMNet."""
from __future__ import division, print_function

import os

import numpy as np
import params as yamnet_params
import resampy
import soundfile as sf
import yamnet as yamnet_model

output = open("covid19_data_0426_yamnet_all.list", "w")


def main():
    params = yamnet_params.Params()
    yamnet = yamnet_model.yamnet_frames_model(params)
    yamnet.load_weights("yamnet.h5")
    yamnet_classes = yamnet_model.class_names("yamnet_class_map.csv")

    """
    Important, preparing the input:
        1. Download the covid19_data_0426.zip from google drive.
        2. Unzip that and move all samples to ./covid19_data_0426_flatten.
    """
    path = "./covid19_data_0426_flatten"  # wav files in a flatten path
    files = os.listdir(path)
    for index, fname in enumerate(files):
        try:
            folder, date, audio = fname.split("___")
        except ValueError:
            print("bad samples")
        else:
            print(folder, date, audio)
            wav_data, sr = sf.read(path + fname, dtype=np.int16)
            assert wav_data.dtype == np.int16, "Bad sample type: %r" % wav_data.dtype
            waveform = wav_data / 32768.0  # Convert to [-1.0, +1.0]
            waveform = waveform.astype("float32")

            # Convert to mono and the sample rate expected by YAMNet.
            if len(waveform.shape) > 1:
                waveform = np.mean(waveform, axis=1)
            if sr != params.sample_rate:
                waveform = resampy.resample(waveform, sr, params.sample_rate)

            # Predict YAMNet classes.
            scores, embeddings, spectrogram = yamnet(waveform)
            # Scores is a matrix of (time_frames, num_classes) classifier scores.
            # Average them along time to get an overall classifier output for the clip.
            prediction = np.mean(scores, axis=0)
            # Report the highest-scoring classes and their scores.
            top5_i = np.argsort(prediction)[::-1][:5]
            print(
                folder
                + "/"
                + date
                + "/"
                + audio
                + ";"
                + ";".join([yamnet_classes[i] + ":{:.3f}".format(prediction[i]) for i in top5_i])
            )
            output.write(
                folder
                + "/"
                + date
                + "/"
                + audio
                + ";"
                + ";".join([yamnet_classes[i] + ":{:.3f}".format(prediction[i]) for i in top5_i])
            )
            output.write("\n")


if __name__ == "__main__":
    main()
