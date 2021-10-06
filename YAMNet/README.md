# Audio Quality Classification Tool

As mentioned in the main readme, we provide a tool to classify if an audio sample is of enough quality to be used for
inference. This tool employs another known network, namely [Yamnet][1]. The sample should contain either:

- breathing (will be tagged with `'b'`),
- cough (will be tagged with `'c'`),
- or voice (will be tagged with `'v'`).

Silent and noisy samples will be filtered accordingly and labelled as `'n'`. This labelling will exclude such files
from further experiments. We have already prepared the annotations for all samples within the provided dataset
(see `Voice check`, `Cough check`, `Breath check` fields in
`results_raw_20210426_lan_yamnet_android/ios/web_noloc.csv`). Nevertheless, we make the tool available in case one
wishes to follow a different data selection process. Please note that this tool requires a **different** environment
to be used, as [Yamnet][1] _requires_ Tensorflow 2.0. To help facilitate the environment we provide an indicative
requirements file, which you can find [here](./requirements.txt).

## YAMNet

For clarity, we put segments of the [YAMNet][1] repository README in order to aid the reader have context about the network
we use. [YAMNet][1] is a pretrained deep net that predicts 521 audio event classes based on the
[AudioSet-YouTube corpus][2], and employing the [Mobilenet_v1][3] depthwise-separable convolution architecture.

We provide instructions on how to install the model in order to be able to use our Audio Quality Classification tool.

## Installation

YAMNet depends on the following Python packages:

- [`numpy`](http://www.numpy.org/)
- [`resampy`](http://resampy.readthedocs.io/en/latest/)
- [`tensorflow`](http://www.tensorflow.org/)
- [`pysoundfile`](https://pysoundfile.readthedocs.io/)

These are all easily installable via, e.g., `pip install numpy` (as in the example command sequence below). Any
reasonably recent version of these packages should work. As mentioned previously, we provide a sample
[requirements](./requirements.txt) file which can be used for the environment creation.

YAMNet also requires the following data file (which is provided in the cloned repository):

- [YAMNet model weights][4] in Keras saved weights in HDF5 format.

After downloading this file into the same directory as this README, the
installation can be tested by running:

```shell
python yamnet_test.py
```

which runs some synthetic signals through the model and checks the outputs.

Here's a sample installation and test session:

```shell
# Upgrade pip first. Also make sure wheel is installed.
python -m pip install --upgrade pip wheel

# Install dependencies.
pip install numpy resampy tensorflow soundfile

# Clone TensorFlow models repo into a 'models' directory.
git clone https://github.com/tensorflow/models.git
cd models/research/audioset/yamnet
# Download data file into same directory as code
# Note: you can skip this if desired, it is provided within the repository
curl -O https://storage.googleapis.com/audioset/yamnet.h5

# Installation ready, let's test it.
python yamnet_test.py
# If we see "Ran 4 tests ... OK ...", then we're all set.
```

## Usage

In order to use our audio classification tool we need to ensure that the files are in the correct location. To do so,
ensure that the `covid19_data_0426.zip` is downloaded from Google Drive (upon request). After that its contents need to
be extracted to `./covid19_data_0426_flatten`. Now, assuming YamNet is installed successfully and the target virtual
environment activated we need to execute:

```shell
python ./inference_save.py
```

Which will create the intermediate file `covid19_data_0426_yamnet_all.list`.

Then, we can do:

```shell
python ./prediction_via_inference.py
```

Moreover, you can run over existing sound files using inference.py:

```shell
python inference.py input_sound.wav
```

The code will report the top-5 highest-scoring classes averaged over all the frames of the input. You can access
greater detail by modifying the example code in inference.py. See the jupyter notebook [yamnet_visualization.ipynb][6]
for an example of displaying the per-frame model output scores.

## About the Model

The YAMNet code layout is as follows:

- [`yamnet.py`](./yamnet.py): Model definition in Keras.
- [`params.py`](./params.py): Model Hyperparameters. You can usefully modify PATCH_HOP_SECONDS.
- [`features.py`](./features.py): Audio feature extraction helpers.
- [`inference.py`](./inference.py): Example code to classify input wav files.
- [`yamnet_test.py`](./yamnet_test.py): Simple test of YAMNet installation.
- [`inference_save.py`](./inference_save.py): The tool that generates the scores for the classification.
- [`prediction_via_inference.py`](./prediction_via_inference.py): The tool that predicts is an audio sample is viable.

### Contact information

This tool was developed by Jing Han (jh2298@cl.cam.ac.uk), Tong Xia (tx229@cl.cam.ac.uk),
Dimitris Spathis (ds809@cl.cam.ac.uk), and Andreas Grammenos (ag926@cl.cam.ac.uk).

[1]: https://www.tensorflow.org/hub/tutorials/yamnet
[2]: http://g.co/audioset
[3]: https://arxiv.org/pdf/1704.04861.pdf
[4]: https://storage.googleapis.com/audioset/yamnet.h5
[5]: https://github.com/tensorflow/models/tree/master/research/audioset/vggish
[6]: ./yamnet_visualization.ipynb
