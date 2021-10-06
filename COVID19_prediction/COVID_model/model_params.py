# coding: utf-8
# author: Tong Xia

"""
Global parameters for the model.

"""

from os.path import join as pjoin

# Training
TASK = "main"  # pre/main
AUDIO_TRAIN_NAME = "covid_model"  # train name
NUM_EPOCHS = 20
BATCH_SIZE = 1
L2 = 1e-6
PATIENCE = 15
DROPOUT_RATE = 0.5
LEARNING_RATE_DECAY = 0.98
SR_VGG = 16000  # VGG pretrained model sample rate
EARLY_STOP = "AUC"

# Architectural consants.
EMBEDDING_SIZE = 128  # Size of embedding layer.
NUM_CLASSES = 2
NUM_SYMPTOMS = 13
TRAINED_LAYERS = 18

INIT_STDDEV = 0.01  # Standard deviation used to initialize weights.
LEARNING_RATE = 1e-5  # Learning rate for the Adam optimizer.
ADAM_EPSILON = 1e-8  # Epsilon for the Adam optimizer.
NUM_UNITS = 40  # hidden units
PREVALANCE = 0.5  # the percentage of covid in the population


MODALITY = "BCV"
ISCOMB = False
ISAUG = False
MODFUSE = "concat"

# Data
TF_DATA_DIR = "../data/"
DATA_NAME = "data/audio_0426En"
IS_ADDDATA = False

# Checkpoint,no need to change
TENSORBOARD_DIR = "../data/tensorboard"  # Tensorboard
AUDIO_CHECKPOINT_DIR = "../data/train"
AUDIO_CHECKPOINT_NAME = "mymodel.ckpt"

# Vggish
VGGISH_CHECKPOINT_DIR = "../vggish"
VGGISH_CHECKPOINT_NAME = "vggish_model.ckpt"
VGGISH_PCA_PARAMS_NAME = "vggish_pca_params.npz"
VGGISH_CHECKPOINT = pjoin(VGGISH_CHECKPOINT_DIR, VGGISH_CHECKPOINT_NAME)
VGGISH_PCA_PARAMS = pjoin(VGGISH_CHECKPOINT_DIR, VGGISH_PCA_PARAMS_NAME)

VGGISH_INPUT_OP_NAME = "vggish/input_features"
VGGISH_INPUT_TENSOR_NAME = VGGISH_INPUT_OP_NAME + ":0"
VGGISH_OUTPUT_OP_NAME = "vggish/embedding"
VGGISH_OUTPUT_TENSOR_NAME = VGGISH_OUTPUT_OP_NAME + ":0"
